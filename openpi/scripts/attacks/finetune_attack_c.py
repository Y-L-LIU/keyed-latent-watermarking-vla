"""Stage 2 of Attack C (Section 12.5): subspace-targeted invariance fine-tune.

Given an estimated watermark subspace `P_K \\in R^{D x k}` (from
`estimate_wm_subspace.py`), fine-tune the *suspect-side* policy so that

    F_{theta'}(z + delta; o)  approx  F_{theta'}(z; o)
    for all delta in span(P_K), z ~ N(0, I), o ~ Demos.

After fine-tune, the verifier's MAP recovery becomes unidentifiable inside
span(P_K): perturbations along the watermark direction leave the action
unchanged, so the recovered z in that subspace collapses to the prior and
its inner product with the owner reference vanishes.

Loss::

    L = L_task(theta')                                 # demo BC
      + lambda * E_{o,z,eps} [ || F(z + P_K eps) - F(z) ||^2 / sigma^2 ]

Where `eps ~ N(0, sigma^2 I)` is sampled per-timestep in raw latent space.
The target invariance is a Jacobian penalty restricted to the estimated
watermark subspace, so model expressivity outside that subspace is
preserved.

Usage (LIBERO descendant, single GPU)::

    python scripts/attacks/finetune_attack_c.py \
        --config-name pi05_libero90_from_libero \
        --pytorch-weight-path /workspace/models/openpi-cache/openpi-assets/checkpoints/pi05_libero \
        --subspace-path /path/to/wm_subspace.npz \
        --save-dir /path/to/attack_c_ckpts \
        --lambda-inv 1.0 --num-train-steps 2000 --batch-size 16 \
        --inv-num-denoising-steps 4 --inv-mc-samples 1

Outputs:
  - `model.safetensors` per save interval under `--save-dir/step_XXXX/`
  - `metrics.jsonl` with per-step task / invariance loss and grad norm.
"""

from __future__ import annotations

import os

# JAX is only used here for jax.tree.map on torch tensors; pin it to CPU so it
# does not race torch for CUDA initialization (segfaults otherwise).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import contextlib
import dataclasses
import json
import logging
import pathlib
import time

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm

# lerobot pulls in HuggingFace datasets; importing it after openpi.training.config
# segfaults this venv. Touch it early to avoid the bad init order.
import lerobot.common.datasets.lerobot_dataset  # noqa: F401

# pi0_pytorch must be imported before openpi.training.config; loading config
# first triggers a CUDA init order that segfaults torch.
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data


# ---------------------------------------------------------------------------
# Subspace bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Subspace:
    """Orthonormal basis P_K of shape [k, D] living in raw-latent space."""

    components: torch.Tensor  # [k, D]
    mean: torch.Tensor  # [D]
    rank: int
    raw_dim: int

    @classmethod
    def load(cls, path: str | pathlib.Path, device: torch.device) -> "Subspace":
        data = np.load(path, allow_pickle=True)
        components = torch.as_tensor(np.asarray(data["components"], dtype=np.float32), device=device)
        mean = torch.as_tensor(np.asarray(data["mean"], dtype=np.float32), device=device)
        if components.ndim != 2:
            raise ValueError(f"components must be [k, D], got {tuple(components.shape)}")
        return cls(components=components, mean=mean, rank=int(components.shape[0]), raw_dim=int(components.shape[1]))

    def project_into(self, eps: torch.Tensor) -> torch.Tensor:
        """Project `eps` (..., D) onto span(P_K). Returns same shape."""
        if eps.shape[-1] != self.raw_dim:
            raise ValueError(f"Last dim {eps.shape[-1]} != subspace D={self.raw_dim}")
        # delta = (eps @ P_K^T) @ P_K  (treat components rows as basis vectors).
        # `components` is [k, D]; `eps` is [..., D].
        coords = torch.einsum("...d,kd->...k", eps, self.components)
        return torch.einsum("...k,kd->...d", coords, self.components)


# ---------------------------------------------------------------------------
# Inference-time prefix builder used by the invariance loss
# ---------------------------------------------------------------------------


def _build_prefix_inputs(
    model: pi0_pytorch.PI0Pytorch,
    observation,
    *,
    detach: bool = True,
):
    """Replicate `sample_actions` prefix wiring without gradients on paligemma.

    The action expert path receives the cached `past_key_values` as a constant.
    Returns `(state, prefix_pad_masks, past_key_values)` — the same triple that
    `sample_actions_from_noise` consumes.
    """

    ctx = torch.no_grad() if detach else contextlib.nullcontext()
    with ctx:
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # local import to avoid cycle issues

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        # Match the dtype handling in sample_actions().
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    return (state, prefix_pad_masks, past_key_values)


def _make_time_grid(num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Invariance loss
# ---------------------------------------------------------------------------


def invariance_loss(
    model: pi0_pytorch.PI0Pytorch,
    observation,
    subspace: Subspace,
    *,
    num_denoising_steps: int,
    eps_sigma: float,
    mc_samples: int,
    raw_dim: int,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    """E_{z,eps}[ || F(z + P_K eps) - F(z) ||^2 / sigma^2 ]."""
    model_inputs = _build_prefix_inputs(model, observation, detach=True)
    bsize = model_inputs[0].shape[0]
    time_grid = _make_time_grid(num_denoising_steps, device=device, dtype=dtype)

    losses = []
    a_norms = []
    delta_norms = []
    for _ in range(int(mc_samples)):
        z = torch.randn(bsize, horizon, raw_dim, dtype=dtype, device=device)
        eps = eps_sigma * torch.randn_like(z)
        delta = subspace.project_into(eps)
        z_pert = z + delta

        a_clean = model.sample_actions_from_noise(model_inputs, z, time_grid)
        a_pert = model.sample_actions_from_noise(model_inputs, z_pert, time_grid)
        diff = (a_pert - a_clean).float()
        losses.append(diff.pow(2).mean() / max(eps_sigma * eps_sigma, 1e-8))
        a_norms.append(a_clean.float().pow(2).mean().detach())
        delta_norms.append(delta.float().pow(2).mean().detach())

    loss = torch.stack(losses).mean()
    info = {
        "inv_loss": float(loss.detach().item()),
        "a_rms": float(torch.stack(a_norms).mean().sqrt().item()),
        "delta_rms": float(torch.stack(delta_norms).mean().sqrt().item()),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(model: torch.nn.Module, save_dir: pathlib.Path, step: int) -> None:
    out_dir = save_dir / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, str(out_dir / "model.safetensors"))


def _freeze_paligemma(model: pi0_pytorch.PI0Pytorch) -> int:
    """Freeze paligemma (vision + language) so only the action expert trains."""
    n_frozen = 0
    for name, param in model.named_parameters():
        if "paligemma" in name and "gemma_expert" not in name:
            param.requires_grad_(False)
            n_frozen += 1
    return n_frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Train config name (must match the suspect policy).")
    parser.add_argument("--pytorch-weight-path", required=True, help="Watermarked checkpoint dir (contains model.safetensors).")
    parser.add_argument("--subspace-path", required=True, help="Output of estimate_wm_subspace.py.")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lambda-inv", type=float, default=1.0)
    parser.add_argument("--inv-num-denoising-steps", type=int, default=4,
                        help="Steps used by sample_actions_from_noise during the invariance loss; small -> cheap.")
    parser.add_argument("--inv-eps-sigma", type=float, default=0.5,
                        help="Per-component perturbation scale in raw-latent space.")
    parser.add_argument("--inv-mc-samples", type=int, default=1)
    parser.add_argument("--inv-batch-size", type=int, default=None,
                        help="If set, use a separate (typically smaller) batch for the invariance loss. Defaults to --batch-size.")
    parser.add_argument("--freeze-paligemma", action="store_true",
                        help="Freeze vision/language tower; only train action expert. Recommended for cheap fine-tune.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_seed(args.seed)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    # ------------------------------------------------------------------
    # Build model & data loader exactly the way train_pytorch.py does.
    # ------------------------------------------------------------------
    train_config = _config.get_config(args.config_name)
    if not isinstance(train_config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=train_config.pytorch_training_precision,
            action_dim=train_config.model.action_dim,
            action_horizon=train_config.model.action_horizon,
            max_token_len=train_config.model.max_token_len,
            paligemma_variant=getattr(train_config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(train_config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(train_config.model, "pi05", False),
        )
    else:
        model_cfg = train_config.model
        object.__setattr__(model_cfg, "dtype", train_config.pytorch_training_precision)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    train_dtype = dtype_map.get(str(model_cfg.dtype), torch.float32)

    # batch_size override flows through to the data loader
    object.__setattr__(train_config, "batch_size", int(args.batch_size))

    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Load watermarked starting weights.
    weights_path = pathlib.Path(args.pytorch_weight_path) / "model.safetensors"
    safetensors.torch.load_model(model, str(weights_path))
    logging.info(f"Loaded suspect (watermarked) weights from {weights_path}")

    if args.freeze_paligemma:
        n_frozen = _freeze_paligemma(model)
        logging.info(f"Froze {n_frozen} paligemma params; only action expert trains.")

    raw_dim = int(getattr(model, "action_dim", model_cfg.action_dim))
    horizon = int(model_cfg.action_horizon)

    # Subspace.
    subspace = Subspace.load(args.subspace_path, device=device)
    if subspace.raw_dim != raw_dim:
        raise ValueError(f"Subspace D={subspace.raw_dim} != model raw_dim {raw_dim}")
    logging.info(f"Loaded subspace rank={subspace.rank} D={subspace.raw_dim}")

    # Data loader.
    loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.inv_batch_size is not None and args.inv_batch_size != args.batch_size:
        logging.info(
            f"Note: --inv-batch-size={args.inv_batch_size} differs from --batch-size; "
            "we slice the loader batch to that size for the invariance forward."
        )

    log_buffer: list[dict] = []
    pbar = tqdm.tqdm(total=args.num_train_steps, desc="attack-C")
    step = 0
    start = time.time()

    while step < args.num_train_steps:
        for observation, actions in loader:
            if step >= args.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            optim.zero_grad(set_to_none=True)

            # ---- L_task ---------------------------------------------------
            losses = model(observation, actions)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            l_task = losses.mean()

            # ---- L_inv ----------------------------------------------------
            inv_obs = observation
            if args.inv_batch_size is not None and args.inv_batch_size < int(actions.shape[0]):
                k = int(args.inv_batch_size)
                inv_obs = jax.tree.map(lambda x: x[:k], observation)
            l_inv, inv_info = invariance_loss(
                model,
                inv_obs,
                subspace,
                num_denoising_steps=args.inv_num_denoising_steps,
                eps_sigma=args.inv_eps_sigma,
                mc_samples=args.inv_mc_samples,
                raw_dim=raw_dim,
                horizon=horizon,
                device=device,
                dtype=train_dtype,
            )

            # ---- combined backward ---------------------------------------
            l_total = l_task + float(args.lambda_inv) * l_inv
            l_total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.clip_grad)
            optim.step()

            log_entry = {
                "step": step,
                "l_task": float(l_task.detach().item()),
                "l_inv": inv_info["inv_loss"],
                "l_total": float(l_total.detach().item()),
                "grad_norm": float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "a_rms": inv_info["a_rms"],
                "delta_rms": inv_info["delta_rms"],
                "elapsed_s": float(time.time() - start),
            }
            log_buffer.append(log_entry)

            if step % args.log_every == 0:
                logging.info(
                    "step=%d  L_task=%.4f  L_inv=%.4f  L_total=%.4f  |g|=%.3f  a_rms=%.3f delta_rms=%.3f",
                    step, log_entry["l_task"], log_entry["l_inv"], log_entry["l_total"],
                    log_entry["grad_norm"], log_entry["a_rms"], log_entry["delta_rms"],
                )
                with metrics_path.open("a") as f:
                    for entry in log_buffer:
                        f.write(json.dumps(entry) + "\n")
                log_buffer.clear()

            if (step > 0 and step % args.save_every == 0) or step == args.num_train_steps - 1:
                _save_checkpoint(model, save_dir, step)
                logging.info(f"Saved checkpoint at step {step}")

            step += 1
            pbar.update(1)

    if log_buffer:
        with metrics_path.open("a") as f:
            for entry in log_buffer:
                f.write(json.dumps(entry) + "\n")

    _save_checkpoint(model, save_dir, step)
    pbar.close()
    logging.info("Attack C fine-tune complete.")


if __name__ == "__main__":
    main()
