"""Attack D (Section 12.5): bilevel adversarial fine-tune against the verifier.

Concept
-------

The attacker simulates the verifier's full pipeline (latent MAP recovery
+ cosine / WMF scoring against a key bank) inside the fine-tune loop:

  inner:  z_hat(theta')  =  argmin_z  || F_{theta'}(z; o) - a ||^2
                                   + lambda_z * || z ||^2
  outer:  L  =  L_task(theta')  +  lambda_attack * KeyBankPenalty(z_hat)

KeyBankPenalty pushes the recovered noise to have no large cosine with
any element of an attacker-generated false-key bank. The owner key was
drawn from the same generative distribution (band-passed Gaussian /
Gaussian) so it is statistically a member of that bank --- by symmetry,
its cosine should also collapse.

Because the verifier's MAP is itself an optimization, we backprop through
a **truncated unroll** of M Adam iterations (M small, e.g. 6-10). Each
unroll step holds the action expert autograd graph, so memory scales
linearly with M.

This is more expensive than Attack C but assumes only the public
algorithm; it does not require having a subspace estimate.

Usage::

    python scripts/attacks/finetune_attack_d.py \
        --config-name pi05_libero90_from_libero \
        --pytorch-weight-path /workspace/models/openpi-cache/openpi-assets/checkpoints/pi05_libero \
        --save-dir /path/to/attack_d_ckpts \
        --num-train-steps 1500 --batch-size 8 \
        --inner-iters 6 --inner-lr 0.1 --inner-prior-weight 1.0 \
        --num-false-keys 32 --inv-num-denoising-steps 4 \
        --lambda-attack 1.0 --reference-mode gaussian
"""

from __future__ import annotations

import os

# JAX is only used here for jax.tree.map on torch tensors; pin it to CPU so it
# does not race torch for CUDA initialization (segfaults otherwise).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
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
import openpi.policies.watermark as wm
import openpi.training.config as _config
import openpi.training.data_loader as _data


# ---------------------------------------------------------------------------
# Prefix builder shared with Attack C
# ---------------------------------------------------------------------------


def _build_prefix_inputs(model: pi0_pytorch.PI0Pytorch, observation):
    ctx = torch.no_grad()
    with ctx:
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
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
# Differentiable MAP unroll
# ---------------------------------------------------------------------------


def unrolled_map(
    model: pi0_pytorch.PI0Pytorch,
    model_inputs,
    y_obs: torch.Tensor,
    *,
    time_grid: torch.Tensor,
    num_iters: int,
    lr: float,
    prior_weight: float,
    obs_sigma: float,
    z_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unrolled Adam on the MAP objective; returns z_hat with grad through `model`."""
    bsize, horizon, _ = y_obs.shape
    raw_dim = int(getattr(model, "action_dim", y_obs.shape[-1]))
    device = y_obs.device
    dtype = y_obs.dtype

    if z_init is None:
        z = torch.randn(bsize, horizon, raw_dim, device=device, dtype=dtype)
    else:
        z = z_init.to(device=device, dtype=dtype)

    # Manually-unrolled Adam (so the optimizer state participates in autograd).
    m = torch.zeros_like(z)
    v = torch.zeros_like(z)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    sigma2 = max(obs_sigma * obs_sigma, 1e-12)

    for t in range(1, int(num_iters) + 1):
        z = z.requires_grad_(True)
        a_pred = model.sample_actions_from_noise(model_inputs, z, time_grid)
        obs_loss = 0.5 * ((a_pred - y_obs).pow(2).mean()) / sigma2
        prior_loss = 0.5 * float(prior_weight) * z.pow(2).mean()
        loss = obs_loss + prior_loss
        (g,) = torch.autograd.grad(loss, z, create_graph=True)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g.pow(2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        z = z - float(lr) * m_hat / (v_hat.sqrt() + adam_eps)

    return z


# ---------------------------------------------------------------------------
# False-key bank
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KeyBank:
    references: torch.Tensor  # [K, T, D]
    secret_keys: list[int]

    @classmethod
    def build(
        cls,
        *,
        num_keys: int,
        horizon: int,
        action_dim: int,
        sample_rate_hz: float,
        reference_mode: str,
        watermark_dims: tuple[int, ...] | None,
        seed_offset: int,
        episode_nonce: int,
        chunk_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KeyBank":
        refs = []
        keys: list[int] = []
        rng = np.random.default_rng(seed_offset)
        for _ in range(int(num_keys)):
            secret_key = int(rng.integers(low=1, high=2**31 - 1))
            cfg = wm.InternalNoiseWatermarkConfig(
                secret_key=secret_key,
                control_freq=sample_rate_hz,
                reference_mode=reference_mode,
                watermark_dims=watermark_dims,
            )
            ref = wm.generate_keyed_reference(
                length=horizon,
                action_dim=action_dim,
                sample_rate_hz=sample_rate_hz,
                config=cfg,
                context=wm.WatermarkContext(chunk_index=chunk_index, episode_nonce=episode_nonce),
            )
            refs.append(ref)
            keys.append(secret_key)
        stacked = np.stack(refs, axis=0).astype(np.float32)  # [K, T, D]
        return cls(references=torch.as_tensor(stacked, device=device, dtype=dtype), secret_keys=keys)


# ---------------------------------------------------------------------------
# Outer attack loss
# ---------------------------------------------------------------------------


def keybank_penalty(z_hat: torch.Tensor, bank: KeyBank, *, temp: float = 10.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Soft-max over false-key cosine^2; pushes z_hat to be non-aligned with any candidate key."""
    if z_hat.shape[1:] != bank.references.shape[1:]:
        raise ValueError(
            f"z_hat trailing shape {tuple(z_hat.shape[1:])} != bank refs {tuple(bank.references.shape[1:])}"
        )
    z_flat = z_hat.flatten(1).float()  # [B, T*D]
    r_flat = bank.references.flatten(1).float()  # [K, T*D]
    z_norm = z_flat / (z_flat.norm(dim=-1, keepdim=True) + 1e-8)
    r_norm = r_flat / (r_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cos = z_norm @ r_norm.T  # [B, K]
    sq = cos.pow(2)  # symmetric in sign
    # Smoothed maximum over the bank.
    soft_max = torch.logsumexp(sq * float(temp), dim=-1) / float(temp)
    loss = soft_max.mean()
    info = {
        "cos_max_mean": float(cos.abs().max(dim=-1).values.mean().item()),
        "cos_mean": float(cos.abs().mean().item()),
        "sq_soft_max": float(soft_max.mean().item()),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Helpers
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
    n = 0
    for name, param in model.named_parameters():
        if "paligemma" in name and "gemma_expert" not in name:
            param.requires_grad_(False)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--pytorch-weight-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8, help="Both task and attack batch size.")
    parser.add_argument("--attack-batch-size", type=int, default=None,
                        help="Override sub-batch size for the bilevel attack loss (defaults to --batch-size).")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)

    # Attack-side params.
    parser.add_argument("--lambda-attack", type=float, default=1.0)
    parser.add_argument("--inner-iters", type=int, default=6,
                        help="Unrolled MAP iterations. Each step keeps the action expert graph in memory.")
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--inner-prior-weight", type=float, default=1.0)
    parser.add_argument("--inner-obs-sigma", type=float, default=1e-2)
    parser.add_argument("--inv-num-denoising-steps", type=int, default=4,
                        help="Denoising steps used by sample_actions_from_noise inside MAP.")

    # False-key bank.
    parser.add_argument("--num-false-keys", type=int, default=32)
    parser.add_argument("--reference-mode", choices=("gaussian", "bandpass"), default="gaussian")
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--telemetry-dim", type=int, default=None,
                        help="If set, restrict watermark_dims to range(telemetry_dim); else span the full raw dim.")
    parser.add_argument("--key-bank-seed", type=int, default=0)
    parser.add_argument("--rotate-keys-every", type=int, default=50,
                        help="Re-sample the false-key bank every N steps (avoid overfitting to one bank).")
    parser.add_argument("--soft-max-temp", type=float, default=10.0)

    parser.add_argument("--freeze-paligemma", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_seed(args.seed)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    # ------------------------------------------------------------------
    # Build model
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

    object.__setattr__(train_config, "batch_size", int(args.batch_size))

    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    weights_path = pathlib.Path(args.pytorch_weight_path) / "model.safetensors"
    safetensors.torch.load_model(model, str(weights_path))
    logging.info(f"Loaded suspect (watermarked) weights from {weights_path}")

    if args.freeze_paligemma:
        n = _freeze_paligemma(model)
        logging.info(f"Froze {n} paligemma params; only action expert trains.")

    raw_dim = int(getattr(model, "action_dim", model_cfg.action_dim))
    horizon = int(model_cfg.action_horizon)
    watermark_dims = tuple(range(int(args.telemetry_dim))) if args.telemetry_dim else None

    loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Initial false-key bank.
    bank = KeyBank.build(
        num_keys=args.num_false_keys,
        horizon=horizon,
        action_dim=raw_dim,
        sample_rate_hz=args.sample_rate_hz,
        reference_mode=args.reference_mode,
        watermark_dims=watermark_dims,
        seed_offset=args.key_bank_seed,
        episode_nonce=0,
        chunk_index=0,
        device=device,
        dtype=train_dtype,
    )
    logging.info(f"Built false-key bank: {bank.references.shape}")

    log_buffer: list[dict] = []
    pbar = tqdm.tqdm(total=args.num_train_steps, desc="attack-D")
    step = 0
    start = time.time()

    while step < args.num_train_steps:
        for observation, actions in loader:
            if step >= args.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            optim.zero_grad(set_to_none=True)

            # ---- L_task --------------------------------------------------
            losses = model(observation, actions)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            l_task = losses.mean()

            # ---- Attack: bilevel ----------------------------------------
            atk_obs = observation
            atk_actions = actions
            if args.attack_batch_size is not None and args.attack_batch_size < int(actions.shape[0]):
                k = int(args.attack_batch_size)
                atk_obs = jax.tree.map(lambda x: x[:k], observation)
                atk_actions = actions[:k]

            # MAP inverts on the *raw* action, padded out to raw_dim.
            B = atk_actions.shape[0]
            T = atk_actions.shape[1]
            D = raw_dim
            if atk_actions.shape[-1] < D:
                pad = torch.zeros(B, T, D - atk_actions.shape[-1], device=device, dtype=atk_actions.dtype)
                y_obs = torch.cat([atk_actions, pad], dim=-1)
            else:
                y_obs = atk_actions[..., :D]

            time_grid = _make_time_grid(args.inv_num_denoising_steps, device=device, dtype=train_dtype)
            model_inputs = _build_prefix_inputs(model, atk_obs)

            z_hat = unrolled_map(
                model,
                model_inputs,
                y_obs.to(train_dtype),
                time_grid=time_grid,
                num_iters=args.inner_iters,
                lr=args.inner_lr,
                prior_weight=args.inner_prior_weight,
                obs_sigma=args.inner_obs_sigma,
            )

            l_attack, attack_info = keybank_penalty(z_hat, bank, temp=args.soft_max_temp)

            # ---- combined backward --------------------------------------
            l_total = l_task + float(args.lambda_attack) * l_attack
            l_total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.clip_grad)
            optim.step()

            log_entry = {
                "step": step,
                "l_task": float(l_task.detach().item()),
                "l_attack": float(l_attack.detach().item()),
                "l_total": float(l_total.detach().item()),
                "grad_norm": float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                **{f"atk_{k}": v for k, v in attack_info.items()},
                "elapsed_s": float(time.time() - start),
            }
            log_buffer.append(log_entry)

            if step % args.log_every == 0:
                logging.info(
                    "step=%d  L_task=%.4f  L_attack=%.4f  cos_max=%.3f  cos_mean=%.3f  |g|=%.3f",
                    step, log_entry["l_task"], log_entry["l_attack"],
                    log_entry["atk_cos_max_mean"], log_entry["atk_cos_mean"], log_entry["grad_norm"],
                )
                with metrics_path.open("a") as f:
                    for entry in log_buffer:
                        f.write(json.dumps(entry) + "\n")
                log_buffer.clear()

            if (step > 0 and step % args.save_every == 0) or step == args.num_train_steps - 1:
                _save_checkpoint(model, save_dir, step)
                logging.info(f"Saved checkpoint at step {step}")

            # Rotate the false-key bank periodically so we don't overfit.
            if args.rotate_keys_every > 0 and step > 0 and step % args.rotate_keys_every == 0:
                bank = KeyBank.build(
                    num_keys=args.num_false_keys,
                    horizon=horizon,
                    action_dim=raw_dim,
                    sample_rate_hz=args.sample_rate_hz,
                    reference_mode=args.reference_mode,
                    watermark_dims=watermark_dims,
                    seed_offset=args.key_bank_seed + step,
                    episode_nonce=0,
                    chunk_index=0,
                    device=device,
                    dtype=train_dtype,
                )

            step += 1
            pbar.update(1)

    if log_buffer:
        with metrics_path.open("a") as f:
            for entry in log_buffer:
                f.write(json.dumps(entry) + "\n")

    _save_checkpoint(model, save_dir, step)
    pbar.close()
    logging.info("Attack D fine-tune complete.")


if __name__ == "__main__":
    main()
