"""Attack D (direct adversarial fine-tune against the verifier) — JAX-native.

NO subspace estimate. The attacker simulates the verifier's own pipeline
(latent MAP recovery -> matched-filter / cosine score against a false-key
bank) INSIDE the fine-tune loop and trains the policy to drive that score to
zero while keeping task competence:

    inner:  z_hat(theta')  =  argmin_z  0.5*||F_{theta'}(z; o) - a||^2 / sigma^2
                                       + 0.5*prior * ||z||^2          (unrolled Adam)
    outer:  L  =  L_task(theta')  +  lambda_attack * KeyBankPenalty(z_hat)

KeyBankPenalty pushes the recovered noise to have no large (squared) cosine
with any element of a false-key bank. The owner key was drawn from the same
generative distribution, so by symmetry its matched-filter score collapses
too. We backprop through a truncated unroll of the inner MAP (jax.grad is
composable, so the outer grad w.r.t. theta flows through the inner grad).

This is the JAX analogue of scripts/attacks/finetune_attack_d.py (torch). It
slots into the existing scripts/train.py infrastructure (FSDP mesh, orbax
checkpointing, LoRA freeze filter) by hot-swapping `train_module.train_step`,
exactly like finetune_attack_c_jax.py, so it can run on top of any registered
TrainConfig (e.g. pi05_libero_10_lora_from_libero, whose LiberoHdf5DataConfig
reads the LIBERO-10 hdf5 directly).

Usage::

    python scripts/attacks/finetune_attack_d_jax.py \
        --config-name pi05_libero_10_lora_from_libero \
        --exp-name attack_d_lam1 \
        --lambda-attack 1.0 \
        --inner-iters 4 --inner-lr 0.1 --inner-prior-weight 1.0 --inner-obs-sigma 1e-2 \
        --inv-num-denoising-steps 4 \
        --num-false-keys 32 --reference-mode gaussian --sample-rate-hz 20.0 \
        --attack-batch-size 8 --soft-max-temp 10.0 \
        --batch-size 32 --num-train-steps 2000 --fsdp-devices 1 \
        --checkpoint-base-dir /workspace/vla_out/attack_c/attacked
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# scripts/train.py uses these top-level names; we hot-swap its train_step.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import train as train_module  # noqa: E402  (scripts/train.py)

import openpi.models.model as _model  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.utils as training_utils  # noqa: F401,E402  (used by train_module)
from openpi.models import gemma as _gemma  # noqa: F401,E402  (used by sample_actions_from_noise)
from openpi.policies import watermark as wm  # noqa: E402


# ---------------------------------------------------------------------------
# False-key bank (host-side numpy -> constant jnp array)
# ---------------------------------------------------------------------------


def build_key_bank(
    *,
    num_keys: int,
    horizon: int,
    action_dim: int,
    sample_rate_hz: float,
    reference_mode: str,
    watermark_dims: tuple[int, ...] | None,
    seed: int,
) -> jnp.ndarray:
    """Return a constant [K, T, D] bank of keyed references for random false keys."""
    rng = np.random.default_rng(seed)
    refs = []
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
            context=wm.WatermarkContext(chunk_index=0, episode_nonce=0),
        )
        refs.append(np.asarray(ref, dtype=np.float32))
    stacked = np.stack(refs, axis=0).astype(np.float32)  # [K, T, D]
    return jnp.asarray(stacked, dtype=jnp.float32)


def keybank_penalty(z_hat: jnp.ndarray, bank: jnp.ndarray, *, temp: float):
    """Soft-max over false-key cosine^2; pushes z_hat off every candidate key."""
    z_flat = z_hat.reshape(z_hat.shape[0], -1).astype(jnp.float32)  # [B, T*D]
    r_flat = bank.reshape(bank.shape[0], -1).astype(jnp.float32)  # [K, T*D]
    z_norm = z_flat / (jnp.linalg.norm(z_flat, axis=-1, keepdims=True) + 1e-8)
    r_norm = r_flat / (jnp.linalg.norm(r_flat, axis=-1, keepdims=True) + 1e-8)
    cos = z_norm @ r_norm.T  # [B, K]
    sq = jnp.square(cos)  # sign-symmetric
    soft_max = jax.nn.logsumexp(sq * float(temp), axis=-1) / float(temp)  # [B]
    loss = jnp.mean(soft_max)
    info = {
        "cos_max_mean": jnp.mean(jnp.max(jnp.abs(cos), axis=-1)),
        "cos_mean": jnp.mean(jnp.abs(cos)),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Custom train step (closes over the key bank + attack hyperparameters)
# ---------------------------------------------------------------------------


def build_train_step(
    bank: jnp.ndarray,
    *,
    lambda_attack: float,
    inner_iters: int,
    inner_lr: float,
    inner_prior_weight: float,
    inner_obs_sigma: float,
    inv_num_denoising_steps: int,
    soft_max_temp: float,
    attack_batch_size: int | None,
    inject_beta: float,
    score_dims: int,
):
    raw_D = int(bank.shape[2])
    sdims = int(score_dims) if score_dims and score_dims > 0 else raw_D
    inv_steps = int(inv_num_denoising_steps)
    n_inner = int(inner_iters)
    in_lr = float(inner_lr)
    prior_w = float(inner_prior_weight)
    sigma2 = max(float(inner_obs_sigma) ** 2, 1e-12)
    lam = float(lambda_attack)
    temp = float(soft_max_temp)
    atk_bs = attack_batch_size
    inj_beta = float(inject_beta)
    inj_alpha = float(np.sqrt(max(0.0, 1.0 - inj_beta * inj_beta)))

    def _prefix_inputs(model, observation):
        # One paligemma forward to build the prefix KV cache shared by the unroll.
        from openpi.models.pi0 import make_attn_mask  # local import to dodge cycle

        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        return (observation, prefix_mask, kv_cache)

    def _unrolled_map(model, model_inputs, y_obs, rng):
        """Manually-unrolled Adam on the MAP objective; z_hat keeps grad through model."""
        bsize = y_obs.shape[0]
        horizon = int(model.action_horizon)
        time_grid = jnp.linspace(1.0, 0.0, inv_steps + 1)
        z = jax.random.normal(rng, (bsize, horizon, raw_D), dtype=y_obs.dtype)

        m = jnp.zeros_like(z)
        v = jnp.zeros_like(z)
        b1, b2, adam_eps = 0.9, 0.999, 1e-8

        def inner_obj(z_):
            a_pred = model.sample_actions_from_noise(model_inputs, z_, time_grid)
            obs_loss = 0.5 * jnp.mean(jnp.square(a_pred - y_obs)) / sigma2
            prior_loss = 0.5 * prior_w * jnp.mean(jnp.square(z_))
            return obs_loss + prior_loss

        for t in range(1, n_inner + 1):
            g = jax.grad(inner_obj)(z)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * jnp.square(g)
            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)
            z = z - in_lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)
        return z

    def _attack_loss(model, rng, observation):
        """Direct adversarial: inject a keyed reference into the sampling noise the way
        the verifier does (mix_internal_noise: z_inj = alpha*z + beta*r), roll the model
        forward, run the verifier's MAP recovery, and penalize the matched-filter score
        cos(z_hat, r) of the recovered noise with the injected key. Driving this to zero
        is exactly defeating the verifier's detection statistic."""
        inv_obs = _model.preprocess_observation(None, observation, train=False)
        model_inputs = _prefix_inputs(model, inv_obs)
        bsize = inv_obs.state.shape[0]
        horizon = int(model.action_horizon)
        k_rng, z_rng, map_rng = jax.random.split(rng, 3)

        # One random keyed reference per batch element (rotates the key every step).
        idx = jax.random.randint(k_rng, (bsize,), 0, bank.shape[0])
        r = bank[idx]  # [B, T, D]
        z_base = jax.random.normal(z_rng, (bsize, horizon, raw_D), dtype=r.dtype)
        z_inj = inj_alpha * z_base + inj_beta * r  # norm-preserving injection

        time_grid = jnp.linspace(1.0, 0.0, inv_steps + 1)
        a_wm = model.sample_actions_from_noise(model_inputs, z_inj, time_grid)
        z_hat = _unrolled_map(model, model_inputs, a_wm, map_rng)

        # Penalize the verifier's ACTUAL per-episode statistic: the DECOY-CENTERED
        # matched filter on work-7 dims (0..sdims-1), no subspace/whitening --
        #   st_b = cos(z_hat_b, r_true_b) - mean_{k != true} cos(z_hat_b, r_decoy_k)
        # which is exactly asc.raw_score(tv, nm) the deployed detector thresholds.
        # (Driving the bare true-key cosine to 0 is not enough: detection reads the
        # true-vs-decoy GAP, which is why it survives at near-chance absolute cosine.)
        zf = z_hat[:, :, :sdims].reshape(bsize, -1).astype(jnp.float32)        # [B, T*sdims]
        bf = bank[:, :, :sdims].reshape(bank.shape[0], -1).astype(jnp.float32)  # [K, T*sdims]
        zf = zf / (jnp.linalg.norm(zf, axis=-1, keepdims=True) + 1e-8)
        bf = bf / (jnp.linalg.norm(bf, axis=-1, keepdims=True) + 1e-8)
        cos_all = zf @ bf.T                                                    # [B, K]
        K = int(bank.shape[0])
        true_cos = jnp.take_along_axis(cos_all, idx[:, None], axis=1)[:, 0]    # [B]
        decoy_mean = (jnp.sum(cos_all, axis=1) - true_cos) / max(K - 1, 1)     # [B]
        centered = true_cos - decoy_mean                                       # [B] verifier raw score
        mean_centered = jnp.mean(centered)
        loss = jnp.square(mean_centered)
        info = {"cos_max_mean": mean_centered, "cos_mean": jnp.mean(true_cos)}
        return loss, info

    def train_step(config, rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, observation, actions):
            task_rng, atk_rng = jax.random.split(rng)
            chunked = model.compute_loss(task_rng, observation, actions, train=True)
            l_task = jnp.mean(chunked)
            if lam > 0.0:
                if atk_bs is not None and atk_bs < int(actions.shape[0]):
                    atk_obs = jax.tree.map(lambda x: x[:atk_bs], observation)
                else:
                    atk_obs = observation
                l_attack, info = _attack_loss(model, atk_rng, atk_obs)
                total = l_task + lam * l_attack
            else:
                l_attack = jnp.float32(0.0)
                info = {"cos_max_mean": jnp.float32(0.0), "cos_mean": jnp.float32(0.0)}
                total = l_task
            return total, {"l_task": l_task, "l_attack": l_attack, **info}

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, config.trainable_filter)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(
            state, step=state.step + 1, params=new_params, opt_state=new_opt_state
        )
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        info = {
            "loss": loss,
            "l_task": aux["l_task"],
            "l_attack": aux["l_attack"],
            "cos_max_mean": aux["cos_max_mean"],
            "cos_mean": aux["cos_mean"],
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
        }
        return new_state, info

    return train_step


# ---------------------------------------------------------------------------
# CLI + config plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True, help="Name of a registered TrainConfig.")
    p.add_argument("--exp-name", required=True, help="Experiment name; builds checkpoint_dir.")
    # Attack-side params.
    p.add_argument("--lambda-attack", type=float, default=1.0)
    p.add_argument("--inner-iters", type=int, default=4)
    p.add_argument("--inner-lr", type=float, default=0.1)
    p.add_argument("--inner-prior-weight", type=float, default=1.0)
    p.add_argument("--inner-obs-sigma", type=float, default=1e-2)
    p.add_argument("--inv-num-denoising-steps", type=int, default=4)
    p.add_argument("--num-false-keys", type=int, default=32)
    p.add_argument("--reference-mode", choices=("gaussian", "bandpass"), default="gaussian")
    p.add_argument("--sample-rate-hz", type=float, default=20.0)
    p.add_argument("--telemetry-dim", type=int, default=None,
                   help="If set, restrict watermark_dims to range(telemetry_dim); else full raw dim.")
    p.add_argument("--key-bank-seed", type=int, default=0)
    p.add_argument("--soft-max-temp", type=float, default=10.0)
    p.add_argument("--attack-batch-size", type=int, default=8)
    p.add_argument("--inject-beta", type=float, default=1.0,
                   help="Watermark injection strength used during the attack (deployment uses 1.0).")
    p.add_argument("--score-dims", type=int, default=7,
                   help="Restrict the matched-filter penalty to dims 0..N-1 (work-7 detector uses 0-6).")
    # Training overrides.
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-train-steps", type=int, default=None)
    p.add_argument("--fsdp-devices", type=int, default=None)
    p.add_argument("--checkpoint-base-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_known_args(argv)


def main(argv=None) -> int:
    train_module.init_logging()
    args, _ = _parse_args(sys.argv[1:] if argv is None else argv)

    base = _config.get_config(args.config_name)
    overrides: dict = {"wandb_enabled": False}
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.num_train_steps is not None:
        overrides["num_train_steps"] = int(args.num_train_steps)
    if args.fsdp_devices is not None:
        overrides["fsdp_devices"] = int(args.fsdp_devices)
    if args.checkpoint_base_dir is not None:
        overrides["checkpoint_base_dir"] = str(args.checkpoint_base_dir)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.save_interval is not None:
        overrides["save_interval"] = int(args.save_interval)
    if args.log_interval is not None:
        overrides["log_interval"] = int(args.log_interval)
    if args.overwrite:
        overrides["overwrite"] = True
    if args.resume:
        overrides["resume"] = True
    if hasattr(base, "exp_name"):
        overrides["exp_name"] = args.exp_name
    else:
        overrides["name"] = f"{base.name}/{args.exp_name}"

    config = dataclasses.replace(base, **overrides)
    logging.info("Attack-D JAX fine-tune config:\n%s", config)

    horizon = int(config.model.action_horizon)
    raw_D = int(config.model.action_dim)
    wm_dims = tuple(range(int(args.telemetry_dim))) if args.telemetry_dim else None
    bank = build_key_bank(
        num_keys=args.num_false_keys,
        horizon=horizon,
        action_dim=raw_D,
        sample_rate_hz=args.sample_rate_hz,
        reference_mode=args.reference_mode,
        watermark_dims=wm_dims,
        seed=args.key_bank_seed,
    )
    logging.info("Built false-key bank: %s (lambda_attack=%s)", tuple(bank.shape), args.lambda_attack)

    custom_train_step = build_train_step(
        bank,
        lambda_attack=args.lambda_attack,
        inner_iters=args.inner_iters,
        inner_lr=args.inner_lr,
        inner_prior_weight=args.inner_prior_weight,
        inner_obs_sigma=args.inner_obs_sigma,
        inv_num_denoising_steps=args.inv_num_denoising_steps,
        soft_max_temp=args.soft_max_temp,
        attack_batch_size=args.attack_batch_size,
        inject_beta=args.inject_beta,
        score_dims=args.score_dims,
    )
    # Hot-swap before train_module.main captures it via jax.jit.
    train_module.train_step = custom_train_step
    train_module.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
