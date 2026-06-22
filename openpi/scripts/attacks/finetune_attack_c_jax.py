"""Attack C (§12.5) — JAX-native subspace-targeted invariance fine-tune.

Loads the subspace `P_K` produced by `scripts/attacks/estimate_wm_subspace.py`,
then runs the existing `scripts/train.py` infrastructure (FSDP mesh, orbax
checkpointing, LoRA freeze filter via `pi05_libero_low_mem_finetune`) with a
monkey-patched `train_step` whose loss is::

    L = L_task(θ') + λ * E_{z,ε} [ ||F(z + P_K ε) - F(z)||² / σ² ]

The invariance term shares one prefix KV cache between the clean and
perturbed forwards, so the marginal cost is two suffix-only
`sample_actions_from_noise` calls per training step.

Usage::

    python scripts/attacks/finetune_attack_c_jax.py \
        --config-name pi05_libero_low_mem_finetune \
        --exp-name attack_c_lam1_r8 \
        --subspace-path /workspace/vla/attack_c_data/subspace/wm_subspace_r8.npz \
        --lambda-inv 1.0 \
        --inv-num-denoising-steps 4 \
        --inv-eps-sigma 0.5 \
        --batch-size 128 \
        --num-train-steps 2000 \
        --fsdp-devices 8 \
        --checkpoint-base-dir /workspace/vla/attack_c_data/attacked
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# scripts/train.py uses these top-level names; we'll patch its train_step.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import train as train_module  # noqa: E402  (scripts/train.py)

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.utils as training_utils  # noqa: F401  (used by train_module)
from openpi.models import gemma as _gemma  # noqa: F401  (used by sample_actions_from_noise)


# ---------------------------------------------------------------------------
# Subspace
# ---------------------------------------------------------------------------


def _load_subspace(path: str) -> jnp.ndarray:
    """Load `components` (rows are basis vectors) as a `[k, D]` jnp array."""
    data = np.load(path, allow_pickle=True)
    comps = np.asarray(data["components"], dtype=np.float32)
    if comps.ndim != 2:
        raise ValueError(f"components must be [k, D], got {comps.shape}")
    logging.info(
        "Loaded subspace from %s: rank=%d D=%d top eig=%.4f",
        path,
        comps.shape[0],
        comps.shape[1],
        float(np.asarray(data["singular_values"])[0]) ** 2,
    )
    return jnp.asarray(comps, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Custom train step (closes over P_K and attack hyperparameters)
# ---------------------------------------------------------------------------


def build_train_step(
    P_K: jnp.ndarray,
    *,
    lambda_inv: float,
    inv_num_denoising_steps: int,
    inv_eps_sigma: float,
):
    """Return a `train_step` with the invariance loss baked in.

    The returned function has the same signature as `scripts.train.train_step`,
    so it can be hot-swapped via `train_module.train_step = ...` before the
    `jax.jit(functools.partial(train_step, config), ...)` capture in `main`.
    """

    raw_D = int(P_K.shape[1])
    inv_steps = int(inv_num_denoising_steps)
    eps_sigma = float(inv_eps_sigma)
    eps_sigma_sq = max(eps_sigma * eps_sigma, 1e-12)
    lam = float(lambda_inv)

    def _invariance_loss(model, rng, observation):
        # 1) Build prefix KV cache (one paligemma forward).
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
        from openpi.models.pi0 import make_attn_mask  # local import to dodge cycle

        attn = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn, positions=positions
        )
        model_inputs = (observation, prefix_mask, kv_cache)

        bsize = observation.state.shape[0]
        action_horizon = int(model.action_horizon)
        z_rng, eps_rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (bsize, action_horizon, raw_D))
        eps = eps_sigma * jax.random.normal(eps_rng, z.shape)
        coords = jnp.einsum("btD,kD->btk", eps, P_K)
        delta = jnp.einsum("btk,kD->btD", coords, P_K)

        # Static start=1.0, end=0.0; inv_steps is a Python int → compile-time constant.
        time_grid = jnp.linspace(1.0, 0.0, inv_steps + 1)
        a_clean = model.sample_actions_from_noise(model_inputs, z, time_grid)
        a_pert = model.sample_actions_from_noise(model_inputs, z + delta, time_grid)
        l_inv = jnp.mean(jnp.square(a_pert - a_clean)) / eps_sigma_sq
        return l_inv

    def train_step(
        config,
        rng,
        state,
        batch,
    ):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, observation, actions):
            task_rng, inv_rng = jax.random.split(rng)
            chunked = model.compute_loss(task_rng, observation, actions, train=True)
            l_task = jnp.mean(chunked)
            if lam > 0.0:
                # Inference-time preprocess so observation shapes match the path
                # used by sample_actions / sample_actions_from_noise.
                inv_obs = _model.preprocess_observation(None, observation, train=False)
                l_inv = _invariance_loss(model, inv_rng, inv_obs)
                total = l_task + lam * l_inv
            else:
                l_inv = jnp.float32(0.0)
                total = l_task
            return total, {"l_task": l_task, "l_inv": l_inv}

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
            "l_inv": aux["l_inv"],
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
        }
        return new_state, info

    return train_step


# ---------------------------------------------------------------------------
# CLI + config plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config-name", required=True, help="Name of a registered TrainConfig.")
    parser.add_argument("--exp-name", required=True, help="Experiment name; appended to checkpoint_base_dir.")
    parser.add_argument("--subspace-path", required=True)
    parser.add_argument("--lambda-inv", type=float, default=1.0)
    parser.add_argument("--inv-num-denoising-steps", type=int, default=4)
    parser.add_argument("--inv-eps-sigma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--checkpoint-base-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, rest = parser.parse_known_args(argv)
    return args, rest


def main(argv: list[str] | None = None) -> int:
    train_module.init_logging()

    args, _ = _parse_args(sys.argv[1:] if argv is None else argv)
    base = _config.get_config(args.config_name)

    overrides: dict = {"name": f"{base.name}/{args.exp_name}"}
    overrides["wandb_enabled"] = bool(args.wandb_enabled)  # default False unless explicitly enabled
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
    if args.wandb_enabled:
        overrides["wandb_enabled"] = True
    if args.overwrite:
        overrides["overwrite"] = True
    if args.resume:
        overrides["resume"] = True

    # `exp_name` is a property of TrainConfig used to build `checkpoint_dir`;
    # set it via a direct override field rather than the `name` munging if the
    # TrainConfig exposes that field. Defensive: support both layouts.
    if hasattr(base, "exp_name"):
        overrides["exp_name"] = args.exp_name
        overrides.pop("name", None)

    config = dataclasses.replace(base, **overrides)
    logging.info("Attack-C JAX fine-tune config:\n%s", config)

    P_K = _load_subspace(args.subspace_path)
    custom_train_step = build_train_step(
        P_K,
        lambda_inv=args.lambda_inv,
        inv_num_denoising_steps=args.inv_num_denoising_steps,
        inv_eps_sigma=args.inv_eps_sigma,
    )
    # Hot-swap the train step before train_module.main captures it via `jax.jit`.
    train_module.train_step = custom_train_step

    train_module.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
