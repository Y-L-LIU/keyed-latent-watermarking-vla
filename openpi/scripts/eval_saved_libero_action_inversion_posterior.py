#!/usr/bin/env python3
"""Offline posterior replay from saved LIBERO latent MAP rollouts."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import eval_libero_action_inversion as _base  # noqa: E402
from scripts import eval_saved_libero_action_inversion as _saved  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=str, default="pi05_libero")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--rollout-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("auto", "channel", "full"), default="auto")
    parser.add_argument("--inversion-step", type=int, default=None)
    parser.add_argument("--obs-sigma", type=float, default=1e-4)
    parser.add_argument("--latent-prior-weight", type=float, default=1.0)
    parser.add_argument("--posterior-step-size", type=float, default=1e-3)
    parser.add_argument("--posterior-burnin", type=int, default=20)
    parser.add_argument("--posterior-thinning", type=int, default=10)
    parser.add_argument("--posterior-num-samples", type=int, default=8)
    parser.add_argument("--posterior-map-tether-weight", type=float, default=1.0)
    parser.add_argument("--posterior-grad-clip-norm", type=float, default=100.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.rollout_dir.exists():
        raise FileNotFoundError(f"rollout_dir does not exist: {args.rollout_dir}")
    if not args.rollout_dir.is_dir():
        raise NotADirectoryError(f"rollout_dir is not a directory: {args.rollout_dir}")
    if int(args.posterior_burnin) < 0:
        raise ValueError("posterior_burnin must be >= 0.")
    if int(args.posterior_thinning) <= 0:
        raise ValueError("posterior_thinning must be > 0.")
    if int(args.posterior_num_samples) <= 0:
        raise ValueError("posterior_num_samples must be > 0.")
    if float(args.posterior_step_size) < 0.0:
        raise ValueError("posterior_step_size must be >= 0.")
    if float(args.posterior_map_tether_weight) < 0.0:
        raise ValueError("posterior_map_tether_weight must be >= 0.")
    if float(args.posterior_grad_clip_norm) < 0.0:
        raise ValueError("posterior_grad_clip_norm must be >= 0.")
    if int(args.inversion_step) <= 0 if args.inversion_step is not None else False:
        raise ValueError("inversion_step must be > 0 when provided.")


def _saved_observation_from_trace(
    trace: _base.InversionChunkTrace,
    *,
    episode_nonce: int,
) -> dict[str, np.ndarray | str | int]:
    return {
        "prompt": str(trace.prompt),
        "observation/state": np.asarray(trace.observation_state, dtype=np.float32),
        "observation/image": np.asarray(trace.observation_image, dtype=np.uint8),
        "observation/wrist_image": np.asarray(trace.observation_wrist_image, dtype=np.uint8),
        "chunk_index": int(trace.chunk_index),
        "episode_nonce": int(episode_nonce),
    }


def _output_rollout_path(
    input_path: pathlib.Path,
    *,
    input_root: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    return output_root / input_path.relative_to(input_root)


def _resolve_replay_mode(payload: dict[str, np.ndarray], *, mode: str) -> str:
    if mode != "auto":
        return str(mode)
    if bool(np.asarray(payload.get("fm_full_latent_map", np.asarray(False))).item()):
        return "full"
    if bool(np.asarray(payload.get("fm_latent_map", np.asarray(False))).item()):
        return "channel"
    if bool(np.asarray(payload.get("fm_latent_posterior", np.asarray(False))).item()):
        return "channel"
    raise ValueError("Saved rollout does not indicate a latent MAP/posterior mode that can be replayed offline.")


def _saved_trace_chain_init_mode(trace: _base.InversionChunkTrace) -> str:
    source = str(trace.posterior_chain_init or trace.posterior_init_mode or "random")
    if source in {"map", "map_from_old_reverse"}:
        return source
    return _base._posterior_chain_init_mode(source)


def _empty_posterior_trace(trace: _base.InversionChunkTrace, *, sample_count: int) -> _base.InversionChunkTrace:
    raw_shape = tuple(int(dim) for dim in np.asarray(trace.recovered_noise, dtype=np.float32).shape)
    return dataclasses.replace(
        trace,
        posterior_recovered_noise_samples=np.zeros((int(sample_count), *raw_shape), dtype=np.float32),
        posterior_recovered_noise_mean=np.zeros(raw_shape, dtype=np.float32),
        posterior_recovered_noise_std=np.zeros(raw_shape, dtype=np.float32),
        posterior_restart_energies=np.zeros((0,), dtype=np.float32),
        posterior_best_energy=float("nan"),
        posterior_best_restart_index=-1,
        posterior_init_mode="",
        posterior_chain_init="",
    )


def _clip_pytorch_grad_by_global_norm(grad: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    if float(max_norm) <= 0.0:
        return grad
    flat = grad.reshape(grad.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    eps = torch.finfo(flat.dtype).eps
    scales = torch.clamp(float(max_norm) / torch.clamp(norms, min=eps), max=1.0)
    return grad * scales.reshape((grad.shape[0],) + (1,) * (grad.ndim - 1))


def _pytorch_energy(
    policy,
    *,
    model_inputs,
    y_obs: torch.Tensor,
    time_grid: torch.Tensor,
    z: torch.Tensor,
    z_anchor: torch.Tensor,
    mode: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    pred_actions = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)
    pred_obs = pred_actions[:, :, : y_obs.shape[-1]] if mode == "channel" else pred_actions
    pred_obs = torch.where(torch.isfinite(pred_obs), pred_obs, y_obs)
    obs_loss = 0.5 * torch.mean(torch.square((pred_obs - y_obs) / float(args.obs_sigma)))
    prior_loss = 0.5 * float(args.latent_prior_weight) * torch.mean(torch.square(z))
    tether_weight = float(getattr(args, "posterior_map_tether_weight", 0.0))
    if tether_weight > 0.0:
        tether_loss = 0.5 * tether_weight * torch.mean(torch.square(z - z_anchor))
    else:
        tether_loss = torch.zeros((), dtype=z.dtype, device=z.device)
    return obs_loss + prior_loss + tether_loss


def _run_pytorch_posterior_from_saved_map(
    policy,
    *,
    model_inputs,
    y_obs,
    time_grid,
    z_init: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    device = policy._pytorch_device
    z = torch.from_numpy(np.asarray(z_init, dtype=np.float32)).to(device=device, dtype=torch.float32)[None, ...]
    z_anchor = z.detach().clone()
    y_obs_t = torch.as_tensor(y_obs, dtype=torch.float32, device=device)
    time_grid_t = torch.as_tensor(time_grid, dtype=torch.float32, device=device)
    total_steps = int(args.posterior_burnin) + int(args.posterior_num_samples) * int(args.posterior_thinning)
    step_size = float(args.posterior_step_size)
    samples: list[torch.Tensor] = []
    for step_idx in range(total_steps):
        z = z.detach().requires_grad_(True)
        energy = _pytorch_energy(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs_t,
            time_grid=time_grid_t,
            z=z,
            z_anchor=z_anchor,
            mode=mode,
            args=args,
        )
        grad_z = torch.autograd.grad(energy, z)[0]
        grad_z = torch.nan_to_num(grad_z)
        grad_z = _clip_pytorch_grad_by_global_norm(
            grad_z,
            max_norm=float(getattr(args, "posterior_grad_clip_norm", 0.0)),
        )
        noise = torch.randn_like(z) if step_size > 0.0 else torch.zeros_like(z)
        z = z - 0.5 * step_size * grad_z + (step_size**0.5) * noise
        z = torch.nan_to_num(z.detach())
        if step_idx < int(args.posterior_burnin):
            continue
        if (step_idx - int(args.posterior_burnin)) % int(args.posterior_thinning) != 0:
            continue
        samples.append(z.clone())
    z_samples = torch.stack(samples, dim=1)
    sample_energies = []
    for sample_idx in range(z_samples.shape[1]):
        sample_energies.append(
            _pytorch_energy(
                policy,
                model_inputs=model_inputs,
                y_obs=y_obs_t,
                time_grid=time_grid_t,
                z=z_samples[:, sample_idx],
                z_anchor=z_anchor,
                mode=mode,
                args=args,
            ).detach()
        )
    energies = torch.stack(sample_energies, dim=0)
    return (
        np.asarray(z_samples[0].cpu(), dtype=np.float32),
        np.asarray(energies[:, 0].cpu(), dtype=np.float32),
    )


def _jax_energy(
    policy,
    *,
    model_inputs,
    y_obs,
    time_grid,
    z,
    z_anchor,
    mode: str,
    args: argparse.Namespace,
):
    pred_actions = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)
    pred_obs = pred_actions[:, :, : y_obs.shape[-1]] if mode == "channel" else pred_actions
    pred_obs = _base.jnp.where(_base.jnp.isfinite(pred_obs), pred_obs, y_obs)
    obs_loss = 0.5 * _base.jnp.mean(_base.jnp.square((pred_obs - y_obs) / float(args.obs_sigma)))
    prior_loss = 0.5 * float(args.latent_prior_weight) * _base.jnp.mean(_base.jnp.square(z))
    tether_weight = float(getattr(args, "posterior_map_tether_weight", 0.0))
    if tether_weight > 0.0:
        tether_loss = 0.5 * tether_weight * _base.jnp.mean(_base.jnp.square(z - z_anchor))
    else:
        tether_loss = _base.jnp.asarray(0.0, dtype=z.dtype)
    return obs_loss + prior_loss + tether_loss


def _run_jax_posterior_from_saved_map(
    policy,
    *,
    model_inputs,
    y_obs,
    time_grid,
    z_init: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    z_init_jax = _base.jnp.asarray(np.asarray(z_init, dtype=np.float32)[None, ...], dtype=_base.jnp.float32)
    z_anchor = _base.jnp.asarray(z_init_jax, dtype=_base.jnp.float32)
    y_obs = _base.jnp.asarray(y_obs, dtype=_base.jnp.float32)
    time_grid = _base.jnp.asarray(time_grid, dtype=_base.jnp.float32)
    total_steps = int(args.posterior_burnin) + int(args.posterior_num_samples) * int(args.posterior_thinning)
    step_size = float(args.posterior_step_size)
    sqrt_step = _base.jnp.asarray(step_size**0.5, dtype=_base.jnp.float32)

    def ula_step(carry, step_idx):  # noqa: ARG001
        z, key = carry
        grad_z = _base.jax.grad(
            lambda latent: _jax_energy(
                policy,
                model_inputs=model_inputs,
                y_obs=y_obs,
                time_grid=time_grid,
                z=latent,
                z_anchor=z_anchor,
                mode=mode,
                args=args,
            )
        )(z)
        grad_z = _base.jnp.nan_to_num(grad_z)
        grad_z = _base._clip_jax_grad_by_global_norm(
            grad_z,
            max_norm=float(getattr(args, "posterior_grad_clip_norm", 0.0)),
        )
        key, noise_key = _base.jax.random.split(key)
        noise = (
            _base.jax.random.normal(noise_key, z.shape, dtype=_base.jnp.float32)
            if step_size > 0.0
            else _base.jnp.zeros_like(z)
        )
        z_next = z - 0.5 * step_size * grad_z + sqrt_step * noise
        z_next = _base.jnp.nan_to_num(z_next)
        return (z_next, key), z_next

    (_, _), z_history = _base.jax.lax.scan(
        ula_step,
        (z_init_jax, _base.jax.random.key(0)),
        _base.jnp.arange(total_steps, dtype=_base.jnp.int32),
    )
    sample_indices = _base.jnp.asarray(
        [
            int(args.posterior_burnin) + sample_idx * int(args.posterior_thinning)
            for sample_idx in range(int(args.posterior_num_samples))
        ],
        dtype=_base.jnp.int32,
    )
    z_samples = _base.jnp.take(z_history, sample_indices, axis=0)
    z_samples = _base.jnp.swapaxes(z_samples, 0, 1)
    z_samples = _base.jnp.nan_to_num(z_samples)
    sample_energies = _base.jax.vmap(
        lambda z: _jax_energy(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs,
            time_grid=time_grid,
            z=z[None, ...],
            z_anchor=z_anchor,
            mode=mode,
            args=args,
        )
    )(z_samples[0])
    return (
        np.asarray(z_samples[0], dtype=np.float32),
        np.asarray(sample_energies, dtype=np.float32),
    )


def _augment_trace_with_saved_map_posterior(
    policy,
    *,
    trace: _base.InversionChunkTrace,
    episode_nonce: int,
    mode: str,
    args: argparse.Namespace,
) -> _base.InversionChunkTrace:
    if not trace.selected or int(trace.executed_steps) <= 0:
        return _empty_posterior_trace(trace, sample_count=int(args.posterior_num_samples))

    obs = _saved_observation_from_trace(trace, episode_nonce=episode_nonce)
    z_init = np.asarray(trace.recovered_noise, dtype=np.float32)
    chain_init_mode = _saved_trace_chain_init_mode(trace)

    if policy._is_pytorch_model:
        if mode == "channel":
            model_inputs, y_obs, time_grid = _base._prepare_pytorch_channel_observation_context(
                policy,
                obs=obs,
                env_action_chunk=np.asarray(trace.observed_actions, dtype=np.float32),
            )
        else:
            model_inputs, y_obs, time_grid = _base._prepare_pytorch_full_action_observation_context(
                policy,
                obs=obs,
                raw_action_chunk=np.asarray(trace.raw_actions, dtype=np.float32),
            )
        z_samples, sample_energies = _run_pytorch_posterior_from_saved_map(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs,
            time_grid=time_grid,
            z_init=z_init,
            mode=mode,
            args=args,
        )
    else:
        if mode == "channel":
            model_inputs, y_obs, time_grid = _base._prepare_jax_channel_observation_context(
                policy,
                obs=obs,
                env_action_chunk=np.asarray(trace.observed_actions, dtype=np.float32),
            )
        else:
            model_inputs, y_obs, time_grid = _base._prepare_jax_full_action_observation_context(
                policy,
                obs=obs,
                raw_action_chunk=np.asarray(trace.raw_actions, dtype=np.float32),
            )
        z_samples, sample_energies = _run_jax_posterior_from_saved_map(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs,
            time_grid=time_grid,
            z_init=z_init,
            mode=mode,
            args=args,
        )

    z_samples = np.asarray(z_samples, dtype=np.float32)
    z_mean = np.mean(z_samples, axis=0, dtype=np.float32)
    z_std = np.std(z_samples, axis=0, dtype=np.float32)
    best_index = int(np.argmin(sample_energies)) if sample_energies.size else -1
    best_energy = float(sample_energies[best_index]) if best_index >= 0 else float("nan")
    return dataclasses.replace(
        trace,
        posterior_recovered_noise_samples=z_samples,
        posterior_recovered_noise_mean=np.asarray(z_mean, dtype=np.float32),
        posterior_recovered_noise_std=np.asarray(z_std, dtype=np.float32),
        posterior_restart_energies=np.asarray(sample_energies, dtype=np.float32),
        posterior_best_energy=best_energy,
        posterior_best_restart_index=best_index,
        posterior_init_mode=chain_init_mode,
        posterior_chain_init=chain_init_mode,
    )


def _payload_with_augmented_posterior(
    payload: dict[str, np.ndarray],
    traces: Sequence[_base.InversionChunkTrace],
) -> dict[str, np.ndarray]:
    updated = dict(payload)
    updated["chunk_posterior_recovered_noise_samples"] = np.asarray(
        [trace.posterior_recovered_noise_samples for trace in traces],
        dtype=np.float32,
    )
    updated["chunk_posterior_recovered_noise_mean"] = np.asarray(
        [trace.posterior_recovered_noise_mean for trace in traces],
        dtype=np.float32,
    )
    updated["chunk_posterior_recovered_noise_std"] = np.asarray(
        [trace.posterior_recovered_noise_std for trace in traces],
        dtype=np.float32,
    )
    updated["chunk_posterior_restart_energies"] = np.asarray(
        [trace.posterior_restart_energies for trace in traces],
        dtype=np.float32,
    )
    updated["chunk_posterior_best_energy"] = np.asarray(
        [trace.posterior_best_energy for trace in traces],
        dtype=np.float32,
    )
    updated["chunk_posterior_best_restart_index"] = np.asarray(
        [trace.posterior_best_restart_index for trace in traces],
        dtype=np.int32,
    )
    updated["chunk_posterior_init_mode"] = np.asarray([trace.posterior_init_mode for trace in traces])
    updated["chunk_posterior_chain_init"] = np.asarray([trace.posterior_chain_init for trace in traces])
    return updated


def _load_payload(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _augment_rollout_file(
    policy,
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    payload = _load_payload(input_path)
    record = _saved._load_saved_rollout(input_path)
    replay_mode = _resolve_replay_mode(payload, mode=str(args.mode))
    traces = list(record.inversion_traces)
    if args.inversion_step is not None:
        traces = _saved._traces_for_inversion_step(traces, step_count=int(args.inversion_step))
    augmented_traces = [
        _augment_trace_with_saved_map_posterior(
            policy,
            trace=trace,
            episode_nonce=record.episode_nonce,
            mode=replay_mode,
            args=args,
        )
        for trace in traces
    ]
    updated_payload = _payload_with_augmented_posterior(payload, augmented_traces)
    updated_payload["offline_saved_map_posterior"] = np.asarray(True)
    updated_payload["offline_saved_map_posterior_mode"] = np.asarray(replay_mode)
    updated_payload["offline_saved_map_posterior_source"] = np.asarray("chunk_recovered_noise")
    updated_payload["posterior_step_size"] = np.asarray(float(args.posterior_step_size), dtype=np.float32)
    updated_payload["posterior_burnin"] = np.asarray(int(args.posterior_burnin), dtype=np.int32)
    updated_payload["posterior_thinning"] = np.asarray(int(args.posterior_thinning), dtype=np.int32)
    updated_payload["posterior_num_samples"] = np.asarray(int(args.posterior_num_samples), dtype=np.int32)
    updated_payload["obs_sigma"] = np.asarray(float(args.obs_sigma), dtype=np.float32)
    updated_payload["latent_prior_weight"] = np.asarray(float(args.latent_prior_weight), dtype=np.float32)
    updated_payload["posterior_map_tether_weight"] = np.asarray(
        float(args.posterior_map_tether_weight),
        dtype=np.float32,
    )
    updated_payload["posterior_grad_clip_norm"] = np.asarray(
        float(args.posterior_grad_clip_norm),
        dtype=np.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **updated_payload)
    selected_count = sum(1 for trace in augmented_traces if trace.selected and trace.executed_steps > 0)
    return {
        "path": str(output_path),
        "mode": replay_mode,
        "selected_trace_count": int(selected_count),
        "posterior_sample_count": int(args.posterior_num_samples),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    rollout_paths = sorted(args.rollout_dir.rglob("*.npz"))
    if not rollout_paths:
        raise FileNotFoundError(f"No .npz rollouts found under {args.rollout_dir}")

    runtime_modules = _base.online_eval._load_runtime_modules()
    train_config = runtime_modules["training_config"].get_config(args.config_name)
    policy = runtime_modules["policy_config"].create_trained_policy(train_config, args.checkpoint_dir)
    try:
        summaries = []
        for path in rollout_paths:
            out_path = _output_rollout_path(
                path,
                input_root=args.rollout_dir,
                output_root=args.output_dir,
            )
            summaries.append(
                _augment_rollout_file(
                    policy,
                    input_path=path,
                    output_path=out_path,
                    args=args,
                )
            )
    finally:
        _base._release_policy(policy)

    print("Saved LIBERO latent posterior replay")
    print(f"rollout_dir={args.rollout_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"files={len(summaries)}")
    print(f"posterior_num_samples={int(args.posterior_num_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
