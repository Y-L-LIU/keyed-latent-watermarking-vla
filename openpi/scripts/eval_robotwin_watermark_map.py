#!/usr/bin/env python3
"""Evaluate Pi0.5 internal-noise watermark MAP recovery on RoboTwin tasks."""

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import hashlib
import json
import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi import transforms as transforms_lib  # noqa: E402
from openpi.models import model as model_lib  # noqa: E402
from openpi.models import pi0 as jax_pi0  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.policies import watermark as wm  # noqa: E402
from openpi.training import config as training_config  # noqa: E402


@dataclasses.dataclass(frozen=True)
class ChunkTrace:
    chunk_index: int
    executed_steps: int
    selected: bool
    reference: np.ndarray
    injected_noise: np.ndarray
    recovered_noise: np.ndarray
    raw_actions: np.ndarray
    observed_actions: np.ndarray
    map_restart_energies: np.ndarray
    map_best_restart_index: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=pathlib.Path, default=pathlib.Path("/data_sdh/anon/robotwin/RoboTwin"))
    parser.add_argument("--task-name", type=str, default="beat_block_hammer")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--instruction-type", type=str, default="unseen")
    parser.add_argument("--config-name", type=str, default="pi05_aloha_robotwin_lora")
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data_sdh/anon/openpi-checkpoints/pi05_aloha_robotwin_lora/demo_clean_100_lora_fsdp4/5999"
        ),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/data_sdh/anon/robotwin_watermark_map"))
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=("plain", "watermarked"), default=("plain", "watermarked"))
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--secret-key", type=int, default=17)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--freq-min-hz", type=float, default=1.0)
    parser.add_argument("--freq-max-hz", type=float, default=2.0)
    parser.add_argument("--n-tones", type=int, default=4)
    parser.add_argument("--reference-mode", choices=("bandpass", "gaussian"), default="gaussian")
    parser.add_argument(
        "--chunk-selection-strategy",
        choices=("periodic", "fixed_slots", "stateful_online"),
        default="stateful_online",
    )
    parser.add_argument("--chunk-selection-period", type=int, default=1)
    parser.add_argument("--chunk-selection-count", type=int, default=5)
    parser.add_argument("--chunk-selection-total-slots", type=int, default=None)
    parser.add_argument("--max-score-windows", type=int, default=None)
    parser.add_argument("--detector", choices=("cosine", "dot", "mse", "wmf"), default="wmf")
    parser.add_argument("--null-decoy-count", type=int, default=32)
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--score-step-scope", choices=("executed", "full_chunk"), default="executed")
    parser.add_argument("--latent-map-iters", type=int, default=100)
    parser.add_argument("--latent-map-lr", type=float, default=1e-1)
    parser.add_argument("--latent-prior-weight", type=float, default=1.0)
    parser.add_argument("--obs-sigma", type=float, default=1e-4)
    parser.add_argument("--map-num-starts", type=int, default=1)
    parser.add_argument("--map-random-seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--no-expert-check", action="store_true")
    parser.add_argument("--eval-video-log", action="store_true")
    # §12.5 verifier-uses-base setting: when both are set, rollout uses the
    # suspect (`--checkpoint-dir`) and MAP latent recovery uses the base detector.
    parser.add_argument("--detector-config-name", type=str, default=None)
    parser.add_argument("--detector-checkpoint-dir", type=pathlib.Path, default=None)
    return parser.parse_args()


def _stable_seed(*parts: int | str) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _make_base_noise(*, action_horizon: int, action_dim: int, episode_nonce: int, chunk_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(seed, episode_nonce, chunk_index, action_horizon, action_dim))
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def _make_watermark_config(args: argparse.Namespace, *, action_dim: int) -> wm.InternalNoiseWatermarkConfig:
    return wm.InternalNoiseWatermarkConfig(
        secret_key=int(args.secret_key),
        control_freq=float(args.sample_rate_hz),
        beta=float(args.beta),
        freq_range=(float(args.freq_min_hz), float(args.freq_max_hz)),
        n_tones=int(args.n_tones),
        watermark_dims=tuple(range(int(action_dim))),
        reference_mode=str(args.reference_mode),
        chunk_selection_strategy=str(args.chunk_selection_strategy),
        chunk_selection_period=int(args.chunk_selection_period),
        chunk_selection_count=int(args.chunk_selection_count),
        chunk_selection_total_slots=(
            None if args.chunk_selection_total_slots is None else int(args.chunk_selection_total_slots)
        ),
    )


def _prepare_policy_inputs(policy, obs: dict) -> tuple[model_lib.Observation, dict]:
    obs = policy._strip_runtime_metadata(obs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    return model_lib.Observation.from_dict(inputs), inputs


def _sample_raw_actions(policy, obs: dict, *, noise: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    observation, inputs = _prepare_policy_inputs(policy, obs)
    context = policy._extract_watermark_context(obs)  # noqa: SLF001
    prepared_noise = policy._prepare_internal_noise(  # noqa: SLF001
        noise,
        batch_size=inputs["state"].shape[0],
        sample_rng_or_pytorch_device=None,
        noise_rng=None,
        context=context,
    )
    sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
    sample_kwargs["noise"] = prepared_noise
    sample_kwargs["num_steps"] = int(sample_kwargs.get("num_steps", 10))
    raw_batch = policy._sample_actions(jax.random.key(0), observation, **sample_kwargs)  # noqa: SLF001
    raw_actions = np.asarray(raw_batch[0], dtype=np.float32)
    outputs = {
        "state": np.asarray(inputs["state"][0], dtype=np.float32),
        "actions": raw_actions.copy(),
    }
    transformed = policy._output_transform(outputs)  # noqa: SLF001
    transformed["raw_actions"] = raw_actions
    return transformed, np.asarray(prepared_noise[0], dtype=np.float32)


def _prepare_jax_sampling_context(policy, observation: model_lib.Observation):
    model = policy._model  # noqa: SLF001
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(preprocessed)
    prefix_attn_mask = jax_pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return preprocessed, prefix_mask, kv_cache


def _sample_time_grid(num_steps: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, int(num_steps) + 1, dtype=np.float32)


def _normalize_channel_observation(policy, env_action_chunk: np.ndarray) -> np.ndarray:
    observed = np.asarray(env_action_chunk, dtype=np.float32)
    unnormalize = next(
        (
            transform
            for transform in getattr(policy._output_transform, "transforms", ())  # noqa: SLF001
            if isinstance(transform, transforms_lib.Unnormalize)
        ),
        None,
    )
    if unnormalize is None or unnormalize.norm_stats is None:
        return observed
    normalizer = transforms_lib.Normalize(unnormalize.norm_stats, use_quantiles=bool(unnormalize.use_quantiles))
    return np.asarray(normalizer({"actions": observed.copy()})["actions"], dtype=np.float32)


def _optimize_latent_with_adam_jax(
    *,
    init_noise: jax.Array,
    loss_fn,
    num_steps: int,
    learning_rate: float,
) -> jax.Array:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    loss_and_grad = jax.value_and_grad(loss_fn)

    def step_fn(step_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        latent, first_moment, second_moment = carry
        _, grad = loss_and_grad(latent)
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(grad)
        step = jnp.asarray(step_idx + 1, dtype=jnp.float32)
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)
        latent = latent - learning_rate * first_hat / (jnp.sqrt(second_hat) + eps)
        return latent, first_moment, second_moment

    latent, _, _ = jax.lax.fori_loop(
        0,
        int(num_steps),
        step_fn,
        (init_noise, jnp.zeros_like(init_noise), jnp.zeros_like(init_noise)),
    )
    return latent


def _map_restart_seed(obs: dict, *, args: argparse.Namespace) -> int:
    seed = (
        int(args.map_random_seed)
        + int(obs.get("episode_nonce", 0)) * 1009
        + int(obs.get("chunk_index", 0)) * 9176
    ) % (2**32)
    return int(seed)


def _recover_noise_map_jax(
    policy,
    *,
    obs: dict,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    observation, _ = _prepare_policy_inputs(policy, obs)
    model_inputs = _prepare_jax_sampling_context(policy, observation)
    y_obs = jnp.asarray(raw_action_chunk, dtype=jnp.float32)[None, ...]
    time_grid = jnp.asarray(_sample_time_grid(int(args.num_inference_steps)), dtype=jnp.float32)
    raw_dim = int(getattr(policy._model, "action_dim", 32))  # noqa: SLF001
    horizon = int(y_obs.shape[1])

    def single_energy(z: jax.Array) -> jax.Array:
        # z: (1, horizon, raw_dim) -> scalar (batch-1, matching the batch-1 model prefix).
        a_pred = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)  # noqa: SLF001
        pred_obs = a_pred[:, :, : y_obs.shape[-1]]
        pred_obs = jnp.where(jnp.isfinite(pred_obs), pred_obs, y_obs)
        obs_loss = 0.5 * jnp.mean(jnp.square((pred_obs - y_obs) / float(args.obs_sigma)))
        prior_loss = 0.5 * float(args.latent_prior_weight) * jnp.mean(jnp.square(z))
        return obs_loss + prior_loss

    # Multi-start MAP: the restarts are independent random inits. sample_actions_from_noise
    # concatenates against the batch-1 model prefix, so we can't just widen z's batch; instead
    # vmap single_energy over the restart axis (the prefix is a closed-over constant, broadcast
    # across restarts). Optimizing the SUM of per-restart energies makes d(loss)/d z[i] exactly
    # z[i]'s own single-start gradient and keeps Adam's per-element state independent per
    # restart -- numerically identical to the old sequential loop, but all restarts run in one
    # batched forward/backward (~N x faster on GPU).
    batched_energy = jax.vmap(single_energy)  # (B,1,H,D) -> (B,)
    n_starts = int(args.map_num_starts)
    rng = np.random.default_rng(_map_restart_seed(obs, args=args))
    z0 = rng.standard_normal(size=(n_starts, 1, horizon, raw_dim)).astype(np.float32)
    z_map = _optimize_latent_with_adam_jax(
        init_noise=jnp.asarray(z0, dtype=jnp.float32),
        loss_fn=lambda z: jnp.sum(batched_energy(z)),
        num_steps=int(args.latent_map_iters),
        learning_rate=float(args.latent_map_lr),
    )
    z_map = jnp.nan_to_num(z_map)
    energies = np.asarray(batched_energy(z_map), dtype=np.float32)  # (B,)
    best = int(np.argmin(energies)) if energies.size else -1
    return np.asarray(z_map[best][0], dtype=np.float32), energies, best


def _score_similarity(noise: np.ndarray, reference: np.ndarray, *, detector: str) -> float:
    x = np.asarray(noise, dtype=np.float32).reshape(-1)
    r = np.asarray(reference, dtype=np.float32).reshape(-1)
    n = min(x.size, r.size)
    if n == 0:
        return 0.0
    x = x[:n] - float(np.mean(x[:n]))
    r = r[:n] - float(np.mean(r[:n]))
    if detector == "dot":
        return float(np.dot(x, r) / max(n, 1))
    if detector == "mse":
        return -float(np.mean(np.square(x - r)))
    denom = float(np.linalg.norm(x) * np.linalg.norm(r))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(x, r) / denom)


def _selected_score_traces(traces: list[ChunkTrace], *, max_windows: int | None) -> list[ChunkTrace]:
    selected = [trace for trace in traces if trace.selected and trace.executed_steps > 0]
    if max_windows is None:
        return selected
    return selected[: int(max_windows)]


def _score_vector(
    traces: list[ChunkTrace],
    *,
    detector: str,
    score_step_scope: str,
    max_windows: int | None,
) -> np.ndarray:
    scores = []
    for trace in _selected_score_traces(traces, max_windows=max_windows):
        steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        scores.append(_score_similarity(trace.recovered_noise[:steps], trace.reference[:steps], detector=detector))
    return np.asarray(scores, dtype=np.float32)


def _wmf_score_from_vectors(feature: np.ndarray, null_matrix: np.ndarray, *, subspace_rank: int | None = None) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0 or null_matrix.size == 0:
        return 0.0
    centered_feature = feature - np.mean(null_matrix, axis=0)
    centered_null = null_matrix - np.mean(null_matrix, axis=0, keepdims=True)
    cov = np.cov(centered_null, rowvar=False, bias=False) if null_matrix.shape[0] > 1 else np.eye(feature.size)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov = cov + max(1e-6, 1e-4 * float(np.trace(cov)) / max(feature.size, 1)) * np.eye(feature.size)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if subspace_rank is not None:
        rank = min(int(subspace_rank), feature.size)
        eigvals = eigvals[:rank]
        eigvecs = eigvecs[:, :rank]
    projected = eigvecs.T @ centered_feature
    template = np.sum(eigvecs, axis=0)
    return float(np.dot(template / np.sqrt(np.maximum(eigvals, 1e-8)), projected / np.sqrt(np.maximum(eigvals, 1e-8))))


def _episode_score(
    traces: list[ChunkTrace],
    *,
    reference_config: wm.InternalNoiseWatermarkConfig,
    episode_nonce: int,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    base_detector = "cosine" if args.detector == "wmf" else args.detector
    true_vector = _score_vector(
        traces,
        detector=base_detector,
        score_step_scope=args.score_step_scope,
        max_windows=args.max_score_windows,
    )
    if args.detector != "wmf":
        return float(np.sum(true_vector)), true_vector

    null_vectors = []
    for offset in range(1, int(args.null_decoy_count) + 1):
        cfg = dataclasses.replace(reference_config, secret_key=int(reference_config.secret_key) + offset)
        retargeted = []
        for trace in traces:
            context = wm.WatermarkContext(chunk_index=trace.chunk_index, episode_nonce=episode_nonce)
            ref = wm.generate_keyed_reference(
                length=trace.reference.shape[0],
                action_dim=trace.reference.shape[1],
                sample_rate_hz=float(args.sample_rate_hz),
                config=cfg,
                context=context,
            )
            retargeted.append(dataclasses.replace(trace, reference=ref))
        null_vec = _score_vector(
            retargeted,
            detector=base_detector,
            score_step_scope=args.score_step_scope,
            max_windows=args.max_score_windows,
        )
        if null_vec.shape == true_vector.shape:
            null_vectors.append(null_vec)
    if not null_vectors:
        return 0.0, true_vector
    return _wmf_score_from_vectors(true_vector, np.asarray(null_vectors), subspace_rank=args.subspace_rank), true_vector


def _robotwin_paths(robotwin_root: pathlib.Path) -> None:
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "script"))
    sys.path.insert(0, str(robotwin_root / "description" / "utils"))


def _load_robotwin_config(args: argparse.Namespace) -> dict[str, Any]:
    import yaml

    with open(args.robotwin_root / "task_config" / f"{args.task_config}.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "policy_name": "pi05_openpi_wm_map",
            "ckpt_setting": args.checkpoint_dir.name,
            "eval_mode": True,
            "eval_video_log": bool(args.eval_video_log),
        }
    )
    if args.max_rollout_steps is not None:
        cfg["eval_step_limit_override"] = int(args.max_rollout_steps)
    return cfg


def _create_robotwin_env(task_name: str):
    import importlib

    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def _fill_robotwin_runtime_config(robotwin_root: pathlib.Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import yaml

    from envs import CONFIGS_PATH

    with open(pathlib.Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    with open(pathlib.Path(CONFIGS_PATH) / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    embodiment_type = cfg["embodiment"]

    def embodiment_file(name: str) -> str:
        return embodiment_types[name]["file_path"]

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    else:
        cfg["left_robot_file"] = embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False

    def load_embodiment_config(robot_file: str) -> dict[str, Any]:
        with open(pathlib.Path(robot_file) / "config.yml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    cfg["left_embodiment_config"] = load_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = load_embodiment_config(cfg["right_robot_file"])
    head_camera_type = cfg["camera"]["head_camera_type"]
    cfg["head_camera_h"] = camera_config[head_camera_type]["h"]
    cfg["head_camera_w"] = camera_config[head_camera_type]["w"]
    cfg.setdefault("save_path", str(robotwin_root / "data"))
    return cfg


def _encode_obs(observation: dict, *, prompt: str, chunk_index: int, episode_nonce: int) -> dict:
    images = observation["observation"]
    return {
        "images": {
            "cam_high": np.transpose(images["head_camera"]["rgb"], (2, 0, 1)),
            "cam_right_wrist": np.transpose(images["right_camera"]["rgb"], (2, 0, 1)),
            "cam_left_wrist": np.transpose(images["left_camera"]["rgb"], (2, 0, 1)),
        },
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "prompt": prompt,
        "chunk_index": int(chunk_index),
        "episode_nonce": int(episode_nonce),
    }


def _select_eval_seed(task_env, cfg: dict[str, Any], *, start_seed: int, no_expert_check: bool) -> tuple[int, dict[str, Any] | None]:
    if no_expert_check:
        return start_seed, None
    from envs.utils.create_actor import UnStableError

    now_seed = start_seed
    now_id = 0
    while True:
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **cfg)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            continue
        except Exception:
            task_env.close_env()
            now_seed += 1
            continue
        if task_env.plan_success and task_env.check_success():
            return now_seed, episode_info
        now_seed += 1


def _instruction_for_episode(args: argparse.Namespace, cfg: dict[str, Any], episode_info: dict[str, Any] | None) -> str:
    if episode_info is None:
        return ""
    from generate_episode_instructions import generate_episode_descriptions

    results = generate_episode_descriptions(args.task_name, [episode_info["info"]], 1)
    return str(np.random.choice(results[0][args.instruction_type]))


def _run_episode(
    *,
    task_env,
    policy,
    detector_policy,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    seed: int,
    episode_idx: int,
    episode_info: dict[str, Any] | None,
    variant: str,
    reference_config: wm.InternalNoiseWatermarkConfig,
) -> dict[str, Any]:
    episode_nonce = int(_stable_seed(args.seed, seed, episode_idx))
    instruction = _instruction_for_episode(args, cfg, episode_info)
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **cfg)
    if instruction:
        task_env.set_instruction(instruction=instruction)
    else:
        instruction = str(task_env.get_instruction())

    traces: list[ChunkTrace] = []
    actions_to_execute: deque[np.ndarray] = deque()
    chunk_index = 0
    success = False
    max_steps = int(task_env.step_lim if args.max_rollout_steps is None else min(task_env.step_lim, args.max_rollout_steps))
    saved_observed_actions: list[np.ndarray] = []

    original_watermark_config = policy._watermark_config  # noqa: SLF001
    try:
        policy._watermark_config = reference_config if variant == "watermarked" else None  # noqa: SLF001
        while task_env.take_action_cnt < max_steps:
            observation = task_env.get_obs()
            if not actions_to_execute:
                obs = _encode_obs(
                    observation,
                    prompt=instruction,
                    chunk_index=chunk_index,
                    episode_nonce=episode_nonce,
                )
                action_horizon = int(policy._model.action_horizon)  # noqa: SLF001
                action_dim = int(policy._model.action_dim)  # noqa: SLF001
                base_noise = _make_base_noise(
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    episode_nonce=episode_nonce,
                    chunk_index=chunk_index,
                    seed=args.seed,
                )
                context = wm.WatermarkContext(chunk_index=chunk_index, episode_nonce=episode_nonce)
                selected = bool(wm.should_watermark_chunk(reference_config, context))
                reference = wm.generate_keyed_reference(
                    length=action_horizon,
                    action_dim=action_dim,
                    sample_rate_hz=float(args.sample_rate_hz),
                    config=reference_config,
                    context=context,
                )
                if not selected:
                    reference = np.zeros_like(reference)
                outputs, injected_noise = _sample_raw_actions(policy, obs, noise=base_noise)
                action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
                raw_actions = np.asarray(outputs["raw_actions"], dtype=np.float32)
                planned_steps = int(min(args.pi0_step, action_chunk.shape[0], max_steps - task_env.take_action_cnt))
                observed_chunk = action_chunk[:planned_steps]
                recovered_noise = np.zeros_like(reference)
                energies = np.zeros((0,), dtype=np.float32)
                best_restart = -1
                if selected:
                    obs_dim = action_chunk.shape[-1]
                    recovered_noise, energies, best_restart = _recover_noise_map_jax(
                        detector_policy if detector_policy is not None else policy,
                        obs=obs,
                        raw_action_chunk=raw_actions[:, :obs_dim],
                        args=args,
                    )
                traces.append(
                    ChunkTrace(
                        chunk_index=chunk_index,
                        executed_steps=planned_steps,
                        selected=selected,
                        reference=np.asarray(reference, dtype=np.float32),
                        injected_noise=np.asarray(injected_noise, dtype=np.float32),
                        recovered_noise=np.asarray(recovered_noise, dtype=np.float32),
                        raw_actions=np.asarray(raw_actions, dtype=np.float32),
                        observed_actions=np.asarray(action_chunk, dtype=np.float32),
                        map_restart_energies=energies,
                        map_best_restart_index=best_restart,
                    )
                )
                actions_to_execute.extend(observed_chunk)
                chunk_index += 1

            action = actions_to_execute.popleft()
            saved_observed_actions.append(np.asarray(action, dtype=np.float32))
            task_env.take_action(action)
            if task_env.eval_success:
                success = True
                break
    finally:
        policy._watermark_config = original_watermark_config  # noqa: SLF001
        task_env.close_env()

    score, chunk_scores = _episode_score(
        traces,
        reference_config=reference_config,
        episode_nonce=episode_nonce,
        args=args,
    )
    return {
        "episode_idx": episode_idx,
        "seed": seed,
        "variant": variant,
        "episode_nonce": episode_nonce,
        "instruction": instruction,
        "success": success,
        "steps": int(len(saved_observed_actions)),
        "score": float(score),
        "chunk_scores": chunk_scores,
        "traces": traces,
        "executed_actions": np.asarray(saved_observed_actions, dtype=np.float32),
    }


def _save_episode(out_dir: pathlib.Path, result: dict[str, Any]) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{int(result['episode_idx']):03d}_{result['variant']}.npz"
    traces: list[ChunkTrace] = result["traces"]
    np.savez_compressed(
        path,
        episode_idx=np.asarray(result["episode_idx"], dtype=np.int32),
        seed=np.asarray(result["seed"], dtype=np.int64),
        variant=np.asarray(result["variant"]),
        episode_nonce=np.asarray(result["episode_nonce"], dtype=np.int64),
        instruction=np.asarray(result["instruction"]),
        success=np.asarray(result["success"]),
        steps=np.asarray(result["steps"], dtype=np.int32),
        score=np.asarray(result["score"], dtype=np.float32),
        chunk_scores=np.asarray(result["chunk_scores"], dtype=np.float32),
        executed_actions=np.asarray(result["executed_actions"], dtype=np.float32),
        chunk_index=np.asarray([t.chunk_index for t in traces], dtype=np.int32),
        chunk_executed_steps=np.asarray([t.executed_steps for t in traces], dtype=np.int32),
        chunk_selected=np.asarray([t.selected for t in traces], dtype=bool),
        chunk_reference=np.asarray([t.reference for t in traces], dtype=np.float32),
        chunk_injected_noise=np.asarray([t.injected_noise for t in traces], dtype=np.float32),
        chunk_recovered_noise=np.asarray([t.recovered_noise for t in traces], dtype=np.float32),
        chunk_raw_actions=np.asarray([t.raw_actions for t in traces], dtype=np.float32),
        chunk_observed_actions=np.asarray([t.observed_actions for t in traces], dtype=np.float32),
        chunk_map_restart_energies=np.asarray([t.map_restart_energies for t in traces], dtype=object),
        chunk_map_best_restart_index=np.asarray([t.map_best_restart_index for t in traces], dtype=np.int32),
    )
    return path


def main() -> None:
    args = _parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0")
    if args.pi0_step <= 0:
        raise ValueError("--pi0-step must be > 0")
    if args.map_num_starts <= 0:
        raise ValueError("--map-num-starts must be > 0")
    if args.chunk_selection_period <= 0:
        raise ValueError("--chunk-selection-period must be > 0")
    if args.chunk_selection_strategy == "periodic" and not (
        0 <= args.chunk_selection_count <= args.chunk_selection_period
    ):
        raise ValueError("--chunk-selection-count must be in [0, period] for periodic selection")
    if args.chunk_selection_strategy == "fixed_slots" and args.chunk_selection_total_slots is None:
        raise ValueError("--chunk-selection-total-slots is required for fixed_slots selection")
    if args.chunk_selection_count < 0:
        raise ValueError("--chunk-selection-count must be >= 0")
    if args.max_score_windows is not None and args.max_score_windows <= 0:
        raise ValueError("--max-score-windows must be > 0")

    _robotwin_paths(args.robotwin_root)
    from test_render import Sapien_TEST

    Sapien_TEST()
    cfg = _fill_robotwin_runtime_config(args.robotwin_root, _load_robotwin_config(args))
    train_config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": int(args.num_inference_steps)},
    )
    if policy._is_pytorch_model:  # noqa: SLF001
        raise RuntimeError("This RoboTwin MAP adapter currently supports the JAX checkpoint path.")
    # Optional separate detector policy for §12.5 base-detector verifier setting.
    detector_policy = None
    if args.detector_checkpoint_dir is not None:
        det_config_name = args.detector_config_name or args.config_name
        det_train_config = training_config.get_config(det_config_name)
        detector_policy = policy_config.create_trained_policy(
            det_train_config,
            args.detector_checkpoint_dir,
            sample_kwargs={"num_steps": int(args.num_inference_steps)},
        )
        if detector_policy._is_pytorch_model:  # noqa: SLF001
            raise RuntimeError("Detector must also be a JAX checkpoint.")
        print(
            f"[detector] using base detector config={det_config_name} ckpt={args.detector_checkpoint_dir}",
            flush=True,
        )
    reference_config = _make_watermark_config(args, action_dim=int(policy._model.action_dim))  # noqa: SLF001

    run_dir = args.output_dir / args.task_name / args.config_name / args.checkpoint_dir.name
    task_env = _create_robotwin_env(args.task_name)
    start_seed = 100000 * (1 + int(args.seed))
    records = []
    for episode_idx in range(int(args.num_episodes)):
        if all((run_dir / f"episode_{episode_idx:03d}_{v}.npz").exists() for v in args.variants):
            print(f"[resume] skip episode {episode_idx} (all variants exist)", flush=True)
            continue
        try:
            seed, episode_info = _select_eval_seed(
                task_env,
                cfg,
                start_seed=start_seed + episode_idx,
                no_expert_check=bool(args.no_expert_check),
            )
            for variant in args.variants:
                result = _run_episode(
                    task_env=task_env,
                    policy=policy,
                    detector_policy=detector_policy,
                    args=args,
                    cfg=cfg,
                    seed=seed,
                    episode_idx=episode_idx,
                    episode_info=episode_info,
                    variant=variant,
                    reference_config=reference_config,
                )
                path = _save_episode(run_dir, result)
                records.append(
                    {
                        "episode_idx": int(result["episode_idx"]),
                        "seed": int(result["seed"]),
                        "variant": str(result["variant"]),
                        "success": bool(result["success"]),
                        "steps": int(result["steps"]),
                        "score": float(result["score"]),
                        "chunk_count": int(len(result["traces"])),
                        "selected_chunk_count": int(sum(1 for trace in result["traces"] if trace.selected)),
                        "path": str(path),
                    }
                )
                print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001 — skip unstable/unsolvable seeds, keep the worker alive
            print(f"[skip] episode {episode_idx} failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                task_env.close_env()
            except Exception:
                pass
            continue

    summary = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "config_name": args.config_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "pi0_step": int(args.pi0_step),
        "detector": args.detector,
        "num_episodes": int(args.num_episodes),
        "beta": float(args.beta),
        "reference_mode": str(args.reference_mode),
        "chunk_selection_strategy": str(args.chunk_selection_strategy),
        "chunk_selection_period": int(args.chunk_selection_period),
        "chunk_selection_count": int(args.chunk_selection_count),
        "chunk_selection_total_slots": args.chunk_selection_total_slots,
        "max_score_windows": args.max_score_windows,
        "records": records,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
