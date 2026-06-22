#!/usr/bin/env python3
"""Shared LIBERO watermark helpers used by current inversion and saved-rollout scripts."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import math
import pathlib
from types import SimpleNamespace
from typing import Any

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image

from openpi.policies import watermark as wm
from openpi.policies import policy_config
from openpi.training import config as training_config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_RATE_SEARCH_FACTORS = [0.95, 0.975, 1.0, 1.025, 1.05]


@dataclasses.dataclass(frozen=True)
class ExecutionSegment:
    chunk_index: int
    start_step: int
    end_step: int
    executed_steps: int


@dataclasses.dataclass(frozen=True)
class ChunkTrace:
    chunk_index: int
    start_step: int
    end_step: int
    executed_steps: int
    base_noise: np.ndarray
    applied_noise: np.ndarray
    reference: np.ndarray
    predicted_actions: np.ndarray


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    telemetry: np.ndarray
    success: bool
    chunk_size: int
    task_description: str
    steps: int
    execution_segments: tuple[ExecutionSegment, ...]
    chunk_traces: tuple[ChunkTrace, ...]
    executed_actions: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    output_reference: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    clip_fraction: float = 0.0
    saturation_fraction: float = 0.0
    mean_action_l2: float = 0.0


@dataclasses.dataclass(frozen=True)
class WhiteboxDiagnosticSummary:
    common_chunk_count: int
    common_step_count: int
    internal_noise_delta_rms: float
    action_delta_rms: float
    telemetry_delta_rms: float
    group_internal_to_action: np.ndarray
    group_internal_to_telemetry: np.ndarray


class _ImageTools:
    @staticmethod
    def convert_to_uint8(img: np.ndarray) -> np.ndarray:
        if np.issubdtype(np.asarray(img).dtype, np.floating):
            return (255 * np.asarray(img)).astype(np.uint8)
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
        images = np.asarray(images)
        if images.shape[-3:-1] == (height, width):
            return images
        original_shape = images.shape
        flat = images.reshape(-1, *original_shape[-3:])
        resized = np.stack([_ImageTools._resize_with_pad_pil(Image.fromarray(img), height, width, method) for img in flat])
        return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

    @staticmethod
    def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
        cur_width, cur_height = image.size
        if cur_width == width and cur_height == height:
            return np.asarray(image)
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized = image.resize((resized_width, resized_height), resample=method)
        zero_image = Image.new(resized.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized, (pad_width, pad_height))
        return np.asarray(zero_image)


def _load_runtime_modules() -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "training_config": training_config,
        "policy_config": policy_config,
        "image_tools": _ImageTools,
    }


def _suite_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int, runtime_modules: dict[str, Any] | None = None):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: Sequence[float]) -> np.ndarray:
    quat_np = np.asarray(quat, dtype=np.float32).copy()
    quat_np[3] = float(np.clip(quat_np[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - quat_np[3] * quat_np[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray((quat_np[:3] * 2.0 * math.acos(float(quat_np[3]))) / den, dtype=np.float32)


def _extract_telemetry(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def _prepare_policy_observation(
    obs: dict,
    *,
    task_description: str,
    resize_size: int,
    image_tools: Any,
) -> dict[str, np.ndarray | str]:
    img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
    wrist_img = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return {
        "observation/image": np.asarray(img, dtype=np.uint8),
        "observation/wrist_image": np.asarray(wrist_img, dtype=np.uint8),
        "observation/state": _extract_telemetry(obs),
        "prompt": str(task_description),
    }


def _stable_seed(*parts: int | str) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _make_chunk_base_noise(*, action_horizon: int, action_dim: int, episode_nonce: int, chunk_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(seed, episode_nonce, chunk_index, action_horizon, action_dim))
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def _make_watermark_config(args: Any, *, telemetry_dim: int) -> wm.InternalNoiseWatermarkConfig:
    return wm.InternalNoiseWatermarkConfig(
        secret_key=int(args.secret_key),
        control_freq=float(args.sample_rate_hz),
        beta=float(args.beta),
        freq_range=(float(args.freq_min_hz), float(args.freq_max_hz)),
        n_tones=int(args.n_tones),
        watermark_dims=(
            tuple(int(x) for x in str(getattr(args, "watermark_dims", "")).split(","))
            if getattr(args, "watermark_dims", None) else tuple(range(int(telemetry_dim)))
        ),
        reference_mode=str(args.reference_mode),
        chunk_selection_strategy=str(args.chunk_selection_strategy),
        chunk_selection_period=int(args.chunk_selection_period),
        chunk_selection_count=int(args.chunk_selection_count),
        chunk_selection_total_slots=(
            None if getattr(args, "chunk_selection_total_slots", None) is None else int(args.chunk_selection_total_slots)
        ),
        keying_mode=str(getattr(args, "keying_mode", "nonce")),
        obs_key=str(getattr(args, "obs_key", "observation/state")),
        obs_proj_dims=(
            None
            if not getattr(args, "obs_proj_dims", None)
            else tuple(int(x) for x in str(args.obs_proj_dims).split(","))
        ),
        obs_quantization=float(getattr(args, "obs_quantization", 0.5)),
    )


def _align_reference_to_action_signal(signal: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_np = np.asarray(signal, dtype=np.float32)
    reference_np = np.asarray(reference, dtype=np.float32)
    if signal_np.ndim == 1:
        signal_np = signal_np[:, None]
    if reference_np.ndim == 1:
        reference_np = reference_np[:, None]
    length = min(signal_np.shape[0], reference_np.shape[0])
    dims = min(signal_np.shape[1], reference_np.shape[1])
    return signal_np[:length, :dims], reference_np[:length, :dims]


def _multichannel_band_coherence_score(
    telemetry: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> tuple[float, np.ndarray]:
    aligned_telemetry, aligned_reference = _align_reference_to_action_signal(telemetry, reference)
    return wm.narrow_band_coherency_score(
        aligned_telemetry,
        aligned_reference,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        watermark_dims=tuple(range(aligned_telemetry.shape[1])) if aligned_telemetry.ndim == 2 else None,
    )


def _prepare_arm_detector_trace(trace: np.ndarray) -> np.ndarray:
    trace_np = np.asarray(trace, dtype=np.float32)
    if trace_np.ndim == 1:
        trace_np = trace_np[:, None]
    if trace_np.shape[1] >= 6:
        return trace_np[:, :6]
    return trace_np


def _build_reference_trace_from_segments(
    *,
    total_length: int,
    action_dim: int,
    sample_rate_hz: float,
    config: wm.InternalNoiseWatermarkConfig,
    episode_nonce: int,
    execution_segments: Sequence[ExecutionSegment],
) -> np.ndarray:
    trace = np.zeros((total_length, action_dim), dtype=np.float32)
    for segment in execution_segments:
        context = wm.WatermarkContext(chunk_index=int(segment.chunk_index), episode_nonce=int(episode_nonce))
        if not wm.should_watermark_chunk(config, context):
            continue
        reference = wm.generate_keyed_reference(
            length=int(segment.executed_steps),
            action_dim=action_dim,
            sample_rate_hz=sample_rate_hz,
            config=config,
            context=context,
        )
        trace[int(segment.start_step) : int(segment.start_step) + int(segment.executed_steps)] = reference[: int(segment.executed_steps)]
    return trace


def _welch_spectra(x: np.ndarray, y: np.ndarray, *, sample_rate_hz: float) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    x_np = _prepare_arm_detector_trace(x)
    y_np = _prepare_arm_detector_trace(y)
    length = min(x_np.shape[0], y_np.shape[0])
    x_np = x_np[:length]
    y_np = y_np[:length]
    if length == 0:
        freqs = np.zeros((0,), dtype=np.float32)
        empty = np.zeros((0, x_np.shape[1] if x_np.ndim == 2 else 0), dtype=np.float32)
        return freqs, (empty, empty), empty
    x_centered = x_np - np.mean(x_np, axis=0, keepdims=True)
    y_centered = y_np - np.mean(y_np, axis=0, keepdims=True)
    x_fft = np.fft.rfft(x_centered, axis=0)
    y_fft = np.fft.rfft(y_centered, axis=0)
    freqs = np.fft.rfftfreq(length, d=1.0 / float(sample_rate_hz)).astype(np.float32)
    psd_x = (np.abs(x_fft) ** 2 / max(length, 1)).astype(np.float32)
    psd_y = (np.abs(y_fft) ** 2 / max(length, 1)).astype(np.float32)
    csd = (x_fft * np.conjugate(y_fft) / max(length, 1)).astype(np.complex64)
    return freqs, (psd_x, psd_y), csd


def _detect_presence_for_rollout(
    result: RolloutResult,
    *,
    secret_key: int,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    n_tones: int,
    episode_nonce: int,
    threshold: float,
    lag_search_steps: int = 0,
    rate_search_factors: Sequence[float] = (1.0,),
    watermark_dims: Sequence[int] | None = None,
) -> wm.PresenceDetectionResult:
    telemetry = _prepare_arm_detector_trace(result.telemetry)
    action_dim = int(result.telemetry.shape[1]) if result.telemetry.ndim == 2 else int(telemetry.shape[1])
    if result.execution_segments:
        config = wm.InternalNoiseWatermarkConfig(
            secret_key=secret_key,
            control_freq=sample_rate_hz,
            beta=0.0,
            freq_range=freq_range,
            n_tones=n_tones,
            watermark_dims=tuple(range(action_dim)),
        )
        reference = _build_reference_trace_from_segments(
            total_length=int(result.telemetry.shape[0]),
            action_dim=action_dim,
            sample_rate_hz=sample_rate_hz,
            config=config,
            episode_nonce=episode_nonce,
            execution_segments=result.execution_segments,
        )
    else:
        reference = wm.generate_reference_trace(
            total_length=int(result.telemetry.shape[0]),
            action_dim=action_dim,
            sample_rate_hz=sample_rate_hz,
            chunk_size=max(int(result.chunk_size), 1),
            config=wm.InternalNoiseWatermarkConfig(
                secret_key=secret_key,
                control_freq=sample_rate_hz,
                beta=0.0,
                freq_range=freq_range,
                n_tones=n_tones,
                watermark_dims=tuple(range(action_dim)),
            ),
            episode_nonce=episode_nonce,
        )
    reference = _prepare_arm_detector_trace(reference)
    best_score = float("-inf")
    best_lag = 0
    best_rate = 1.0
    best_per_dim: np.ndarray | None = None
    for rate in tuple(rate_search_factors) or (1.0,):
        warped_reference = wm._resample_trace(reference, rate)
        for lag in range(-int(lag_search_steps), int(lag_search_steps) + 1):
            shifted_reference = wm._shift_trace(warped_reference, lag)
            score, per_dim = _multichannel_band_coherence_score(
                telemetry,
                shifted_reference,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
            )
            if score > best_score:
                best_score = score
                best_lag = lag
                best_rate = rate
                best_per_dim = per_dim
    return wm.PresenceDetectionResult(
        score=best_score,
        detected=best_score >= threshold,
        threshold=threshold,
        best_lag=best_lag,
        best_rate=best_rate,
        per_dim_scores=best_per_dim,
    )


def _pair_telemetry_delta_rms(plain_result: RolloutResult, marked_result: RolloutResult) -> float:
    shared_steps = min(int(plain_result.telemetry.shape[0]), int(marked_result.telemetry.shape[0]))
    if shared_steps <= 0:
        return 0.0
    delta = np.asarray(marked_result.telemetry[:shared_steps] - plain_result.telemetry[:shared_steps], dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(delta))))


def _summarize_whitebox_pair(
    plain_result: RolloutResult,
    marked_result: RolloutResult,
    *,
    group_size: int,
) -> WhiteboxDiagnosticSummary:
    common_chunk_count = min(len(plain_result.chunk_traces), len(marked_result.chunk_traces))
    common_step_count = min(int(plain_result.steps), int(marked_result.steps))
    if common_chunk_count <= 0:
        return WhiteboxDiagnosticSummary(
            common_chunk_count=0,
            common_step_count=common_step_count,
            internal_noise_delta_rms=0.0,
            action_delta_rms=0.0,
            telemetry_delta_rms=_pair_telemetry_delta_rms(plain_result, marked_result),
            group_internal_to_action=np.zeros((int(group_size),), dtype=np.float32),
            group_internal_to_telemetry=np.zeros((int(group_size),), dtype=np.float32),
        )
    plain_traces = plain_result.chunk_traces[:common_chunk_count]
    marked_traces = marked_result.chunk_traces[:common_chunk_count]
    internal_delta = np.concatenate(
        [np.asarray(marked.applied_noise - plain.applied_noise, dtype=np.float32) for plain, marked in zip(plain_traces, marked_traces, strict=True)],
        axis=0,
    )
    action_delta = np.concatenate(
        [np.asarray(marked.predicted_actions - plain.predicted_actions, dtype=np.float32) for plain, marked in zip(plain_traces, marked_traces, strict=True)],
        axis=0,
    )
    internal_rms = float(np.sqrt(np.mean(np.square(internal_delta)))) if internal_delta.size else 0.0
    action_rms = float(np.sqrt(np.mean(np.square(action_delta)))) if action_delta.size else 0.0
    telemetry_rms = _pair_telemetry_delta_rms(plain_result, marked_result)
    action_ratio = action_rms / max(internal_rms, 1e-8)
    telemetry_ratio = telemetry_rms / max(internal_rms, 1e-8)
    return WhiteboxDiagnosticSummary(
        common_chunk_count=common_chunk_count,
        common_step_count=common_step_count,
        internal_noise_delta_rms=internal_rms,
        action_delta_rms=action_rms,
        telemetry_delta_rms=telemetry_rms,
        group_internal_to_action=np.full((int(group_size),), action_ratio, dtype=np.float32),
        group_internal_to_telemetry=np.full((int(group_size),), telemetry_ratio, dtype=np.float32),
    )


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    total = float(pos.size * neg.size)
    for pos_value in pos:
        wins += float(np.sum(pos_value > neg))
        wins += 0.5 * float(np.sum(pos_value == neg))
    return wins / total


def _calibrate_threshold_for_target_fpr(negative_scores: Sequence[float], target_fpr: float) -> float:
    neg = np.sort(np.asarray(negative_scores, dtype=np.float32))
    if neg.size == 0:
        return float("nan")
    keep = int(math.ceil((1.0 - float(target_fpr)) * neg.size)) - 1
    keep = int(np.clip(keep, 0, neg.size - 1))
    return float(neg[keep])


def _binary_metrics(positive_scores: Sequence[float], negative_scores: Sequence[float], threshold: float) -> tuple[float, float]:
    pos = np.asarray(positive_scores, dtype=np.float32)
    neg = np.asarray(negative_scores, dtype=np.float32)
    tpr = float(np.mean(pos >= threshold)) if pos.size else float("nan")
    fpr = float(np.mean(neg >= threshold)) if neg.size else float("nan")
    return tpr, fpr


def _describe_scores(name: str, scores: Sequence[float]) -> str:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return f"{name}: mean=nan std=nan min=nan max=nan"
    return (
        f"{name}: mean={float(np.mean(values)):.4f} "
        f"std={float(np.std(values)):.4f} "
        f"min={float(np.min(values)):.4f} "
        f"max={float(np.max(values)):.4f}"
    )
