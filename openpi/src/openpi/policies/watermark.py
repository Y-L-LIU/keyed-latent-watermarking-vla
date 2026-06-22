"""Watermark utilities for internal-noise and output-action experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Sequence

import numpy as np
import torch

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - jax may be unavailable in lightweight test environments.
    jnp = None


@dataclasses.dataclass(frozen=True)
class WatermarkContext:
    """Runtime context used to key the internal watermark reference.

    `obs_seed` enables observation-tied keying (the distillation-survivable mode):
    when set, the keyed reference is derived from this stable hash of the (quantized)
    observation instead of from the episode nonce / chunk index. Left ``None`` the
    behavior is unchanged (nonce-keyed, per-sample fingerprint).
    """

    chunk_index: int = 0
    episode_nonce: int = 0
    obs_seed: int | None = None


@dataclasses.dataclass(frozen=True)
class InternalNoiseWatermarkConfig:
    """Configuration for internal-noise watermark mixing.

    The reference is synthesized in the same temporal domain as the initial
    sampler noise. `control_freq` therefore refers to the control/sample rate of
    the action chunk, not to any post-decoder actuator hook.
    """

    secret_key: int
    control_freq: float
    beta: float = 0.02
    freq_range: tuple[float, float] = (0.5, 3.0)
    n_tones: int = 4
    watermark_dims: tuple[int, ...] | None = None
    reference_mode: str = "bandpass"
    chunk_selection_strategy: str = "periodic"
    chunk_selection_period: int = 1
    chunk_selection_count: int = 1
    chunk_selection_total_slots: int | None = None
    chunk_index_key: str = "chunk_index"
    episode_nonce_key: str | None = "episode_nonce"
    fallback_chunk_index_key: str | None = "global_step"
    # Observation-tied keying (distillation-survivable mode). When keying_mode ==
    # "observation" the reference is a deterministic function of the observation
    # rather than the per-episode nonce, so the keyed action warp survives behavior
    # cloning / distillation. `obs_key` selects which runtime-obs field to hash,
    # `obs_proj_dims` restricts it to a few stable coordinates (None = all), and
    # `obs_quantization` is the rounding resolution that bounds the bucket count so
    # the warp field stays learnable and robust to teacher/student trajectory drift.
    keying_mode: str = "nonce"
    obs_key: str = "observation/state"
    obs_proj_dims: tuple[int, ...] | None = None
    obs_quantization: float = 0.5


@dataclasses.dataclass(frozen=True)
class OutputActionWatermarkConfig:
    """Configuration for additive watermarking on final executed actions."""

    secret_key: int
    control_freq: float
    beta: float = 0.02
    family: str = "bandpass"
    freq_range: tuple[float, float] = (0.5, 3.0)
    watermark_dims: tuple[int, ...] | None = None
    code_type: str = "balanced_sign"
    detector: str = "coherence"


# Backwards-compatible alias for callers that were already importing the old
# config symbol. The semantics are now internal-noise watermarking only.
GlobalPhaseWatermarkConfig = InternalNoiseWatermarkConfig


@dataclasses.dataclass(frozen=True)
class PresenceDetectionResult:
    """Presence detector output.

    The score is a normalized narrow-band coherence between telemetry and the
    keyed reference projected directly in telemetry space. No decoder inversion
    or latent recovery is attempted.
    """

    score: float
    detected: bool
    threshold: float
    best_lag: int = 0
    best_rate: float = 1.0
    per_dim_scores: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class OutputActionApplyResult:
    """Result bundle for executed-action watermark injection."""

    watermarked_actions: np.ndarray
    reference: np.ndarray
    clip_fraction: float
    saturation_fraction: float
    delta_rms: float


def generate_keyed_reference(
    *,
    length: int,
    action_dim: int,
    sample_rate_hz: float,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
) -> np.ndarray:
    """Generate a deterministic band-limited reference for one action chunk."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if action_dim <= 0:
        raise ValueError("action_dim must be > 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if not (0.0 <= config.beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {config.beta}")
    if config.n_tones <= 0:
        raise ValueError("n_tones must be > 0")
    if config.reference_mode not in {"bandpass", "gaussian"}:
        raise ValueError(f"Unsupported reference_mode={config.reference_mode!r}")
    if config.chunk_selection_strategy not in {"periodic", "fixed_slots", "stateful_online"}:
        raise ValueError(f"Unsupported chunk_selection_strategy={config.chunk_selection_strategy!r}")
    if config.chunk_selection_period <= 0:
        raise ValueError("chunk_selection_period must be > 0")
    if config.chunk_selection_total_slots is not None and config.chunk_selection_total_slots <= 0:
        raise ValueError("chunk_selection_total_slots must be > 0 when provided")
    if config.chunk_selection_strategy == "periodic":
        if not (0 <= config.chunk_selection_count <= config.chunk_selection_period):
            raise ValueError("chunk_selection_count must be in [0, chunk_selection_period]")
    elif config.chunk_selection_strategy == "fixed_slots":
        if config.chunk_selection_total_slots is None:
            raise ValueError("chunk_selection_total_slots is required when chunk_selection_strategy='fixed_slots'")
        if not (0 <= config.chunk_selection_count <= config.chunk_selection_total_slots):
            raise ValueError("chunk_selection_count must be in [0, chunk_selection_total_slots]")
    elif config.chunk_selection_count < 0:
        raise ValueError("chunk_selection_count must be >= 0")

    f_min, f_max = config.freq_range
    if not (0.0 < f_min <= f_max < sample_rate_hz / 2.0):
        raise ValueError(
            f"freq_range={config.freq_range} must satisfy 0 < f_min <= f_max < Nyquist ({sample_rate_hz / 2.0})"
        )

    dims = _resolve_dims(action_dim, config.watermark_dims)
    reference = np.zeros((length, action_dim), dtype=np.float32)
    if not dims:
        return reference

    for dim in dims:
        seed = _stable_seed(config.secret_key, context, dim)
        if config.reference_mode == "bandpass":
            reference[:, dim] = _generate_band_passed_gaussian(
                seed=seed,
                length=length,
                sample_rate_hz=sample_rate_hz,
                freq_range=config.freq_range,
            )
        else:
            reference[:, dim] = _generate_gaussian_reference(seed=seed, length=length)

    return reference


def generate_output_action_reference(
    *,
    length: int,
    action_dim: int,
    sample_rate_hz: float,
    config: OutputActionWatermarkConfig,
    context: WatermarkContext,
) -> np.ndarray:
    """Generate a deterministic reference for output-action watermarking."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if action_dim <= 0:
        raise ValueError("action_dim must be > 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if not (0.0 <= config.beta):
        raise ValueError(f"beta must be >= 0, got {config.beta}")
    if config.family not in {"bandpass", "timecode"}:
        raise ValueError(f"Unsupported family={config.family!r}")
    if config.detector not in {"coherence", "matched_filter", "glrt"}:
        raise ValueError(f"Unsupported detector={config.detector!r}")

    dims = _resolve_dims(action_dim, config.watermark_dims)
    reference = np.zeros((length, action_dim), dtype=np.float32)
    if not dims:
        return reference

    if config.family == "bandpass":
        f_min, f_max = config.freq_range
        if not (0.0 < f_min <= f_max < sample_rate_hz / 2.0):
            raise ValueError(
                f"freq_range={config.freq_range} must satisfy 0 < f_min <= f_max < Nyquist ({sample_rate_hz / 2.0})"
            )

    for dim in dims:
        seed = _stable_seed(config.secret_key, context, dim)
        if config.family == "bandpass":
            reference[:, dim] = _generate_band_passed_gaussian(
                seed=seed,
                length=length,
                sample_rate_hz=sample_rate_hz,
                freq_range=config.freq_range,
            )
        else:
            reference[:, dim] = _generate_timecode_reference(seed=seed, length=length, code_type=config.code_type)
    return reference


def apply_output_action_watermark(
    base_actions: np.ndarray,
    *,
    sample_rate_hz: float,
    config: OutputActionWatermarkConfig,
    context: WatermarkContext,
    clip_low: float | np.ndarray | None = None,
    clip_high: float | np.ndarray | None = None,
) -> OutputActionApplyResult:
    """Add a keyed watermark directly to final executed actions."""
    actions = np.asarray(base_actions, dtype=np.float32)
    squeeze = False
    if actions.ndim == 1:
        actions = actions[None, :]
        squeeze = True
    if actions.ndim != 2:
        raise ValueError(f"base_actions must have rank 1 or 2, got shape={actions.shape}")

    reference = generate_output_action_reference(
        length=actions.shape[0],
        action_dim=actions.shape[1],
        sample_rate_hz=sample_rate_hz,
        config=config,
        context=context,
    )
    watermarked = actions + float(config.beta) * reference
    unclipped = watermarked.copy()
    if clip_low is not None or clip_high is not None:
        low = -np.inf if clip_low is None else clip_low
        high = np.inf if clip_high is None else clip_high
        watermarked = np.clip(watermarked, low, high)
        clip_mask = np.abs(watermarked - unclipped) > 1e-6
        clip_fraction = float(np.mean(clip_mask))
        sat_mask = np.zeros_like(watermarked, dtype=bool)
        if clip_low is not None:
            sat_mask |= watermarked <= np.asarray(clip_low, dtype=np.float32) + 1e-6
        if clip_high is not None:
            sat_mask |= watermarked >= np.asarray(clip_high, dtype=np.float32) - 1e-6
        saturation_fraction = float(np.mean(sat_mask))
    else:
        clip_fraction = 0.0
        saturation_fraction = 0.0

    delta_rms = float(np.sqrt(np.mean(np.square(watermarked - actions)))) if watermarked.size else 0.0
    if squeeze:
        actions_out = watermarked[0]
        reference_out = reference[0]
    else:
        actions_out = watermarked
        reference_out = reference
    return OutputActionApplyResult(
        watermarked_actions=np.asarray(actions_out, dtype=np.float32),
        reference=np.asarray(reference_out, dtype=np.float32),
        clip_fraction=clip_fraction,
        saturation_fraction=saturation_fraction,
        delta_rms=delta_rms,
    )


def generate_reference_trace(
    *,
    total_length: int,
    action_dim: int,
    sample_rate_hz: float,
    chunk_size: int,
    config: InternalNoiseWatermarkConfig,
    episode_nonce: int = 0,
    start_chunk_index: int = 0,
) -> np.ndarray:
    """Concatenate chunk-wise keyed references across a full telemetry trace."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    trace = np.zeros((total_length, action_dim), dtype=np.float32)
    offset = 0
    chunk_index = int(start_chunk_index)
    while offset < total_length:
        window = min(chunk_size, total_length - offset)
        context = WatermarkContext(chunk_index=chunk_index, episode_nonce=episode_nonce)
        if should_watermark_chunk(config, context):
            trace[offset : offset + window] = generate_keyed_reference(
                length=window,
                action_dim=action_dim,
                sample_rate_hz=sample_rate_hz,
                config=config,
                context=context,
            )
        offset += window
        chunk_index += 1
    return trace


def mix_internal_noise(
    base_noise,
    *,
    sample_rate_hz: float,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
):
    """Mix a keyed reference into the sampler's internal noise carrier.

    This is the only supported injection path for the MVP:

        omega_wm = sqrt(1 - beta^2) * omega_base + beta * r_keyed

    The function preserves shape, dtype family, and framework where possible.
    """
    noise_np, restore = _to_numpy_and_restore(base_noise)

    squeeze_batch = False
    if noise_np.ndim == 2:
        noise_np = noise_np[None, ...]
        squeeze_batch = True
    if noise_np.ndim != 3:
        raise ValueError(f"base_noise must have rank 2 or 3, got shape={noise_np.shape}")

    mixed = noise_np.astype(np.float32, copy=True)
    dims = _resolve_dims(mixed.shape[-1], config.watermark_dims)
    if not dims or config.beta == 0.0:
        out = mixed[0] if squeeze_batch else mixed
        return restore(out)
    if not should_watermark_chunk(config, context):
        out = mixed[0] if squeeze_batch else mixed
        return restore(out)

    reference = generate_keyed_reference(
        length=mixed.shape[1],
        action_dim=mixed.shape[2],
        sample_rate_hz=sample_rate_hz,
        config=config,
        context=context,
    )
    beta = float(config.beta)
    alpha = math.sqrt(max(0.0, 1.0 - beta * beta))
    mixed[:, :, dims] = alpha * mixed[:, :, dims] + beta * reference[None, :, dims]

    out = mixed[0] if squeeze_batch else mixed
    return restore(out)


def narrow_band_coherency_score(
    telemetry: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    watermark_dims: Sequence[int] | None = None,
) -> tuple[float, np.ndarray]:
    """Compute a normalized band-limited coherence score.

    The detector compares the keyed reference directly against external motor or
    joint telemetry. It assumes the watermark survives the unknown decoder and
    plant dynamics strongly enough to preserve some narrow-band correlation.
    """
    telemetry = _coerce_trace(telemetry)
    reference = _coerce_trace(reference)
    if telemetry.shape != reference.shape:
        raise ValueError(f"telemetry/reference shape mismatch: {telemetry.shape} vs {reference.shape}")

    dims = _resolve_dims(telemetry.shape[1], watermark_dims)
    if not dims:
        return 0.0, np.zeros((0,), dtype=np.float32)

    telemetry_bp = _band_limit(telemetry, sample_rate_hz=sample_rate_hz, freq_range=freq_range)
    reference_bp = _band_limit(reference, sample_rate_hz=sample_rate_hz, freq_range=freq_range)

    per_dim = []
    for dim in dims:
        x = telemetry_bp[:, dim] - np.mean(telemetry_bp[:, dim])
        r = reference_bp[:, dim] - np.mean(reference_bp[:, dim])
        denom = np.linalg.norm(x) * np.linalg.norm(r)
        if denom < 1e-8:
            per_dim.append(0.0)
            continue
        per_dim.append(float(abs(np.dot(x, r)) / denom))
    scores = np.asarray(per_dim, dtype=np.float32)
    return float(np.mean(scores)), scores


def matched_filter_score(
    trace: np.ndarray,
    reference: np.ndarray,
    *,
    watermark_dims: Sequence[int] | None = None,
) -> tuple[float, np.ndarray]:
    """Compute a per-dimension normalized matched-filter score."""
    trace = _coerce_trace(trace)
    reference = _coerce_trace(reference)
    if trace.shape != reference.shape:
        raise ValueError(f"trace/reference shape mismatch: {trace.shape} vs {reference.shape}")

    dims = _resolve_dims(trace.shape[1], watermark_dims)
    if not dims:
        return 0.0, np.zeros((0,), dtype=np.float32)

    per_dim = []
    for dim in dims:
        x = trace[:, dim] - np.mean(trace[:, dim])
        r = reference[:, dim] - np.mean(reference[:, dim])
        denom = np.linalg.norm(r)
        if denom < 1e-8:
            per_dim.append(0.0)
            continue
        per_dim.append(float(abs(np.dot(x, r)) / denom))
    scores = np.asarray(per_dim, dtype=np.float32)
    return float(np.mean(scores)), scores


def glrt_score(
    trace: np.ndarray,
    reference: np.ndarray,
    *,
    watermark_dims: Sequence[int] | None = None,
) -> tuple[float, np.ndarray]:
    """Compute a simple GLRT-style score with unknown amplitude and noise variance."""
    trace = _coerce_trace(trace)
    reference = _coerce_trace(reference)
    if trace.shape != reference.shape:
        raise ValueError(f"trace/reference shape mismatch: {trace.shape} vs {reference.shape}")

    dims = _resolve_dims(trace.shape[1], watermark_dims)
    if not dims:
        return 0.0, np.zeros((0,), dtype=np.float32)

    per_dim = []
    for dim in dims:
        x = trace[:, dim] - np.mean(trace[:, dim])
        r = reference[:, dim] - np.mean(reference[:, dim])
        ref_energy = float(np.dot(r, r))
        if ref_energy < 1e-8:
            per_dim.append(0.0)
            continue
        amplitude = float(np.dot(x, r) / ref_energy)
        residual = x - amplitude * r
        noise_scale = float(np.sqrt(np.mean(np.square(residual))))
        if noise_scale < 1e-8:
            noise_scale = 1e-8
        per_dim.append(float(abs(amplitude) * np.sqrt(ref_energy) / noise_scale))
    scores = np.asarray(per_dim, dtype=np.float32)
    return float(np.mean(scores)), scores


def detect_output_action_watermark(
    trace: np.ndarray,
    *,
    sample_rate_hz: float,
    config: OutputActionWatermarkConfig,
    context: WatermarkContext,
    action_dim: int | None = None,
    threshold: float = 0.0,
    lag_search_steps: int = 0,
    rate_search_factors: Sequence[float] = (1.0,),
) -> PresenceDetectionResult:
    """Score an output-action watermark from executed outputs or telemetry."""
    trace = _coerce_trace(trace)
    action_dim = action_dim or trace.shape[1]
    reference = generate_output_action_reference(
        length=trace.shape[0],
        action_dim=action_dim,
        sample_rate_hz=sample_rate_hz,
        config=config,
        context=context,
    )

    best_score = float("-inf")
    best_lag = 0
    best_rate = 1.0
    best_per_dim: np.ndarray | None = None
    for rate in tuple(rate_search_factors) or (1.0,):
        warped_reference = _resample_trace(reference, rate)
        for lag in range(-lag_search_steps, lag_search_steps + 1):
            shifted_reference = _shift_trace(warped_reference, lag)
            if config.detector == "coherence":
                score, per_dim = narrow_band_coherency_score(
                    trace,
                    shifted_reference,
                    sample_rate_hz=sample_rate_hz,
                    freq_range=config.freq_range,
                    watermark_dims=config.watermark_dims,
                )
            elif config.detector == "matched_filter":
                score, per_dim = matched_filter_score(
                    trace,
                    shifted_reference,
                    watermark_dims=config.watermark_dims,
                )
            else:
                score, per_dim = glrt_score(
                    trace,
                    shifted_reference,
                    watermark_dims=config.watermark_dims,
                )
            if score > best_score:
                best_score = score
                best_lag = lag
                best_rate = rate
                best_per_dim = per_dim
    return PresenceDetectionResult(
        score=best_score,
        detected=best_score >= threshold,
        threshold=threshold,
        best_lag=best_lag,
        best_rate=best_rate,
        per_dim_scores=best_per_dim,
    )


def detect_watermark_presence(
    telemetry: np.ndarray,
    *,
    secret_key: int,
    sample_rate_hz: float,
    chunk_size: int,
    action_dim: int | None = None,
    freq_range: tuple[float, float] = (0.5, 3.0),
    n_tones: int = 4,
    watermark_dims: Sequence[int] | None = None,
    episode_nonce: int = 0,
    start_chunk_index: int = 0,
    threshold: float = 0.5,
    lag_search_steps: int = 0,
    rate_search_factors: Sequence[float] = (1.0,),
) -> PresenceDetectionResult:
    """Run telemetry-only presence detection.

    Assumptions and limitations:
    - The detector only sees external telemetry and the secret key.
    - The detector replays the same keyed band-limited reference used on the
      sampler noise and never attempts to invert the policy decoder or latent.
    - Chunk cadence and sample rate are assumed known.
    - Any lag/rate search is intentionally small; this is a minimal presence
      detector, not attribution or owner verification.
    """
    telemetry = _coerce_trace(telemetry)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    action_dim = action_dim or telemetry.shape[1]
    config = InternalNoiseWatermarkConfig(
        secret_key=secret_key,
        control_freq=sample_rate_hz,
        beta=0.0,
        freq_range=freq_range,
        n_tones=n_tones,
        watermark_dims=tuple(watermark_dims) if watermark_dims is not None else None,
    )
    reference = generate_reference_trace(
        total_length=telemetry.shape[0],
        action_dim=action_dim,
        sample_rate_hz=sample_rate_hz,
        chunk_size=chunk_size,
        config=config,
        episode_nonce=episode_nonce,
        start_chunk_index=start_chunk_index,
    )

    best_score = float("-inf")
    best_lag = 0
    best_rate = 1.0
    best_per_dim: np.ndarray | None = None
    search_rates = tuple(rate_search_factors) or (1.0,)
    for rate in search_rates:
        warped_reference = _resample_trace(reference, rate)
        for lag in range(-lag_search_steps, lag_search_steps + 1):
            shifted_reference = _shift_trace(warped_reference, lag)
            score, per_dim = narrow_band_coherency_score(
                telemetry,
                shifted_reference,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
                watermark_dims=watermark_dims,
            )
            if score > best_score:
                best_score = score
                best_lag = lag
                best_rate = rate
                best_per_dim = per_dim

    return PresenceDetectionResult(
        score=best_score,
        detected=best_score >= threshold,
        threshold=threshold,
        best_lag=best_lag,
        best_rate=best_rate,
        per_dim_scores=best_per_dim,
    )


def inject_global_phase_watermark(*args, **kwargs):
    """Action-side watermarking has been intentionally removed.

    This function remains only to fail loudly if old code paths try to add a
    watermark after decode/output, which the MVP explicitly forbids.
    """
    raise RuntimeError(
        "Action-side watermarking after decode/output is disabled. "
        "Use mix_internal_noise(...) before model.sample_actions(..., noise=...)."
    )


def should_watermark_chunk(config: InternalNoiseWatermarkConfig, context: WatermarkContext) -> bool:
    """Deterministically choose whether a chunk should carry watermark energy."""
    if config.chunk_selection_count == 0:
        return False
    if config.chunk_selection_strategy == "fixed_slots":
        if config.chunk_selection_total_slots is None:
            raise ValueError("chunk_selection_total_slots is required when chunk_selection_strategy='fixed_slots'")
        if context.chunk_index < 0 or context.chunk_index >= config.chunk_selection_total_slots:
            return False
        selected = _fixed_count_selected_chunk_indices(
            secret_key=config.secret_key,
            episode_nonce=context.episode_nonce,
            total_slots=config.chunk_selection_total_slots,
            count=config.chunk_selection_count,
        )
        return context.chunk_index in selected
    if config.chunk_selection_strategy == "stateful_online":
        return _stateful_online_should_watermark_chunk(
            secret_key=config.secret_key,
            episode_nonce=context.episode_nonce,
            chunk_index=context.chunk_index,
            count=config.chunk_selection_count,
            max_gap=config.chunk_selection_period,
        )
    if config.chunk_selection_count >= config.chunk_selection_period:
        return True
    bucket = _stable_bucket(config.secret_key, context, "chunk-selector", config.chunk_selection_period)
    return bucket < config.chunk_selection_count


def _fixed_count_selected_chunk_indices(*, secret_key: int, episode_nonce: int, total_slots: int, count: int) -> set[int]:
    if total_slots <= 0 or count <= 0:
        return set()
    count = min(int(count), int(total_slots))
    base_context = WatermarkContext(chunk_index=0, episode_nonce=episode_nonce)
    offset = _stable_bucket(secret_key, base_context, "chunk-selector-offset", total_slots)
    stride = _stable_bucket(secret_key, base_context, "chunk-selector-stride", total_slots - 1) + 1 if total_slots > 1 else 1
    while math.gcd(stride, total_slots) != 1:
        stride += 1
        if stride >= total_slots:
            stride = 1
    return {int((offset + i * stride) % total_slots) for i in range(count)}


def _stateful_online_should_watermark_chunk(
    *,
    secret_key: int,
    episode_nonce: int,
    chunk_index: int,
    count: int,
    max_gap: int,
) -> bool:
    if chunk_index < 0 or count <= 0:
        return False
    next_index = _stateful_online_first_index(secret_key=secret_key, episode_nonce=episode_nonce, max_gap=max_gap)
    if next_index == chunk_index:
        return True
    for selection_index in range(1, count):
        next_index += _stateful_online_gap(
            secret_key=secret_key,
            episode_nonce=episode_nonce,
            selection_index=selection_index,
            max_gap=max_gap,
        )
        if next_index == chunk_index:
            return True
        if next_index > chunk_index:
            return False
    return False


def _stateful_online_first_index(*, secret_key: int, episode_nonce: int, max_gap: int) -> int:
    context = WatermarkContext(chunk_index=0, episode_nonce=episode_nonce)
    return _stable_bucket(secret_key, context, "chunk-selector-online-offset", max_gap)


def _stateful_online_gap(*, secret_key: int, episode_nonce: int, selection_index: int, max_gap: int) -> int:
    context = WatermarkContext(chunk_index=selection_index, episode_nonce=episode_nonce)
    return 1 + _stable_bucket(secret_key, context, "chunk-selector-online-gap", max_gap)


def _resolve_dims(action_dim: int, watermark_dims: Sequence[int] | None) -> tuple[int, ...]:
    if watermark_dims is None:
        return tuple(range(action_dim))
    dims = tuple(int(dim) for dim in watermark_dims)
    if dims and (min(dims) < 0 or max(dims) >= action_dim):
        raise ValueError(f"watermark_dims={dims} out of range for action_dim={action_dim}")
    return dims


def _generate_gaussian_reference(*, seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(length, dtype=np.float32)
    signal = signal - np.mean(signal)
    std = float(np.std(signal))
    if std < 1e-6:
        std = 1.0
    return (signal / std).astype(np.float32, copy=False)


def _generate_timecode_reference(*, seed: int, length: int, code_type: str) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be > 0")
    if code_type not in {"balanced_sign", "sign", "prbs"}:
        raise ValueError(f"Unsupported code_type={code_type!r}")

    rng = np.random.default_rng(seed)
    if code_type == "balanced_sign":
        half = length // 2
        signal = np.concatenate(
            [
                np.ones((half,), dtype=np.float32),
                -np.ones((length - half,), dtype=np.float32),
            ]
        )
        rng.shuffle(signal)
    else:
        signal = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length, replace=True)
    signal = signal - np.mean(signal)
    std = float(np.std(signal))
    if std < 1e-6:
        std = 1.0
    return (signal / std).astype(np.float32, copy=False)


def compute_obs_seed(
    feature: np.ndarray | Sequence[float],
    *,
    quantization: float,
    proj_dims: Sequence[int] | None = None,
) -> int:
    """Hash a low-dimensional observation feature into a stable integer seed.

    Used by observation-tied keying: the keyed reference becomes a deterministic
    function of the (quantized) observation instead of the episode nonce, so the
    keyed action warp is a consistent function of the input that a behavior-cloned /
    distilled student inherits. The same function is called at injection time (to key
    the teacher) and at verification time (to recompute the reference from the
    suspect's recorded observation), so it must be bit-stable across both.

    Quantization rounds each selected coordinate to a coarse grid: this makes the
    seed robust to the small state differences between the teacher's and a distilled
    student's trajectories and bounds the number of distinct references so the warp
    field stays learnable rather than degenerating to a per-timestep one-time pad.
    """
    if quantization <= 0:
        raise ValueError("quantization must be > 0")
    arr = np.asarray(feature, dtype=np.float64).reshape(-1)
    if proj_dims is not None:
        idx = [int(d) for d in proj_dims]
        if idx and (min(idx) < 0 or max(idx) >= arr.shape[0]):
            raise ValueError(f"obs proj_dims={idx} out of range for feature of length {arr.shape[0]}")
        arr = arr[idx]
    buckets = np.round(arr / float(quantization)).astype(np.int64)
    payload = ("obs:" + ":".join(str(int(b)) for b in buckets.tolist())).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_seed(secret_key: int, context: WatermarkContext, dim: int) -> int:
    if context.obs_seed is not None:
        # Observation-tied keying: the reference depends only on the (quantized)
        # observation and the channel, NOT on the episode nonce or chunk index, so
        # the keyed action warp is a consistent function of the input that survives
        # distillation / behavior cloning.
        payload = f"{int(secret_key)}:obs:{int(context.obs_seed)}:{int(dim)}".encode("utf-8")
    else:
        payload = f"{int(secret_key)}:{int(context.episode_nonce)}:{int(context.chunk_index)}:{int(dim)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_bucket(secret_key: int, context: WatermarkContext, label: str, modulus: int) -> int:
    payload = f"{int(secret_key)}:{int(context.episode_nonce)}:{int(context.chunk_index)}:{label}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % modulus


def _generate_band_passed_gaussian(
    *,
    seed: int,
    length: int,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be > 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    f_min, f_max = freq_range
    nyquist = sample_rate_hz / 2.0
    if not (0.0 < f_min <= f_max < nyquist):
        raise ValueError(f"freq_range={freq_range} must satisfy 0 < f_min <= f_max < Nyquist ({nyquist})")

    rng = np.random.default_rng(seed)
    fft_length = max(length, int(math.ceil((8.0 * sample_rate_hz) / max(f_min, 1e-6))))
    fft_length = max(fft_length, 32)
    if fft_length % 2 == 1:
        fft_length += 1

    white = rng.standard_normal(fft_length).astype(np.float32)
    white = white - np.mean(white)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate_hz)
    band_mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band_mask):
        return np.zeros((length,), dtype=np.float32)

    spectrum[~band_mask] = 0.0
    colored = np.fft.irfft(spectrum, n=fft_length).astype(np.float32, copy=False)[:length]
    colored = colored - np.mean(colored)
    std = np.std(colored)
    if std < 1e-6:
        std = 1.0
    return (colored / std).astype(np.float32, copy=False)


def _coerce_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim == 1:
        trace = trace[:, None]
    if trace.ndim != 2:
        raise ValueError(f"Expected telemetry/reference with rank 1 or 2, got shape={trace.shape}")
    return trace


def _band_limit(trace: np.ndarray, *, sample_rate_hz: float, freq_range: tuple[float, float]) -> np.ndarray:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    trace = _coerce_trace(trace)
    freqs = np.fft.rfftfreq(trace.shape[0], d=1.0 / sample_rate_hz)
    f_min, f_max = freq_range
    mask = (freqs >= f_min) & (freqs <= f_max)

    spectrum = np.fft.rfft(trace - np.mean(trace, axis=0, keepdims=True), axis=0)
    spectrum[~mask, :] = 0.0
    return np.fft.irfft(spectrum, n=trace.shape[0], axis=0).astype(np.float32, copy=False)


def _shift_trace(trace: np.ndarray, lag: int) -> np.ndarray:
    trace = _coerce_trace(trace)
    if lag == 0:
        return trace
    shifted = np.zeros_like(trace)
    if lag > 0:
        shifted[lag:] = trace[:-lag]
    else:
        shifted[:lag] = trace[-lag:]
    return shifted


def _resample_trace(trace: np.ndarray, rate: float) -> np.ndarray:
    trace = _coerce_trace(trace)
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if abs(rate - 1.0) < 1e-8:
        return trace

    source_positions = np.arange(trace.shape[0], dtype=np.float64)
    warped_positions = source_positions / rate
    resampled = np.zeros_like(trace)
    for dim in range(trace.shape[1]):
        resampled[:, dim] = np.interp(warped_positions, source_positions, trace[:, dim], left=0.0, right=0.0)
    return resampled


def _to_numpy_and_restore(array):
    if isinstance(array, np.ndarray):
        original_dtype = array.dtype
        return np.asarray(array), lambda out: np.asarray(out, dtype=original_dtype)

    if isinstance(array, torch.Tensor):
        original_dtype = array.dtype
        device = array.device
        return array.detach().cpu().numpy(), lambda out: torch.as_tensor(out, dtype=original_dtype, device=device)

    # Treat unknown array-likes as JAX arrays if jax is available, otherwise
    # fall back to plain numpy reconstruction.
    out = np.asarray(array)
    if jnp is not None:
        original_dtype = getattr(array, "dtype", out.dtype)
        return out, lambda restored: jnp.asarray(restored, dtype=original_dtype)
    return out, np.asarray
