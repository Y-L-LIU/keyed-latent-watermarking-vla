"""Watermark utilities for internal-noise watermarking on LingBot-VA.

Adapted from OpenPI's watermark.py. The key difference is tensor shape handling:
- OpenPI: noise shape [B, T, C] (batch, horizon, action_dim)
- LingBot-VA: noise shape [B, C, F, H, 1] (batch, action_dim, frames, steps_per_frame, 1)
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Sequence

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class WatermarkContext:
    chunk_index: int = 0
    episode_nonce: int = 0
    obs_seed: int | None = None  # set -> observation-tied keying (distillation-survivable mode)


@dataclasses.dataclass(frozen=True)
class InternalNoiseWatermarkConfig:
    secret_key: int
    control_freq: float
    beta: float = 1.0
    freq_range: tuple[float, float] = (0.5, 3.0)
    n_tones: int = 4
    watermark_dims: tuple[int, ...] | None = None
    reference_mode: str = "gaussian"
    chunk_selection_strategy: str = "stateful_online"
    chunk_selection_period: int = 6
    chunk_selection_count: int = 5
    chunk_selection_total_slots: int | None = None
    chunk_start_min: int = 2
    # Observation-tied keying (distillation-survivable mode). See openpi watermark.py.
    keying_mode: str = "nonce"
    obs_proj_dims: tuple[int, ...] | None = None
    obs_quantization: float = 0.5


def generate_keyed_reference(
    *,
    length: int,
    action_dim: int,
    sample_rate_hz: float,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
) -> np.ndarray:
    """Generate a deterministic reference for one action chunk.

    Returns shape [length, action_dim].
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if action_dim <= 0:
        raise ValueError("action_dim must be > 0")

    dims = _resolve_dims(action_dim, config.watermark_dims)
    reference = np.zeros((length, action_dim), dtype=np.float32)
    if not dims:
        return reference

    if config.reference_mode == "dc":
        # pi0.5-faithful DC seed (path B): ONE (secret_key, obs-bucket)-keyed constant
        # vector over the active dims, tiled over the whole chunk -> non-zero-mean, the
        # component behavior cloning retains. Byte-identical to distill/dc_keying.dc_offset
        # so injection here and the raw matched-filter detector cannot drift from the pi0.5
        # latent-DC arm. The key index is the obs bucket (obs_seed), already folded mod
        # N_KEYS by the caller; fall back to a nonce bucket if obs keying is disabled.
        bucket = (
            context.obs_seed
            if context.obs_seed is not None
            else _stable_bucket(config.secret_key, context, "dc-nonce", 2**31)
        )
        c = _dc_offset_vector(config.secret_key, bucket, len(dims))
        for j, dim in enumerate(dims):
            reference[:, dim] = np.float32(c[j])
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


def mix_internal_noise(
    base_noise: torch.Tensor,
    *,
    sample_rate_hz: float,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
    active_channel_ids: Sequence[int],
    frame_chunk_size: int,
    action_per_frame: int,
) -> torch.Tensor:
    """Mix a keyed reference into the action sampler's initial noise.

    Formula: z_wm = sqrt(1 - beta^2) * z_base + beta * r_keyed

    Args:
        base_noise: [B, C_total, F, H, 1] — full 30-dim action noise
        active_channel_ids: which channels carry actual action (e.g., [0..7])
        frame_chunk_size: F dimension
        action_per_frame: H dimension
    Returns:
        watermarked noise, same shape as base_noise
    """
    if config.beta == 0.0:
        return base_noise
    if not should_watermark_chunk(config, context):
        return base_noise

    length = frame_chunk_size * action_per_frame
    action_dim = len(active_channel_ids)

    reference = generate_keyed_reference(
        length=length,
        action_dim=action_dim,
        sample_rate_hz=sample_rate_hz,
        config=config,
        context=context,
    )

    # reference shape: [length, action_dim] → [action_dim, F, H, 1]
    ref_tensor = torch.from_numpy(reference).to(
        device=base_noise.device, dtype=base_noise.dtype
    )
    ref_tensor = ref_tensor.T.reshape(action_dim, frame_chunk_size, action_per_frame, 1)

    beta = float(config.beta)
    alpha = math.sqrt(max(0.0, 1.0 - beta * beta))

    mixed = base_noise.clone()
    for i, ch in enumerate(active_channel_ids):
        mixed[:, ch] = alpha * base_noise[:, ch] + beta * ref_tensor[i]

    return mixed


def should_watermark_chunk(config: InternalNoiseWatermarkConfig, context: WatermarkContext) -> bool:
    if config.chunk_selection_count == 0:
        return False
    if config.chunk_selection_strategy == "stateful_online":
        return _stateful_online_should_watermark_chunk(
            secret_key=config.secret_key,
            episode_nonce=context.episode_nonce,
            chunk_index=context.chunk_index,
            count=config.chunk_selection_count,
            max_gap=config.chunk_selection_period,
            start_min=config.chunk_start_min,
        )
    if config.chunk_selection_strategy == "periodic":
        if config.chunk_selection_count >= config.chunk_selection_period:
            return True
        bucket = _stable_bucket(config.secret_key, context, "chunk-selector", config.chunk_selection_period)
        return bucket < config.chunk_selection_count
    if config.chunk_selection_strategy == "fixed_slots":
        if config.chunk_selection_total_slots is None:
            raise ValueError("chunk_selection_total_slots required for fixed_slots strategy")
        if context.chunk_index < 0 or context.chunk_index >= config.chunk_selection_total_slots:
            return False
        selected = _fixed_count_selected_chunk_indices(
            secret_key=config.secret_key,
            episode_nonce=context.episode_nonce,
            total_slots=config.chunk_selection_total_slots,
            count=config.chunk_selection_count,
        )
        return context.chunk_index in selected
    raise ValueError(f"Unsupported chunk_selection_strategy={config.chunk_selection_strategy!r}")


# --- Internal helpers ---

def _resolve_dims(action_dim: int, watermark_dims: Sequence[int] | None) -> tuple[int, ...]:
    if watermark_dims is None:
        return tuple(range(action_dim))
    return tuple(int(d) for d in watermark_dims)


def compute_obs_seed(feature, *, quantization: float, proj_dims=None) -> int:
    """Stable integer seed from a quantized observation feature (obs-tied keying).

    Same scheme as openpi watermark.compute_obs_seed: project to a few stable dims,
    quantize coarsely, hash. Called identically at injection and verification.
    """
    if quantization <= 0:
        raise ValueError("quantization must be > 0")
    arr = np.asarray(feature, dtype=np.float64).reshape(-1)
    if proj_dims is not None:
        idx = [int(d) for d in proj_dims]
        if idx and (min(idx) < 0 or max(idx) >= arr.shape[0]):
            raise ValueError(f"obs proj_dims={idx} out of range for feature length {arr.shape[0]}")
        arr = arr[idx]
    buckets = np.round(arr / float(quantization)).astype(np.int64)
    payload = ("obs:" + ":".join(str(int(b)) for b in buckets.tolist())).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)


def _stable_seed(secret_key: int, context: WatermarkContext, dim: int) -> int:
    if context.obs_seed is not None:
        payload = f"{int(secret_key)}:obs:{int(context.obs_seed)}:{int(dim)}".encode()
    else:
        payload = f"{int(secret_key)}:{int(context.episode_nonce)}:{int(context.chunk_index)}:{int(dim)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_bucket(secret_key: int, context: WatermarkContext, label: str, modulus: int) -> int:
    payload = f"{int(secret_key)}:{int(context.episode_nonce)}:{int(context.chunk_index)}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % modulus


def _dc_offset_vector(secret_key: int, task_seed: int, n: int) -> np.ndarray:
    """Constant (NON-zero-mean) per-dim offset vector keyed on (secret_key, task_seed).

    Byte-identical to distill/dc_keying.dc_offset (same blake2b digest, mod 2**32, default_rng,
    standard_normal) so the LingBot path-B injection/detection pair matches the pi0.5 latent-DC
    arm exactly. This is the distillation-survivable component: a zero-mean reference is averaged
    away by behavior cloning, whereas a constant DC offset is the part the student's conditional
    mean retains."""
    s = int.from_bytes(
        hashlib.blake2b(f"{int(secret_key)}:{int(task_seed)}".encode("utf-8"), digest_size=8).digest(),
        "little",
    ) % (2**32)
    return np.random.default_rng(s).standard_normal(n).astype(np.float64)


def _generate_gaussian_reference(*, seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(length).astype(np.float32)
    signal = signal - np.mean(signal)
    std = float(np.std(signal))
    if std < 1e-6:
        std = 1.0
    return (signal / std).astype(np.float32)


def _generate_band_passed_gaussian(
    *,
    seed: int,
    length: int,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> np.ndarray:
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
    colored = np.fft.irfft(spectrum, n=fft_length).astype(np.float32)[:length]
    colored = colored - np.mean(colored)
    std = np.std(colored)
    if std < 1e-6:
        std = 1.0
    return (colored / std).astype(np.float32)


def _stateful_online_should_watermark_chunk(
    *,
    secret_key: int,
    episode_nonce: int,
    chunk_index: int,
    count: int,
    max_gap: int,
    start_min: int = 0,
) -> bool:
    if chunk_index < 0 or count <= 0:
        return False
    next_index = _stateful_online_first_index(secret_key=secret_key, episode_nonce=episode_nonce, max_gap=max_gap)
    next_index = max(next_index, start_min)
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
