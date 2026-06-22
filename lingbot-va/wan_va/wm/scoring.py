"""WMF detector and scoring utilities for watermark detection.

The WMF (Watermark Matched Filter) detector compares recovered noise against
keyed references in a whitened subspace derived from null (wrong-key) references.
"""

from __future__ import annotations

import numpy as np

from .watermark import (
    InternalNoiseWatermarkConfig,
    WatermarkContext,
    generate_keyed_reference,
)


def wmf_score_from_vectors(
    feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None = 3,
) -> float:
    """Compute WMF score: whitened projection of feature against null subspace.

    Args:
        feature: 1D vector (score for true key)
        null_matrix: [num_nulls, feature_dim] matrix of wrong-key scores
        subspace_rank: number of principal components to use

    Returns:
        scalar WMF score (higher = more likely watermarked)
    """
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0:
        return 0.0

    mu = np.mean(null_matrix, axis=0)
    centered_feature = feature - mu

    projected_feature, eigvecs, eigvals = _select_whitened_subspace(
        centered_feature, null_matrix, subspace_rank=subspace_rank
    )
    if projected_feature.size == 0:
        return 0.0

    template = np.sum(eigvecs, axis=0)
    whitened = projected_feature / np.sqrt(np.maximum(eigvals, 1e-8))
    template_whitened = template / np.sqrt(np.maximum(eigvals, 1e-8))
    return float(np.dot(template_whitened, whitened))


def build_score_vector_from_noise(
    recovered_noise: np.ndarray,
    *,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
    sample_rate_hz: float,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
) -> np.ndarray:
    """Compute per-dimension cosine similarity between recovered noise and reference.

    Args:
        recovered_noise: [C_total, F, H, 1] or [C_total, F, H] — recovered latent
        config: watermark config with secret_key
        context: chunk context (chunk_index, episode_nonce)
        sample_rate_hz: action sample rate
        active_channel_ids: active action channel indices
        frame_chunk_size: F
        action_per_frame: H

    Returns:
        1D array of per-dim cosine scores, length = len(active_channel_ids)
    """
    length = frame_chunk_size * action_per_frame
    action_dim = len(active_channel_ids)

    reference = generate_keyed_reference(
        length=length,
        action_dim=action_dim,
        sample_rate_hz=sample_rate_hz,
        config=config,
        context=context,
    )

    # Extract active channels from recovered noise and flatten temporal
    noise = np.asarray(recovered_noise, dtype=np.float32)
    if noise.ndim == 4:
        noise = noise[:, :, :, 0]  # drop W=1 dim
    # noise: [C_total, F, H] → select active → [C_active, F, H] → [C_active, F*H]
    active_noise = noise[active_channel_ids].reshape(action_dim, length)

    # reference: [length, action_dim] → [action_dim, length]
    ref = reference.T

    scores = []
    for d in range(action_dim):
        x = active_noise[d] - np.mean(active_noise[d])
        r = ref[d] - np.mean(ref[d])
        denom = np.linalg.norm(x) * np.linalg.norm(r)
        if denom < 1e-8:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(x, r) / denom))

    return np.asarray(scores, dtype=np.float32)


def build_null_score_matrix(
    recovered_noise: np.ndarray,
    *,
    true_config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
    sample_rate_hz: float,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
    null_count: int = 32,
) -> np.ndarray:
    """Build null matrix by scoring recovered noise against wrong-key references.

    Returns: [null_count, action_dim] matrix
    """
    null_vectors = []
    for i in range(null_count):
        wrong_key = true_config.secret_key + 1000 + i
        null_config = InternalNoiseWatermarkConfig(
            secret_key=wrong_key,
            control_freq=true_config.control_freq,
            beta=true_config.beta,
            freq_range=true_config.freq_range,
            n_tones=true_config.n_tones,
            watermark_dims=true_config.watermark_dims,
            reference_mode=true_config.reference_mode,
            chunk_selection_strategy=true_config.chunk_selection_strategy,
            chunk_selection_period=true_config.chunk_selection_period,
            chunk_selection_count=true_config.chunk_selection_count,
        )
        vec = build_score_vector_from_noise(
            recovered_noise,
            config=null_config,
            context=context,
            sample_rate_hz=sample_rate_hz,
            active_channel_ids=active_channel_ids,
            frame_chunk_size=frame_chunk_size,
            action_per_frame=action_per_frame,
        )
        null_vectors.append(vec)
    return np.stack(null_vectors, axis=0)


def score_chunk(
    recovered_noise: np.ndarray,
    *,
    config: InternalNoiseWatermarkConfig,
    context: WatermarkContext,
    sample_rate_hz: float,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
    null_count: int = 32,
    subspace_rank: int = 3,
) -> float:
    """Full WMF scoring for a single recovered chunk.

    Returns WMF score (scalar).
    """
    true_vector = build_score_vector_from_noise(
        recovered_noise,
        config=config,
        context=context,
        sample_rate_hz=sample_rate_hz,
        active_channel_ids=active_channel_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
    )
    null_matrix = build_null_score_matrix(
        recovered_noise,
        true_config=config,
        context=context,
        sample_rate_hz=sample_rate_hz,
        active_channel_ids=active_channel_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
        null_count=null_count,
    )
    return wmf_score_from_vectors(true_vector, null_matrix, subspace_rank=subspace_rank)


# --- Internal ---

def _select_whitened_subspace(
    centered_feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-based whitened subspace selection from null distribution."""
    centered_null = null_matrix - np.mean(null_matrix, axis=0, keepdims=True)
    if centered_null.shape[0] < 2 or centered_null.shape[1] < 1:
        return np.array([]), np.array([]), np.array([])

    cov = np.cov(centered_null, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top-k
    rank = subspace_rank if subspace_rank is not None else len(eigvals)
    rank = min(rank, len(eigvals), centered_null.shape[0] - 1)
    rank = max(rank, 1)

    eigvals = eigvals[:rank]
    eigvecs = eigvecs[:, :rank]

    # Project feature
    projected = centered_feature @ eigvecs  # [rank]
    return projected, eigvecs, eigvals
