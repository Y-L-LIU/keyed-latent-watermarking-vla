"""Offline watermark detection: rescore saved NPZ rollouts and report binary classification metrics.

Loads paired rollout NPZs (plain + watermarked), computes episode-level WMF
z-scores using MAP-recovered noise, and reports ROC AUC / TPR@FPR.

Usage:
    python wan_va/wm/rescore_libero_watermark.py \
        --wm-dir outputs/wm_libero10/libero_10 \
        --plain-dir outputs/plain_libero10/libero_10 \
        --secret-key 42 --target-fpr 0.01
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ChunkTrace:
    chunk_index: int
    recovered_noise: np.ndarray   # [C_total, F, H, 1] — MAP z
    injected_noise: np.ndarray    # [C_total, F, H, 1] — actual WM noise used


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    task_id: int
    episode_idx: int
    episode_nonce: int
    variant: str  # "plain" or "watermarked"
    success: bool
    secret_key: int
    beta: float
    total_steps: int
    num_chunks: int
    chunk_watermarked_flags: np.ndarray
    chunk_traces: list[ChunkTrace]  # only for watermarked chunks with MAP


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def load_episode(path: Path) -> EpisodeRecord:
    data = np.load(str(path), allow_pickle=True)

    task_id = int(data["task_id"])
    episode_idx = int(data["episode_idx"])
    episode_nonce = int(data["episode_nonce"])
    success = bool(data["success"])
    secret_key = int(data["secret_key"])
    beta = float(data["beta"])
    total_steps = int(data["total_steps"])
    num_chunks = int(data["num_chunks"])
    chunk_watermarked_flags = np.asarray(data["chunk_watermarked_flags"], dtype=bool)

    # Variant: try from NPZ, fallback to beta-based heuristic
    if "variant" in data:
        variant = str(data["variant"])
    else:
        variant = "watermarked" if beta > 0 else "plain"

    # Load MAP traces for watermarked chunks
    chunk_traces = []
    wm_noises = data["chunk_wm_noises"]  # [N_chunks, 30, F, H, 1]
    wm_chunk_indices = np.where(chunk_watermarked_flags)[0]

    if "map_z" in data:
        map_z = data["map_z"]  # [N_wm, 30, F, H, 1]
        for i, chunk_idx in enumerate(wm_chunk_indices):
            if i >= len(map_z):
                break
            chunk_traces.append(ChunkTrace(
                chunk_index=int(chunk_idx),
                recovered_noise=np.asarray(map_z[i], dtype=np.float32),
                injected_noise=np.asarray(wm_noises[chunk_idx], dtype=np.float32),
            ))
    else:
        # No MAP data: use sampler noise directly as "recovered" noise.
        # Valid for plain (beta=0) episodes where the sampler noise IS the true z.
        for chunk_idx in wm_chunk_indices:
            chunk_traces.append(ChunkTrace(
                chunk_index=int(chunk_idx),
                recovered_noise=np.asarray(wm_noises[chunk_idx], dtype=np.float32),
                injected_noise=np.asarray(wm_noises[chunk_idx], dtype=np.float32),
            ))

    return EpisodeRecord(
        path=path,
        task_id=task_id,
        episode_idx=episode_idx,
        episode_nonce=episode_nonce,
        variant=variant,
        success=success,
        secret_key=secret_key,
        beta=beta,
        total_steps=total_steps,
        num_chunks=num_chunks,
        chunk_watermarked_flags=chunk_watermarked_flags,
        chunk_traces=chunk_traces,
    )


def collect_pairs(
    wm_dir: Path, plain_dir: Path
) -> list[tuple[EpisodeRecord, EpisodeRecord]]:
    """Pair plain and watermarked episodes by (task_id, episode_idx)."""
    wm_records = {
        (r.task_id, r.episode_idx): r
        for r in (load_episode(p) for p in sorted(wm_dir.glob("*.npz")))
    }
    plain_records = {
        (r.task_id, r.episode_idx): r
        for r in (load_episode(p) for p in sorted(plain_dir.glob("*.npz")))
    }

    pairs = []
    for key in sorted(set(wm_records) & set(plain_records)):
        pairs.append((plain_records[key], wm_records[key]))

    return pairs


# ---------------------------------------------------------------------------
# Scoring: per-chunk cosine similarity between recovered noise and reference
# ---------------------------------------------------------------------------

def _cosine_per_dim(
    recovered_noise: np.ndarray,
    reference: np.ndarray,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
) -> np.ndarray:
    """Per-dimension cosine between recovered noise and reference.

    Args:
        recovered_noise: [C_total, F, H, 1]
        reference: [length, action_dim] from generate_keyed_reference
        active_channel_ids: which channels to use
        frame_chunk_size: F
        action_per_frame: H

    Returns:
        [action_dim] array of per-dim cosine scores
    """
    action_dim = len(active_channel_ids)
    length = frame_chunk_size * action_per_frame

    # Extract active channels and flatten: [C_total, F, H, 1] → [action_dim, F*H]
    noise = np.asarray(recovered_noise, dtype=np.float32)
    if noise.ndim == 4:
        noise = noise[:, :, :, 0]  # [C_total, F, H]
    active_noise = noise[active_channel_ids].reshape(action_dim, length)

    # Reference: [length, action_dim] → [action_dim, length]
    ref = np.asarray(reference, dtype=np.float32).T

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


def episode_score_vector(
    record: EpisodeRecord,
    *,
    candidate_key: int,
    config_template,
    sample_rate_hz: float,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
) -> np.ndarray:
    """Compute concatenated per-chunk cosine vectors for an episode with a given key.

    Returns: [N_chunks * action_dim] feature vector.
    """
    from wan_va.wm.watermark import (
        InternalNoiseWatermarkConfig,
        WatermarkContext,
        generate_keyed_reference,
    )

    length = frame_chunk_size * action_per_frame
    action_dim = len(active_channel_ids)

    # Build config with candidate key
    config = InternalNoiseWatermarkConfig(
        secret_key=candidate_key,
        control_freq=config_template.control_freq,
        beta=config_template.beta,
        freq_range=config_template.freq_range,
        n_tones=config_template.n_tones,
        watermark_dims=config_template.watermark_dims,
        reference_mode=config_template.reference_mode,
        chunk_selection_strategy=config_template.chunk_selection_strategy,
        chunk_selection_period=config_template.chunk_selection_period,
        chunk_selection_count=config_template.chunk_selection_count,
        chunk_start_min=config_template.chunk_start_min,
    )

    vectors = []
    for trace in record.chunk_traces:
        context = WatermarkContext(
            chunk_index=trace.chunk_index,
            episode_nonce=record.episode_nonce,
        )
        reference = generate_keyed_reference(
            length=length,
            action_dim=action_dim,
            sample_rate_hz=sample_rate_hz,
            config=config,
            context=context,
        )
        cos_vec = _cosine_per_dim(
            trace.recovered_noise, reference,
            active_channel_ids, frame_chunk_size, action_per_frame,
        )
        vectors.append(cos_vec)

    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vectors)


def wmf_episode_score(
    record: EpisodeRecord,
    *,
    true_key: int,
    null_decoy_count: int,
    config_template,
    sample_rate_hz: float,
    active_channel_ids: list[int],
    frame_chunk_size: int,
    action_per_frame: int,
    subspace_rank: int | None = None,
) -> tuple[float, float]:
    """Compute WMF score and z-score for an episode.

    Returns: (wmf_score, z_score)
    """
    # True-key feature vector
    true_vector = episode_score_vector(
        record,
        candidate_key=true_key,
        config_template=config_template,
        sample_rate_hz=sample_rate_hz,
        active_channel_ids=active_channel_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
    )
    if true_vector.size == 0:
        return 0.0, 0.0

    # Null vectors from wrong keys
    null_vectors = []
    for i in range(null_decoy_count):
        wrong_key = true_key + 1 + i
        null_vec = episode_score_vector(
            record,
            candidate_key=wrong_key,
            config_template=config_template,
            sample_rate_hz=sample_rate_hz,
            active_channel_ids=active_channel_ids,
            frame_chunk_size=frame_chunk_size,
            action_per_frame=action_per_frame,
        )
        if null_vec.shape == true_vector.shape:
            null_vectors.append(null_vec)

    if not null_vectors:
        return 0.0, 0.0

    null_matrix = np.asarray(null_vectors, dtype=np.float64)
    true_vector_64 = np.asarray(true_vector, dtype=np.float64)

    # WMF score
    wmf = _wmf_score(true_vector_64, null_matrix, subspace_rank=subspace_rank)

    # z-score: score false keys with same null bank, compute (true - mean) / std
    false_scores = []
    for null_vec in null_vectors:
        false_scores.append(_wmf_score(
            np.asarray(null_vec, dtype=np.float64),
            null_matrix,
            subspace_rank=subspace_rank,
        ))
    false_scores_np = np.asarray(false_scores, dtype=np.float64)
    false_std = float(np.std(false_scores_np))
    if false_std < 1e-6:
        false_std = 1.0
    z_score = (wmf - float(np.mean(false_scores_np))) / false_std

    return float(wmf), float(z_score)


# ---------------------------------------------------------------------------
# WMF scoring (self-contained, same as scoring.py but operates on flat vectors)
# ---------------------------------------------------------------------------

def _wmf_score(
    feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None = None,
) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0:
        return 0.0

    mu = np.mean(null_matrix, axis=0)
    centered_feature = feature - mu
    centered_null = null_matrix - mu

    if centered_null.shape[0] < 2 or centered_null.shape[1] < 1:
        return 0.0

    cov = np.cov(centered_null, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    rank = subspace_rank if subspace_rank is not None else len(eigvals)
    rank = min(rank, len(eigvals), centered_null.shape[0] - 1)
    rank = max(rank, 1)
    eigvals = eigvals[:rank]
    eigvecs = eigvecs[:, :rank]

    projected = centered_feature @ eigvecs  # [rank]
    template = np.sum(eigvecs, axis=0)
    whitened = projected / np.sqrt(np.maximum(eigvals, 1e-8))
    template_whitened = template / np.sqrt(np.maximum(eigvals, 1e-8))
    return float(np.dot(template_whitened, whitened))


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------

def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    for pv in pos:
        wins += float(np.sum(pv > neg))
        wins += 0.5 * float(np.sum(pv == neg))
    return wins / float(pos.size * neg.size)


def calibrate_threshold(negative_scores: np.ndarray, target_fpr: float) -> float:
    neg = np.sort(np.asarray(negative_scores, dtype=np.float32))
    if neg.size == 0:
        return float("nan")
    keep = int(math.ceil((1.0 - target_fpr) * neg.size)) - 1
    keep = int(np.clip(keep, 0, neg.size - 1))
    return float(neg[keep])


def binary_metrics(positive: np.ndarray, negative: np.ndarray, threshold: float) -> tuple[float, float]:
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    tpr = float(np.mean(pos >= threshold)) if pos.size else float("nan")
    fpr = float(np.mean(neg >= threshold)) if neg.size else float("nan")
    return tpr, fpr


def pairwise_accuracy(marked: np.ndarray, plain: np.ndarray) -> float:
    count = min(marked.size, plain.size)
    if count == 0:
        return float("nan")
    return float(np.mean(marked[:count] > plain[:count]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm-dir", type=Path, required=True,
                        help="Directory with watermarked rollout NPZs")
    parser.add_argument("--plain-dir", type=Path, required=True,
                        help="Directory with plain rollout NPZs")
    parser.add_argument("--secret-key", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=float, default=16.0)
    parser.add_argument("--freq-min-hz", type=float, default=0.5)
    parser.add_argument("--freq-max-hz", type=float, default=3.0)
    parser.add_argument("--reference-mode", type=str, default="gaussian")
    parser.add_argument("--chunk-selection-strategy", type=str, default="stateful_online")
    parser.add_argument("--chunk-selection-period", type=int, default=6)
    parser.add_argument("--chunk-selection-count", type=int, default=5)
    parser.add_argument("--chunk-start-min", type=int, default=2)
    parser.add_argument("--null-decoy-count", type=int, default=32)
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--frame-chunk-size", type=int, default=4)
    parser.add_argument("--action-per-frame", type=int, default=4)
    parser.add_argument("--active-channels", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from wan_va.wm.watermark import InternalNoiseWatermarkConfig

    config_template = InternalNoiseWatermarkConfig(
        secret_key=args.secret_key,
        control_freq=args.sample_rate_hz,
        beta=args.beta,
        freq_range=(args.freq_min_hz, args.freq_max_hz),
        reference_mode=args.reference_mode,
        chunk_selection_strategy=args.chunk_selection_strategy,
        chunk_selection_period=args.chunk_selection_period,
        chunk_selection_count=args.chunk_selection_count,
        chunk_start_min=args.chunk_start_min,
    )

    active_channel_ids = list(args.active_channels)

    # Load pairs
    print(f"Loading pairs: wm={args.wm_dir}, plain={args.plain_dir}")
    pairs = collect_pairs(args.wm_dir, args.plain_dir)
    print(f"Found {len(pairs)} matched episode pairs")

    if not pairs:
        print("ERROR: No matched pairs found.")
        return

    # Score each episode
    positive_scores = []  # watermarked z-scores
    negative_scores = []  # plain z-scores
    positive_wmf = []
    negative_wmf = []
    wm_successes = []
    plain_successes = []

    score_kwargs = dict(
        true_key=args.secret_key,
        null_decoy_count=args.null_decoy_count,
        config_template=config_template,
        sample_rate_hz=args.sample_rate_hz,
        active_channel_ids=active_channel_ids,
        frame_chunk_size=args.frame_chunk_size,
        action_per_frame=args.action_per_frame,
        subspace_rank=args.subspace_rank,
    )

    for i, (plain_rec, wm_rec) in enumerate(pairs):
        # Score watermarked episode
        if wm_rec.chunk_traces:
            wm_wmf, wm_z = wmf_episode_score(wm_rec, **score_kwargs)
            positive_wmf.append(wm_wmf)
            positive_scores.append(wm_z)
        else:
            positive_wmf.append(0.0)
            positive_scores.append(0.0)

        # Score plain episode
        if plain_rec.chunk_traces:
            pl_wmf, pl_z = wmf_episode_score(plain_rec, **score_kwargs)
            negative_wmf.append(pl_wmf)
            negative_scores.append(pl_z)
        else:
            negative_wmf.append(0.0)
            negative_scores.append(0.0)

        wm_successes.append(float(wm_rec.success))
        plain_successes.append(float(plain_rec.success))

        print(f"  [{i+1}/{len(pairs)}] T{wm_rec.task_id}E{wm_rec.episode_idx}: "
              f"wm_z={positive_scores[-1]:.3f} plain_z={negative_scores[-1]:.3f} "
              f"wm_wmf={positive_wmf[-1]:.3f} plain_wmf={negative_wmf[-1]:.3f}")

    # Compute metrics
    pos_np = np.asarray(positive_scores, dtype=np.float32)
    neg_np = np.asarray(negative_scores, dtype=np.float32)
    pos_wmf_np = np.asarray(positive_wmf, dtype=np.float32)
    neg_wmf_np = np.asarray(negative_wmf, dtype=np.float32)

    auc_zscore = roc_auc(pos_np, neg_np)
    auc_wmf = roc_auc(pos_wmf_np, neg_wmf_np)

    threshold_z = calibrate_threshold(neg_np, args.target_fpr)
    tpr_z, fpr_z = binary_metrics(pos_np, neg_np, threshold_z)

    threshold_wmf = calibrate_threshold(neg_wmf_np, args.target_fpr)
    tpr_wmf, fpr_wmf = binary_metrics(pos_wmf_np, neg_wmf_np, threshold_wmf)

    pw_acc = pairwise_accuracy(pos_np, neg_np)

    # Report
    print(f"\n{'='*60}")
    print("WATERMARK DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {len(pairs)} pairs")
    print(f"WM success rate: {np.mean(wm_successes):.2%}")
    print(f"Plain success rate: {np.mean(plain_successes):.2%}")
    print(f"\n--- z-score metrics ---")
    print(f"ROC AUC (z-score): {auc_zscore:.4f}")
    print(f"Threshold (target FPR={args.target_fpr:.2%}): {threshold_z:.4f}")
    print(f"TPR@{args.target_fpr:.0%}FPR: {tpr_z:.4f}")
    print(f"FPR: {fpr_z:.4f}")
    print(f"Pairwise accuracy: {pw_acc:.4f}")
    print(f"WM z-scores: mean={np.mean(pos_np):.4f} std={np.std(pos_np):.4f}")
    print(f"Plain z-scores: mean={np.mean(neg_np):.4f} std={np.std(neg_np):.4f}")
    print(f"\n--- raw WMF metrics ---")
    print(f"ROC AUC (raw WMF): {auc_wmf:.4f}")
    print(f"TPR@{args.target_fpr:.0%}FPR (WMF): {tpr_wmf:.4f}")
    print(f"WM WMF: mean={np.mean(pos_wmf_np):.4f} std={np.std(pos_wmf_np):.4f}")
    print(f"Plain WMF: mean={np.mean(neg_wmf_np):.4f} std={np.std(neg_wmf_np):.4f}")
    print(f"{'='*60}")

    # Save JSON report
    report = {
        "num_pairs": len(pairs),
        "secret_key": args.secret_key,
        "beta": args.beta,
        "null_decoy_count": args.null_decoy_count,
        "subspace_rank": args.subspace_rank,
        "target_fpr": args.target_fpr,
        "auc_zscore": float(auc_zscore),
        "auc_wmf": float(auc_wmf),
        "threshold_z": float(threshold_z),
        "tpr_at_target_fpr_z": float(tpr_z),
        "fpr_z": float(fpr_z),
        "threshold_wmf": float(threshold_wmf),
        "tpr_at_target_fpr_wmf": float(tpr_wmf),
        "pairwise_accuracy": float(pw_acc),
        "wm_zscore_mean": float(np.mean(pos_np)),
        "wm_zscore_std": float(np.std(pos_np)),
        "plain_zscore_mean": float(np.mean(neg_np)),
        "plain_zscore_std": float(np.std(neg_np)),
        "wm_wmf_mean": float(np.mean(pos_wmf_np)),
        "plain_wmf_mean": float(np.mean(neg_wmf_np)),
        "wm_success_rate": float(np.mean(wm_successes)),
        "plain_success_rate": float(np.mean(plain_successes)),
        "per_episode": [
            {
                "task_id": int(pairs[i][1].task_id),
                "episode_idx": int(pairs[i][1].episode_idx),
                "wm_zscore": float(positive_scores[i]),
                "plain_zscore": float(negative_scores[i]),
                "wm_wmf": float(positive_wmf[i]),
                "plain_wmf": float(negative_wmf[i]),
                "wm_success": bool(pairs[i][1].success),
                "plain_success": bool(pairs[i][0].success),
                "wm_num_map_chunks": len(pairs[i][1].chunk_traces),
                "plain_num_map_chunks": len(pairs[i][0].chunk_traces),
            }
            for i in range(len(pairs))
        ],
    }

    out_path = args.out_json or (args.wm_dir.parent / "detection_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
