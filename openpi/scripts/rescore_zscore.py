"""Compute per-episode z-scores and multi-episode aggregated AUC from saved .npz files."""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openpi.policies import watermark as wm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _wmf_score(feature: np.ndarray, null_matrix: np.ndarray, subspace_rank: int | None = None) -> float:
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


def _score_vector_from_npz(data, *, score_step_scope: str, max_windows: int = 5) -> np.ndarray:
    selected = data["chunk_selected"]
    recovered = data["chunk_recovered_noise"]
    reference = data["chunk_reference"]
    executed_steps = data["chunk_executed_steps"]
    scores = []
    count = 0
    for i in range(len(selected)):
        if not selected[i] or executed_steps[i] <= 0:
            continue
        if count >= max_windows:
            break
        steps = reference[i].shape[0] if score_step_scope == "full_chunk" else int(executed_steps[i])
        scores.append(cosine_sim(recovered[i][:steps], reference[i][:steps]))
        count += 1
    return np.asarray(scores, dtype=np.float32)


def _null_vectors_from_npz(data, *, null_count: int, score_step_scope: str, max_windows: int = 5,
                           secret_key: int = 12345, sample_rate_hz: float = 50.0) -> np.ndarray:
    selected = data["chunk_selected"]
    recovered = data["chunk_recovered_noise"]
    reference = data["chunk_reference"]
    executed_steps = data["chunk_executed_steps"]
    chunk_indices = data["chunk_index"]
    episode_nonce = int(data["episode_nonce"])
    action_dim = int(reference.shape[-1])
    horizon = int(reference.shape[1])

    null_vectors = []
    for offset in range(1, null_count + 1):
        cfg = wm.InternalNoiseWatermarkConfig(secret_key=secret_key + offset, control_freq=sample_rate_hz)
        scores = []
        count = 0
        for i in range(len(selected)):
            if not selected[i] or executed_steps[i] <= 0:
                continue
            if count >= max_windows:
                break
            context = wm.WatermarkContext(chunk_index=int(chunk_indices[i]), episode_nonce=episode_nonce)
            ref = wm.generate_keyed_reference(
                length=horizon,
                action_dim=action_dim,
                sample_rate_hz=sample_rate_hz,
                config=cfg,
                context=context,
            )
            steps = ref.shape[0] if score_step_scope == "full_chunk" else int(executed_steps[i])
            scores.append(cosine_sim(recovered[i][:steps], ref[:steps]))
            count += 1
        null_vectors.append(scores)
    return np.asarray(null_vectors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_dirs", nargs="+", help="Directories containing episode_*.npz files (one per seed)")
    parser.add_argument("--null-count", type=int, default=32)
    parser.add_argument("--score-step-scope", choices=("executed", "full_chunk"), default="executed")
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--group-sizes", type=int, nargs="*", default=[1, 3, 5, 10])
    args = parser.parse_args()

    all_rows = []
    for npz_dir in args.npz_dirs:
        files = sorted(glob.glob(os.path.join(npz_dir, "episode_*.npz")))
        for path in files:
            data = np.load(path, allow_pickle=True)
            variant = str(data["variant"])
            ep = int(data["episode_idx"])
            seed_val = int(data["seed"])

            true_vec = _score_vector_from_npz(data, score_step_scope=args.score_step_scope,
                                              max_windows=args.max_windows)
            null_mat = _null_vectors_from_npz(data, null_count=args.null_count,
                                             score_step_scope=args.score_step_scope,
                                             max_windows=args.max_windows)

            wmf = _wmf_score(true_vec, null_mat, subspace_rank=args.subspace_rank)
            null_wmf_scores = []
            for ni in range(null_mat.shape[0]):
                null_wmf_scores.append(_wmf_score(null_mat[ni], np.delete(null_mat, ni, axis=0),
                                                  subspace_rank=args.subspace_rank))
            null_wmf = np.array(null_wmf_scores, dtype=np.float64)
            null_std = float(np.std(null_wmf))
            if null_std < 1e-6:
                null_std = 1.0
            z = float((wmf - np.mean(null_wmf)) / null_std)

            all_rows.append({
                "dir": npz_dir, "ep": ep, "seed": seed_val, "variant": variant,
                "wmf": wmf, "z": z, "null_mean": float(np.mean(null_wmf)), "null_std": null_std,
            })

    # Print per-episode
    plain_z = [r["z"] for r in all_rows if r["variant"] == "plain"]
    wm_z = [r["z"] for r in all_rows if r["variant"] == "watermarked"]
    plain_wmf = [r["wmf"] for r in all_rows if r["variant"] == "plain"]
    wm_wmf = [r["wmf"] for r in all_rows if r["variant"] == "watermarked"]

    print(f"=== Per-Episode Results ({len(plain_z)} plain, {len(wm_z)} watermarked) ===")
    print(f"WMF   - plain: mean={np.mean(plain_wmf):.1f} std={np.std(plain_wmf):.1f} | wm: mean={np.mean(wm_wmf):.1f} std={np.std(wm_wmf):.1f}")
    print(f"Z-score - plain: mean={np.mean(plain_z):.2f} std={np.std(plain_z):.2f} | wm: mean={np.mean(wm_z):.2f} std={np.std(wm_z):.2f}")

    p, w = np.array(plain_z), np.array(wm_z)
    auc = sum(1 for pi in p for wi in w if wi > pi) / (len(p) * len(w))
    print(f"Single-episode AUC (z-score): {auc:.4f}")
    print(f"  min(wm_z)={w.min():.2f}  max(plain_z)={p.max():.2f}")
    print()

    # Multi-episode aggregation (sum of z-scores)
    print("=== Multi-Episode Aggregation (sum of z-scores) ===")
    rng = np.random.default_rng(42)
    for gs in args.group_sizes:
        if gs > len(plain_z) or gs > len(wm_z):
            continue
        n_samples = 1000
        plain_group_scores = []
        wm_group_scores = []
        for _ in range(n_samples):
            idx_p = rng.choice(len(plain_z), size=gs, replace=False)
            idx_w = rng.choice(len(wm_z), size=gs, replace=False)
            plain_group_scores.append(sum(plain_z[i] for i in idx_p))
            wm_group_scores.append(sum(wm_z[i] for i in idx_w))
        pg = np.array(plain_group_scores)
        wg = np.array(wm_group_scores)
        group_auc = sum(1 for pi in pg for wi in wg if wi > pi) / (len(pg) * len(wg))
        print(f"  group_size={gs:2d}: AUC={group_auc:.4f}  plain_sum_mean={pg.mean():.2f}  wm_sum_mean={wg.mean():.2f}")


if __name__ == "__main__":
    main()
