"""Stage 1 of Attack C (Section 12.5).

Estimate the watermark-occupied subspace of the sampler latent z from saved
MAP-recovered noise tensors, without knowing the owner's secret key.

Inputs are the per-episode .npz files produced by
`scripts/eval_libero_action_inversion.py` (key `chunk_recovered_noise` of
shape `[N_chunks, T, D]`). The attacker provides one or two rollout
collections:

  * `--wm-dirs` (required): rollouts from the watermarked policy (the only
    artifact the attacker actually needs to obtain).
  * `--plain-dirs` (optional): rollouts from the same policy with beta=0.
    When provided, the script estimates the *contrast* subspace
    `Cov(z_wm) - Cov(z_plain)`, which isolates the keyed direction more
    cleanly than naive PCA on watermarked noise.

The estimated subspace is a `[D, k]` orthonormal basis P_K in raw-latent
space (D = action-expert raw dim, typically 32 for pi0/pi05). Per-chunk
references live in `[T, D]` and the empirically dominant variance lies in
the per-timestep D-axis, so we treat each timestep `[D]` slice as one
sample. This also keeps P_K small enough to use as a hard projection
matrix during fine-tune.

Usage::

    python scripts/attacks/estimate_wm_subspace.py \
        --wm-dirs /path/to/wm/episode_dir1 /path/to/wm/episode_dir2 \
        --plain-dirs /path/to/plain/episode_dir1 \
        --rank 8 \
        --selected-only \
        --out /path/to/wm_subspace.npz

The output .npz contains:

  * `mean`         : `[D]` mean used for centering
  * `components`   : `[k, D]` orthonormal basis (rows are subspace dirs)
  * `singular_values` : `[k]`
  * `eigenvalues`  : `[D]` full spectrum (for diagnostics)
  * `mode`         : `pca` or `pca_diff`
  * `n_samples`    : number of `[D]` vectors aggregated
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib

import numpy as np


def _gather_recovered_noise(
    npz_dirs: list[str],
    *,
    selected_only: bool,
    max_chunks_per_episode: int | None = None,
) -> np.ndarray:
    """Stack `chunk_recovered_noise` slices into a single `[N, D]` matrix."""

    rows: list[np.ndarray] = []
    files: list[str] = []
    for d in npz_dirs:
        # Older saved-rollout layouts use `episode_*.npz`; the LIBERO
        # action-inversion script saves as `task_NNN_episode_NNN_<variant>.npz`.
        # We accept both. Filter to watermarked-only when both are present.
        candidates = sorted(glob.glob(os.path.join(d, "*.npz")))
        for path in candidates:
            name = os.path.basename(path)
            if not (name.startswith("episode_") or "_episode_" in name):
                continue
            if "_plain.npz" in name:
                # Plain rollouts have base-noise recovered z; we only want
                # watermarked recovered z for subspace estimation.
                continue
            files.append(path)

    if not files:
        raise FileNotFoundError(f"No episode_*.npz under {npz_dirs!r}")

    for path in files:
        data = np.load(path, allow_pickle=True)
        recovered = np.asarray(data["chunk_recovered_noise"], dtype=np.float32)
        if recovered.ndim != 3:
            raise ValueError(f"{path}: chunk_recovered_noise must be [N,T,D], got {recovered.shape}")
        n_chunks = recovered.shape[0]

        if selected_only and "chunk_selected" in data.files:
            selected = np.asarray(data["chunk_selected"], dtype=bool)
        else:
            selected = np.ones(n_chunks, dtype=bool)

        if "chunk_executed_steps" in data.files:
            executed = np.asarray(data["chunk_executed_steps"], dtype=np.int32)
        else:
            executed = np.full(n_chunks, recovered.shape[1], dtype=np.int32)

        kept = 0
        for i in range(n_chunks):
            if not selected[i] or executed[i] <= 0:
                continue
            if max_chunks_per_episode is not None and kept >= max_chunks_per_episode:
                break
            steps = int(executed[i])
            rows.append(recovered[i, :steps].reshape(-1, recovered.shape[-1]))
            kept += 1

    if not rows:
        raise ValueError("No usable chunks found (all unselected or zero-step?).")

    return np.concatenate(rows, axis=0)  # [N, D]


def _empirical_cov(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, covariance) of `samples` shape `[N, D]`."""
    mean = samples.mean(axis=0)
    centered = samples - mean
    cov = (centered.T @ centered) / max(1, samples.shape[0] - 1)
    return mean, cov


def main():
    parser = argparse.ArgumentParser(description="Estimate watermark subspace from MAP-recovered noise.")
    parser.add_argument("--wm-dirs", nargs="+", required=True, help="NPZ directories from watermarked rollouts.")
    parser.add_argument("--plain-dirs", nargs="*", default=None, help="Optional NPZ directories from beta=0 rollouts.")
    parser.add_argument("--mode", choices=("pca", "pca_diff"), default=None,
                        help="Default: pca_diff if plain-dirs given, else pca.")
    parser.add_argument("--rank", type=int, default=8, help="How many subspace components to retain.")
    parser.add_argument("--selected-only", action="store_true",
                        help="Only use chunks marked selected (i.e. presumed keyed) by the verifier pipeline.")
    parser.add_argument("--max-chunks-per-episode", type=int, default=None,
                        help="Cap the number of chunks taken per episode (matches verifier max_windows).")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    args = parser.parse_args()

    mode = args.mode or ("pca_diff" if args.plain_dirs else "pca")

    print(f"[stage1] Gathering watermarked rollouts from {args.wm_dirs}")
    wm_samples = _gather_recovered_noise(
        args.wm_dirs,
        selected_only=args.selected_only,
        max_chunks_per_episode=args.max_chunks_per_episode,
    )
    print(f"[stage1] Watermarked samples: {wm_samples.shape}")

    if mode == "pca_diff":
        if not args.plain_dirs:
            raise ValueError("pca_diff mode requires --plain-dirs.")
        print(f"[stage1] Gathering plain rollouts from {args.plain_dirs}")
        plain_samples = _gather_recovered_noise(
            args.plain_dirs,
            selected_only=False,
            max_chunks_per_episode=args.max_chunks_per_episode,
        )
        print(f"[stage1] Plain samples: {plain_samples.shape}")

        mean_wm, cov_wm = _empirical_cov(wm_samples)
        _, cov_plain = _empirical_cov(plain_samples)
        # Watermark adds structured variance on top of base-noise variance.
        # cov_wm - cov_plain isolates that excess; its top eigenvectors
        # are the directions where the keyed reference concentrates energy.
        target = cov_wm - cov_plain
        target = 0.5 * (target + target.T)  # numerical symmetrization
        center = mean_wm
    else:
        center, cov_wm = _empirical_cov(wm_samples)
        target = cov_wm

    eigvals, eigvecs = np.linalg.eigh(target)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    rank = int(min(args.rank, eigvecs.shape[1]))
    components = eigvecs[:, :rank].T.astype(np.float32)  # [k, D]
    singular_values = np.sqrt(np.maximum(eigvals[:rank], 0.0)).astype(np.float32)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mean=center.astype(np.float32),
        components=components,
        singular_values=singular_values,
        eigenvalues=eigvals.astype(np.float32),
        mode=np.array(mode),
        n_samples=np.int64(wm_samples.shape[0]),
        selected_only=bool(args.selected_only),
    )

    print(f"[stage1] Mode={mode}  rank={rank}  D={components.shape[1]}")
    print(f"[stage1] Top {rank} eigenvalues: {eigvals[:rank].tolist()}")
    if mode == "pca_diff" and eigvals.size > rank:
        gap = float(eigvals[rank - 1] - eigvals[rank])
        print(f"[stage1] Spectral gap at k={rank}: {gap:.4g}")
    summary = {
        "mode": mode,
        "rank": rank,
        "n_samples": int(wm_samples.shape[0]),
        "top_eigenvalues": [float(v) for v in eigvals[:rank]],
    }
    print(f"[stage1] Wrote {out_path}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
