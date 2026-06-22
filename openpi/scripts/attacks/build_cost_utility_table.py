"""Build the §12.5 cost-utility table from Stage 3 eval results.

After the JAX fine-tune sweep finishes and each attacked checkpoint is
evaluated with `eval_libero_action_inversion_postprocess_robustness.py`,
this script:

  1. Walks `/workspace/vla/attack_c_data/eval/lam<value>/libero_10/task_rollout/`
     for each `lam`.
  2. Pulls per-episode WMF z-scores via the same logic as `rescore_zscore.py`
     (multi-trajectory cross-task aggregation at N ∈ {1, 5, 10, 15}).
  3. Counts watermarked-policy task successes (`success` field in each NPZ).
  4. Emits `cost_utility_table.json` with one row per λ::

       {"lam": 1.0,
        "task_success_wm": 0.72,
        "task_success_plain": 0.78,
        "single_auc": 0.62,
        "cross_at_5_auc": 0.81,
        "cross_at_15_auc": 0.94,
        "wm_minus_plain_mean": 12.3}

Usage::

    python scripts/attacks/build_cost_utility_table.py \
        --eval-root /workspace/vla/attack_c_data/eval \
        --lams 0 0.1 1 10 100 \
        --out /workspace/vla/attack_c_data/cost_utility_table.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys

import numpy as np

# Reuse the WMF scoring helpers from rescore_zscore.py without re-importing the
# whole CLI (which would clobber argparse). Inline copies are tiny.

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
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
    return float(
        np.dot(template / np.sqrt(np.maximum(eigvals, 1e-8)), projected / np.sqrt(np.maximum(eigvals, 1e-8)))
    )


def _score_episode(data, *, score_step_scope: str, max_windows: int) -> np.ndarray:
    selected = data["chunk_selected"]
    recovered = data["chunk_recovered_noise"]
    reference = data["chunk_reference"]
    executed = data["chunk_executed_steps"]
    out: list[float] = []
    count = 0
    for i in range(len(selected)):
        if not selected[i] or executed[i] <= 0:
            continue
        if count >= max_windows:
            break
        steps = reference[i].shape[0] if score_step_scope == "full_chunk" else int(executed[i])
        out.append(_cosine_sim(recovered[i][:steps], reference[i][:steps]))
        count += 1
    return np.asarray(out, dtype=np.float32)


def _null_episode(data, secret_offset: int, *, score_step_scope: str, max_windows: int,
                  secret_key: int, sample_rate_hz: float) -> np.ndarray:
    # Lazy import — keep the script standalone-ish but reuse the watermark module.
    from openpi.policies import watermark as wm

    selected = data["chunk_selected"]
    recovered = data["chunk_recovered_noise"]
    reference = data["chunk_reference"]
    executed = data["chunk_executed_steps"]
    chunk_indices = data["chunk_index"] if "chunk_index" in data.files else data["chunk_chunk_index"]
    episode_nonce = int(data["episode_nonce"])
    action_dim = int(reference.shape[-1])
    horizon = int(reference.shape[1])

    cfg = wm.InternalNoiseWatermarkConfig(secret_key=secret_key + secret_offset, control_freq=sample_rate_hz)
    out: list[float] = []
    count = 0
    for i in range(len(selected)):
        if not selected[i] or executed[i] <= 0:
            continue
        if count >= max_windows:
            break
        ctx = wm.WatermarkContext(chunk_index=int(chunk_indices[i]), episode_nonce=episode_nonce)
        ref = wm.generate_keyed_reference(
            length=horizon,
            action_dim=action_dim,
            sample_rate_hz=sample_rate_hz,
            config=cfg,
            context=ctx,
        )
        steps = ref.shape[0] if score_step_scope == "full_chunk" else int(executed[i])
        out.append(_cosine_sim(recovered[i][:steps], ref[:steps]))
        count += 1
    return np.asarray(out, dtype=np.float32)


def _per_episode_z(data, *, null_count: int, max_windows: int, score_step_scope: str,
                   subspace_rank: int) -> tuple[float, float]:
    """Returns (wmf_score, z_score)."""
    secret_key = int(data["secret_key"]) if "secret_key" in data.files else 12345
    sample_rate_hz = float(data["sample_rate_hz"]) if "sample_rate_hz" in data.files else 50.0
    true_vec = _score_episode(data, score_step_scope=score_step_scope, max_windows=max_windows)
    null_rows = np.stack([
        _null_episode(data, off + 1, score_step_scope=score_step_scope, max_windows=max_windows,
                      secret_key=secret_key, sample_rate_hz=sample_rate_hz)
        for off in range(null_count)
    ])
    wmf = _wmf_score(true_vec, null_rows, subspace_rank=subspace_rank)
    null_wmfs = np.array([
        _wmf_score(null_rows[i], np.delete(null_rows, i, axis=0), subspace_rank=subspace_rank)
        for i in range(null_rows.shape[0])
    ], dtype=np.float64)
    null_std = float(np.std(null_wmfs))
    if null_std < 1e-6:
        null_std = 1.0
    z = (wmf - float(np.mean(null_wmfs))) / null_std
    return float(wmf), float(z)


def _process_lambda(eval_root: pathlib.Path, lam: str, *,
                    null_count: int = 32, max_windows: int = 5,
                    score_step_scope: str = "full_chunk",
                    subspace_rank: int = 3,
                    group_sizes: tuple[int, ...] = (1, 5, 10, 15)) -> dict:
    rollout_dir = eval_root / f"lam{lam}" / "libero_10" / f"attack_c_lam{lam}" / "task_rollout"
    if not rollout_dir.exists():
        # Allow flexibility in the run-tag naming.
        candidates = list((eval_root / f"lam{lam}" / "libero_10").glob("*/task_rollout"))
        if candidates:
            rollout_dir = candidates[0]
        else:
            return {"lam": lam, "error": f"no task_rollout dir under {eval_root / f'lam{lam}'}"}

    plain_z, wm_z, plain_succ, wm_succ = [], [], [], []
    for npz in sorted(rollout_dir.glob("*.npz")):
        data = np.load(npz, allow_pickle=True)
        variant = str(data["variant"])
        success = bool(data["success"]) if "success" in data.files else False
        wmf, z = _per_episode_z(data, null_count=null_count, max_windows=max_windows,
                                score_step_scope=score_step_scope, subspace_rank=subspace_rank)
        if variant == "plain":
            plain_z.append(z)
            plain_succ.append(success)
        else:
            wm_z.append(z)
            wm_succ.append(success)

    p = np.asarray(plain_z)
    w = np.asarray(wm_z)
    single_auc = float(sum(1 for pi in p for wi in w if wi > pi) / max(len(p) * len(w), 1))

    rng = np.random.default_rng(42)
    group_aucs: dict[str, float] = {}
    for gs in group_sizes:
        if gs > len(p) or gs > len(w):
            group_aucs[f"cross_at_{gs}_auc"] = float("nan")
            continue
        n_samples = 1000
        pg, wg = [], []
        for _ in range(n_samples):
            pg.append(p[rng.choice(len(p), size=gs, replace=False)].sum())
            wg.append(w[rng.choice(len(w), size=gs, replace=False)].sum())
        pg = np.asarray(pg)
        wg = np.asarray(wg)
        group_aucs[f"cross_at_{gs}_auc"] = float(sum(1 for pi in pg for wi in wg if wi > pi) / (len(pg) * len(wg)))

    return {
        "lam": lam,
        "n_plain_episodes": int(len(p)),
        "n_wm_episodes": int(len(w)),
        "task_success_plain": float(np.mean(plain_succ)) if plain_succ else float("nan"),
        "task_success_wm": float(np.mean(wm_succ)) if wm_succ else float("nan"),
        "wm_z_mean": float(np.mean(w)) if len(w) else float("nan"),
        "plain_z_mean": float(np.mean(p)) if len(p) else float("nan"),
        "wm_minus_plain_mean": float(np.mean(w) - np.mean(p)) if len(w) and len(p) else float("nan"),
        "single_auc": single_auc,
        **group_aucs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="/workspace/vla/attack_c_data/eval")
    parser.add_argument("--lams", nargs="+", default=["0", "0.1", "1", "10", "100"])
    parser.add_argument("--out", default="/workspace/vla/attack_c_data/cost_utility_table.json")
    parser.add_argument("--null-count", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--score-step-scope", default="full_chunk")
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 5, 10, 15])
    args = parser.parse_args()

    eval_root = pathlib.Path(args.eval_root)
    rows = []
    for lam in args.lams:
        print(f"[cost-utility] processing lam={lam} ...", file=sys.stderr, flush=True)
        row = _process_lambda(
            eval_root, lam,
            null_count=args.null_count,
            max_windows=args.max_windows,
            score_step_scope=args.score_step_scope,
            subspace_rank=args.subspace_rank,
            group_sizes=tuple(args.group_sizes),
        )
        print(json.dumps(row), file=sys.stderr, flush=True)
        rows.append(row)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2))

    # Markdown table for quick eyeballing.
    print("| λ | success_wm | success_plain | single AUC | cross@5 | cross@10 | cross@15 | Δ z |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            print(f"| {r['lam']} | ERROR: {r['error']} | | | | | | |")
            continue
        print(
            f"| {r['lam']} | {r['task_success_wm']:.2f} | {r['task_success_plain']:.2f}"
            f" | {r['single_auc']:.3f} | {r.get('cross_at_5_auc', float('nan')):.3f}"
            f" | {r.get('cross_at_10_auc', float('nan')):.3f} | {r.get('cross_at_15_auc', float('nan')):.3f}"
            f" | {r['wm_minus_plain_mean']:.2f} |"
        )


if __name__ == "__main__":
    main()
