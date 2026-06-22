"""Compute recovery error + watermark-score AUC for the 8-cell recovery table
(family × obs ∈ {full, partial} × recovery ∈ {ODE, MAP}).

Reads existing rollout npz files; expects these fields per chunk:
  chunk_injected_noise              (n_chunks, H, D)   ground-truth z₀
  chunk_recovered_noise             (n_chunks, H, D)   reverse-Euler ẑ from raw action (full+ODE)
  chunk_map_restart_recovered_noise (n_chunks, R, H, D) MAP restarts (typically partial+MAP)
  chunk_map_best_restart_index      (n_chunks,)        which restart was best
  chunk_raw_actions                 (n_chunks, H, D_full)  full sampled action
  chunk_observed_actions            (n_chunks, H, D_env)   env-visible projection
  chunk_watermarked_flags (optional, only chunks the WM was injected on count)

Outputs (per family):
  - per-chunk recovery error ‖ẑ - z₀‖ / ‖z₀‖   (for full+ODE, partial+MAP using existing npz)
  - per-episode AUC and TPR@1%, TPR@10% (H1 = watermarked s_true, H0 = plain s_true)

Two cells (full+MAP, partial+ODE) are NOT in the npz; they require running the policy.
See finish_recovery_cells.py for that.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def load_episode(npz_path: Path) -> dict | None:
    d = np.load(npz_path, allow_pickle=True)
    if "chunk_injected_noise" not in d.files:
        return None
    sel = np.asarray(d.get("chunk_selected", np.ones(d["chunk_injected_noise"].shape[0], bool)), bool)
    return {
        "variant": str(d["variant"]) if "variant" in d.files else (
            "watermarked" if float(d.get("beta", 0)) > 0 else "plain"),
        "z_true":      np.asarray(d["chunk_injected_noise"])[sel],            # (n_sel, H, D)
        "z_ode_full":  np.asarray(d.get("chunk_recovered_noise"))[sel] if "chunk_recovered_noise" in d.files else None,
        "z_map_part":  _map_best(d, sel),
        "task_id":     int(d.get("task_id", -1)),
        "episode_idx": int(d.get("episode_idx", -1)),
        "secret_key":  int(d.get("secret_key", -1)),
        "wmf_scores":  np.asarray(d.get("wmf_scores", np.array([]))),
    }


def _map_best(d, sel):
    if "chunk_map_restart_recovered_noise" not in d.files:
        return None
    restarts = np.asarray(d["chunk_map_restart_recovered_noise"])[sel]  # (n_sel, R, H, D)
    if "chunk_map_best_restart_index" in d.files:
        best = np.asarray(d["chunk_map_best_restart_index"])[sel]
        n = len(best)
        return np.stack([restarts[i, best[i]] for i in range(n)])
    return restarts[:, 0]


def recovery_error(z_true: np.ndarray, z_recovered: np.ndarray) -> dict:
    """Per-chunk relative L2 + cosine."""
    if z_recovered is None or z_true is None:
        return {"n": 0}
    z_true_f = z_true.reshape(z_true.shape[0], -1).astype(np.float64)
    z_rec_f = z_recovered.reshape(z_recovered.shape[0], -1).astype(np.float64)
    err = np.linalg.norm(z_true_f - z_rec_f, axis=1) / (np.linalg.norm(z_true_f, axis=1) + 1e-12)
    cos = (z_true_f * z_rec_f).sum(axis=1) / (np.linalg.norm(z_true_f, axis=1) *
                                              np.linalg.norm(z_rec_f, axis=1) + 1e-12)
    return {"n": int(len(err)),
            "rel_l2_mean": float(err.mean()), "rel_l2_med": float(np.median(err)),
            "cos_mean": float(cos.mean()), "cos_med": float(np.median(cos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True, type=Path,
                    help="Directory containing *_watermarked.npz and *_plain.npz")
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    npzs = sorted(args.rollout_dir.rglob("*.npz"))
    print(f"# scanning {len(npzs)} npz in {args.rollout_dir}")

    wm_eps, plain_eps = [], []
    for p in npzs:
        ep = load_episode(p)
        if ep is None: continue
        (wm_eps if ep["variant"] == "watermarked" else plain_eps).append(ep)
    print(f"# loaded {len(wm_eps)} watermarked + {len(plain_eps)} plain episodes")

    # recovery error: only for watermarked episodes (where z_true is the keyed pattern)
    out = {"n_wm": len(wm_eps), "n_plain": len(plain_eps), "cells": {}}
    for cell_name, key in [("full+ODE", "z_ode_full"), ("partial+MAP", "z_map_part")]:
        errs = []
        for ep in wm_eps:
            if ep[key] is None or ep["z_true"] is None: continue
            r = recovery_error(ep["z_true"], ep[key])
            if r.get("n", 0) > 0:
                errs.append(r)
        if not errs:
            out["cells"][cell_name] = {"available": False, "reason": "field not present in npz"}
            continue
        out["cells"][cell_name] = {
            "available": True,
            "n_chunks_total": int(sum(r["n"] for r in errs)),
            "rel_l2_mean": float(np.mean([r["rel_l2_mean"] for r in errs])),
            "rel_l2_med": float(np.mean([r["rel_l2_med"] for r in errs])),
            "cos_mean": float(np.mean([r["cos_mean"] for r in errs])),
            "cos_med": float(np.mean([r["cos_med"] for r in errs])),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
