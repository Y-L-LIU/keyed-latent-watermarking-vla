"""Export per-episode RAW verifier scores in the paper-delivery schema.

Unlike build_*_table.py (which collapse everything to z-scores / AUCs), this
dumps one CSV row per episode with the raw whitened-matched-filter score for
the true key and for J false keys, so the statistician can do their own
z-score calibration / H0 construction downstream.

Schema (one row per episode):
    episode_id, model, dataset, obs, obs_ratio, recovery, attack, m,
    s_true, s_false_1 ... s_false_J

Reuses the exact scoring helpers from build_cost_utility_table.py so the raw
scores are identical to what fed the existing AUC tables.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_cost_utility_table import _score_episode, _null_episode, _wmf_score


def per_episode_raw_scores(data, *, null_count, max_windows, score_step_scope,
                           subspace_rank):
    """Return (s_true, [s_false_1..J], m).

    s_true   = WMF(true_feature, null_matrix)
    s_false_i = WMF(null_row_i, null_matrix \\ row_i)   (leave-one-out)
    m        = number of keyed chunks actually scored for this episode
    """
    secret_key = int(data["secret_key"]) if "secret_key" in data.files else 12345
    sample_rate_hz = float(data["sample_rate_hz"]) if "sample_rate_hz" in data.files else 50.0
    true_vec = _score_episode(data, score_step_scope=score_step_scope, max_windows=max_windows)
    m = int(true_vec.shape[0])
    null_rows = np.stack([
        _null_episode(data, off + 1, score_step_scope=score_step_scope,
                      max_windows=max_windows, secret_key=secret_key,
                      sample_rate_hz=sample_rate_hz)
        for off in range(null_count)
    ])
    s_true = _wmf_score(true_vec, null_rows, subspace_rank=subspace_rank)
    s_false = [
        _wmf_score(null_rows[i], np.delete(null_rows, i, axis=0), subspace_rank=subspace_rank)
        for i in range(null_rows.shape[0])
    ]
    return float(s_true), [float(x) for x in s_false], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True,
                    help="dir containing episode *.npz (searched recursively)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--model", required=True, help="pi0 / pi0.5 / lingbot")
    ap.add_argument("--dataset", required=True, help="benchmark name, e.g. libero_10")
    ap.add_argument("--attack", required=True,
                    help="clean / clip / ema / jitter / delay / finetune / ...")
    ap.add_argument("--attack-strength", default="",
                    help="numeric attack strength, e.g. 0.5; empty/0 for clean")
    ap.add_argument("--obs", required=True, choices=["full", "partial"])
    ap.add_argument("--obs-ratio", type=float, required=True, help="D_env/D_raw")
    ap.add_argument("--recovery", required=True, choices=["ode", "map"])
    ap.add_argument("--null-count", type=int, default=32, help="J false keys")
    ap.add_argument("--max-windows", type=int, default=5)
    ap.add_argument("--score-step-scope", default="full_chunk")
    ap.add_argument("--subspace-rank", type=int, default=3)
    ap.add_argument("--variant-filter", default=None,
                    choices=[None, "plain", "watermarked"],
                    help="if set, only export this variant")
    args = ap.parse_args()

    rollout_dir = pathlib.Path(args.rollout_dir)
    npzs = sorted(rollout_dir.rglob("*.npz"))
    if not npzs:
        print(f"[export] no npz under {rollout_dir}", file=sys.stderr)
        sys.exit(1)

    J = args.null_count
    header = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
               "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for npz in npzs:
            data = np.load(npz, allow_pickle=True)
            variant = str(data["variant"]) if "variant" in data.files else "watermarked"
            if args.variant_filter and variant != args.variant_filter:
                continue
            # episode_id: stable, encodes variant so plain/wm don't collide.
            episode_id = f"{args.model}|{args.dataset}|{args.attack}|{args.obs}|{args.recovery}|{npz.stem}"
            s_true, s_false, m = per_episode_raw_scores(
                data, null_count=J, max_windows=args.max_windows,
                score_step_scope=args.score_step_scope, subspace_rank=args.subspace_rank)
            w.writerow([episode_id, variant, args.model, args.dataset, args.obs,
                        f"{args.obs_ratio:.4f}", args.recovery, args.attack,
                        args.attack_strength, m,
                        f"{s_true:.8f}"] + [f"{x:.8f}" for x in s_false])
            n_written += 1
            if n_written % 20 == 0:
                print(f"[export] {n_written} episodes ...", file=sys.stderr, flush=True)

    print(f"[export] wrote {n_written} episode rows (J={J}) -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
