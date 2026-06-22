#!/usr/bin/env python3
"""Recompute pi0.5/LIBERO-10 LoRA descendant per-episode score CSVs.

The old descendant table kept this cell as a committed ceiling row because the
rescored CSV was not retained. The rollout NPZs are still present under
eval_out/libero10_from_goal, so this script rebuilds the descendant raw and
whitened CSVs with the same OpenPI scorer used by the other descendant cells.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import ablation_scorer_allcells as asc  # noqa: E402
import analyze_verification as av  # noqa: E402

J = asc.J
HEADER = (
    ["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
     "attack", "attack_strength", "m", "s_true"]
    + [f"s_false_{i+1}" for i in range(J)]
)

ROLLOUT_DIR = ROOT / "eval_out/libero10_from_goal/rollouts/none/task_rollout"
RAW_OUT = ROOT / "attack_c_data/per_episode_scores_descendant_raw/pi05_libero_descendant.csv"
WHIT_OUT = ROOT / "attack_c_data/per_episode_scores_descendant_whit/pi05_libero_descendant.csv"


def row_for(path: Path):
    scored = asc.score_openpi(path)
    if scored is None:
        return None
    variant, st_w, sf_w, st_r, sf_r = scored
    base = [
        f"pi0.5|libero_10|descendant|partial|map|{path.stem}",
        variant,
        "pi0.5",
        "libero_10",
        "partial",
        "0.2188",
        "map",
        "descendant",
        "",
        1,
    ]
    raw = base + [f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r]
    whit = base + [f"{st_w:.8f}"] + [f"{x:.8f}" for x in sf_w]
    return raw, whit


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    cal = av.calibrate(df)
    rng = np.random.default_rng(0)
    return {
        "n_wm": int(len(cal.z_h1)),
        "n_plain": int(len(cal.z_h0_plain)),
        "auc1_vs_plain": float(av.auc(cal.z_h1, cal.z_h0_plain)),
        "auc1_vs_decoy": float(av.auc(cal.z_h1, cal.z_null.ravel())),
        "auc16": float(av.auc_group(cal.z_h1, cal.z_null, 16, rng)),
        "tpr16_at_1pct": float(av.tpr_point(cal.z_h1, cal.z_null, 16, 0.01, rng)),
    }


def main():
    paths = sorted(ROLLOUT_DIR.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZs found under {ROLLOUT_DIR}")
    raw_rows, whit_rows = [], []
    skipped = 0
    for path in paths:
        try:
            rows = row_for(path)
        except Exception as exc:
            skipped += 1
            print(f"skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if rows is None:
            skipped += 1
            continue
        raw, whit = rows
        raw_rows.append(raw)
        whit_rows.append(whit)

    write_csv(RAW_OUT, raw_rows)
    write_csv(WHIT_OUT, whit_rows)
    print(f"wrote {RAW_OUT} rows={len(raw_rows)} skipped={skipped}")
    print(f"wrote {WHIT_OUT} rows={len(whit_rows)} skipped={skipped}")
    for label, path in [("raw", RAW_OUT), ("whitened", WHIT_OUT)]:
        m = metrics(path)
        print(
            f"{label}: n_wm={m['n_wm']} n_plain={m['n_plain']} "
            f"AUC1_plain={m['auc1_vs_plain']:.4f} AUC1_decoy={m['auc1_vs_decoy']:.4f} "
            f"AUC16={m['auc16']:.4f} TPR16@1%={m['tpr16_at_1pct']:.4f}"
        )


if __name__ == "__main__":
    main()
