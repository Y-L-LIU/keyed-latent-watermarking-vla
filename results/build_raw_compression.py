#!/usr/bin/env python3
"""Per-episode 32-decoy gallery scores for the owner-side COMPRESSION variants
(prune30 / int8 quant), reusing build_raw_perep's scorer verbatim.

These rollouts were always on disk (eval_out_compression/, WM+plain paired, with
the recovered latent saved) -- they were just never fed to the raw scorer, so the
prune/quant cells were blank/deferred in tab_compression (AUC16) and absent from
identification. This wires them in with zero GPU.

Writes RAW (s_true = raw matched filter) into per_episode_scores_compression_raw/.
analyze_identification reads this dir as the weight-level "compression" group, and
the AUC16 for the pi0.5/LIBERO prune/quant cells of tab_compression is computed
from the same CSVs (see --auc).

Usage: build_raw_compression.py [--probe N] [--auc]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_raw_perep as brp          # reuse build/score_one/META/HEADER/write_csv
import analyze_verification as av      # AUC / group-AUC for the tab_compression cells

COMP = brp.DATA / "attack_c_data" / "per_episode_scores_compression_raw"

EC = "eval_out_compression"
# (cell, attack, strength, filestem, [dirs]) -- same shape as build_raw_perep.CONDS.
# LingBot keeps WM and plain in separate dirs; openpi mixes both in one task_rollout dir.
CONDS = [
    ("lingbot/libero_10",  "prune30", "", "lingbot_libero_10_partial_map_prune30",
        [f"{EC}/lingbot_libero10_prune_wm/libero_10", f"{EC}/lingbot_libero10_prune_plain/libero_10"]),
    ("lingbot/libero_10",  "quant",   "", "lingbot_libero_10_partial_map_quant",
        [f"{EC}/lingbot_libero10_quant_wm/libero_10", f"{EC}/lingbot_libero10_quant_plain/libero_10"]),
    ("lingbot/robotwin10", "prune30", "", "lingbot_robotwin_partial_map_prune30",
        [f"{EC}/lingbot_robotwin10_prune_wm_snr05", f"{EC}/lingbot_robotwin10_prune_plain_snr05"]),
    ("lingbot/robotwin10", "quant",   "", "lingbot_robotwin_partial_map_quant",
        [f"{EC}/lingbot_robotwin10_quant_wm_snr05", f"{EC}/lingbot_robotwin10_quant_plain_snr05"]),
    ("pi0.5/libero_10",    "prune30", "", "pi05_libero_10_partial_map_prune30",
        [f"{EC}/libero_goal_prune30/rollouts/none/task_rollout"]),
    ("pi0.5/libero_10",    "quant",   "", "pi05_libero_10_partial_map_quant",
        [f"{EC}/libero_goal_quant/rollouts/none/task_rollout"]),
    # pi0.5/RoboTwin compresses ACTION-ONLY (action expert + heads, ~16% of params; the
    # VLM backbone is left intact -- whole-model compression there discards the backbone).
    # wm+plain paired npz (episode_XXX_{watermarked,plain}.npz) under per-task subdirs.
    ("pi0.5/robotwin10",   "prune30", "", "pi05_robotwin10_partial_map_prune30",
        [f"{EC}/openpi_robotwin10_prune30_actiononly"]),
    ("pi0.5/robotwin10",   "quant",   "", "pi05_robotwin10_partial_map_quant",
        [f"{EC}/openpi_robotwin10_quant_actiononly"]),
]


def auc_report():
    """Per-episode AUC (|G|=1) and group AUC16 from the written CSVs, for the
    pi0.5/LIBERO tab_compression cells (and a sanity check on LingBot)."""
    rng = np.random.default_rng(av.RNG_SEED + 1)
    print(f"{'cell':42s} {'n_wm':>4} {'n_pl':>4} {'AUC':>7} {'AUC16':>7}")
    for cond in CONDS:
        stem = cond[3]
        p = COMP / f"{stem}.csv"
        if not p.exists():
            print(f"{stem:42s}  (missing)"); continue
        df = pd.read_csv(p)
        cal = av.calibrate(df)
        is_wm = df["variant"].to_numpy() == "watermarked"
        auc = av.auc(cal.z_h1, cal.z_null.ravel())
        auc16 = av.auc_group(cal.z_h1, cal.z_null, av.G_TABLE, rng)
        print(f"{stem:42s} {int(is_wm.sum()):4d} {int((~is_wm).sum()):4d} {auc:7.3f} {auc16:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--auc", action="store_true", help="report AUC/AUC16 from written CSVs and exit")
    args = ap.parse_args()
    if args.auc:
        auc_report(); return
    if not args.probe:
        COMP.mkdir(exist_ok=True)
    for cond in CONDS:
        stem, rr, rw = brp.build(cond, probe=args.probe)
        nwm = sum(1 for r in rr if r[1] == "watermarked")
        print(f"{stem:42s} n={len(rr):4d} (wm={nwm}, pl={len(rr)-nwm})")
        if args.probe and rr:
            print(f"      id[0]={rr[0][0]}")
        if not args.probe:
            brp.write_csv(COMP / f"{stem}.csv", rr)
    print("DONE" + ("" if args.probe else f" -> {COMP}"))


if __name__ == "__main__":
    main()
