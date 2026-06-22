#!/usr/bin/env python3
"""Score LingBot Attack-D (direct adversarial) LIBERO-10 rollouts with the SAME canonical
RAW matched-filter pipeline pi0.5 used (ablation_scorer_allcells.score_lingbot -> raw s_true/
s_false, then fuse_detectors.all_metrics), so the LingBot cost-utility points are directly
co-plottable with pi0.5's results/score_attackd.py output.

Per cell: det@G1 (AUC_1), det@G16 (group AUC |G|=16), DIR@1%FAR, and task success (wm/plain)
read from each NPZ's `success` field. lambda=0 anchor = the existing LingBot LIBERO descendant
rollouts (the same ones build_raw_desc.py scores), so the no-attack point matches the paper.

Writes per-episode raw CSVs to per_episode_scores_attackd_lingbot/ + _attackd_lingbot_summary.csv.

Usage: python score_attackd_lingbot.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc  # noqa: E402  (score_lingbot, PRESETS, J, DATA)
import fuse_detectors as fd             # noqa: E402

J = asc.J
DATA = asc.DATA
OUT = DATA / "attack_c_data" / "per_episode_scores_attackd_lingbot"
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

ER = "eval_out/lingbot_attackd"           # attacked eval rollouts
DR = "eval_out"                            # descendant (lambda=0) rollouts
# (label, attack_strength, [rollout dirs (wm + plain)])
CONDS = [
    ("lam0", "0",   [f"{DR}/lingbot_libero10_descendant/libero_10",
                     f"{DR}/lingbot_libero10_descendant_plain/libero_10"]),
    ("tw100", "100", [f"{ER}/lam10_tw100_watermarked/libero_10", f"{ER}/lam10_tw100_plain/libero_10"]),
    ("tw10",  "10",  [f"{ER}/lam10_tw10_watermarked/libero_10",  f"{ER}/lam10_tw10_plain/libero_10"]),
    ("tw1",   "1",   [f"{ER}/lam10_tw1_watermarked/libero_10",   f"{ER}/lam10_tw1_plain/libero_10"]),
    ("tw0",   "inf", [f"{ER}/lam10_tw0_watermarked/libero_10",   f"{ER}/lam10_tw0_plain/libero_10"]),
]


def build_cell(label, dirs):
    """Score every NPZ with the RAW lingbot matched filter; return (rows, succ_wm, succ_pl)."""
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    rows = []
    succ = {"watermarked": [], "plain": []}
    for p in npz:
        try:
            r = asc.score_lingbot(p, asc.PRESETS["libero"])  # -> (variant, st_w, sf_w, st_r, sf_r)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, _stw, _sfw, st_r, sf_r = r
        d = np.load(p, allow_pickle=True)
        if "success" in d.files:
            succ.setdefault(variant, []).append(bool(d["success"]))
        eid = f"lingbot|libero_10|attackd_{label}|partial|map|{p.parent.name}_{p.stem}_{variant}"
        rows.append([eid, variant, "lingbot", "libero_10", "partial", "0.2333", "map",
                     f"attackd_{label}", "", 1, f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
    sw = float(np.mean(succ["watermarked"])) if succ["watermarked"] else float("nan")
    sp = float(np.mean(succ["plain"])) if succ["plain"] else float("nan")
    return rows, sw, sp, len(succ["watermarked"]), len(succ["plain"])


def main():
    OUT.mkdir(exist_ok=True)
    print("LingBot Attack-D  RAW matched filter (asc.score_lingbot, preset=libero)  -> co-plottable with pi0.5\n")
    hdr = (f"{'cell':10s}{'atk_str':>8s}{'n(wm/pl)':>10s} | {'succ_wm':>8s}{'succ_pl':>8s} | "
           f"{'det@G1':>8s}{'det@G16':>9s}{'DIR1%':>8s}")
    print(hdr); print("-" * len(hdr))
    summ = []
    for label, atk_str, dirs in CONDS:
        if not any((DATA / d).exists() and list((DATA / d).rglob("*.npz")) for d in dirs):
            print(f"{label:10s}{atk_str:>8s}  (no rollouts yet)"); continue
        rows, sw, sp, nw, npl = build_cell(label, dirs)
        if not rows:
            print(f"{label:10s}  (no scored rows)"); continue
        with open(OUT / f"lingbot_libero_attackd_{label}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
        df = pd.DataFrame(rows, columns=HEADER)
        for c in ["s_true"] + [f"s_false_{i+1}" for i in range(J)]:
            df[c] = df[c].astype(float)
        is_wm, zt, zd = fd.keyed_z(df)
        mm = fd.all_metrics(zt, zd, is_wm)
        print(f"{label:10s}{atk_str:>8s}{f'{int(is_wm.sum())}/{int((~is_wm).sum())}':>10s} | "
              f"{sw:8.3f}{sp:8.3f} | {mm['d1']:8.3f}{mm['d16']:9.3f}{mm['dir16']:8.3f}")
        summ.append([label, atk_str, nw, npl, sw, sp, mm['d1'], mm['d16'], mm['r1'], mm['r16'], mm['dir16']])
    if summ:
        sdf = pd.DataFrame(summ, columns=["cell", "atk_strength", "n_wm", "n_pl", "succ_wm", "succ_pl",
                                          "detG1", "detG16", "idR1G1", "idR1G16", "DIR1G16"])
        out = OUT / "_attackd_lingbot_summary.csv"
        sdf.to_csv(out, index=False)
        print(f"\nwrote per-episode CSVs + summary -> {out}")


if __name__ == "__main__":
    main()
