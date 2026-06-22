#!/usr/bin/env python3
"""Raw + whitened per-episode CSVs for the LoRA descendants (Stage B).

Same machinery as build_raw_perep (reuses ablation_scorer_allcells scoring), for
the 3 descendant cells. Feeds tab:descendant (group AUC/TPR/pairwise) and the
descendant rows of tab:identification. Writes per_episode_scores_descendant_raw/
and per_episode_scores_descendant_whit/. Gate vs committed descendant CSVs.

Usage: build_raw_desc.py [--probe N]
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc  # noqa: E402

J = asc.J
DATA = asc.DATA
RAW = DATA / "attack_c_data" / "per_episode_scores_descendant_raw"
WHIT = DATA / "attack_c_data" / "per_episode_scores_descendant_whit"
COMMITTED = DATA / "attack_c_data" / "per_episode_scores_descendant"
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

META = {
    "lingbot/libero_10":  dict(model="lingbot", dataset="libero_10",  obs_ratio="0.2333", pipe="lingbot", preset="libero"),
    "lingbot/robotwin10": dict(model="lingbot", dataset="robotwin10", obs_ratio="0.5333", pipe="lingbot", preset="robotwin"),
    "pi0.5/robotwin10":   dict(model="pi0.5",   dataset="robotwin10", obs_ratio="0.4375", pipe="openpi"),
}
# (cell, filestem, [dirs])
CONDS = [
    ("lingbot/libero_10",  "lingbot_libero_descendant",
        ["eval_out/lingbot_libero10_descendant/libero_10", "eval_out/lingbot_libero10_descendant_plain/libero_10"]),
    ("lingbot/robotwin10", "lingbot_robotwin_descendant", ["eval_out/descendant_lingbot_n100"]),
    ("pi0.5/robotwin10",   "pi05_robotwin_descendant",    ["eval_out/descendant_openpi_n100"]),
]


def episode_id(cell, p, variant):
    m = META[cell]
    if m["pipe"] == "lingbot":
        return f"lingbot|{m['dataset']}|descendant|partial|map|{p.parent.name}_{p.stem}_{variant}"
    return f"pi0.5|{m['dataset']}|descendant|partial|map|{p.stem}"


def score_one(cell, p):
    m = META[cell]
    if m["pipe"] == "lingbot":
        return asc.score_lingbot(p, asc.PRESETS[m["preset"]])
    return asc.score_openpi(p)


def build(cond, probe=0):
    cell, stem, dirs = cond
    m = META[cell]
    rr, rw = [], []
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    if probe:
        npz = npz[:probe]
    for p in npz:
        try:
            r = score_one(cell, p)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, st_w, sf_w, st_r, sf_r = r
        eid = episode_id(cell, p, variant)
        base = [eid, variant, m["model"], m["dataset"], "partial", m["obs_ratio"], "map", "descendant", "", 1]
        rr.append(base + [f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
        rw.append(base + [f"{st_w:.8f}"] + [f"{x:.8f}" for x in sf_w])
    return stem, rr, rw


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)


def gate(stem, rw):
    import analyze_verification as av
    cpath = COMMITTED / f"{stem}.csv"
    if not cpath.exists():
        return f"(no committed {stem})"
    cal_c = av.calibrate(pd.read_csv(cpath)); auc_c = av.auc(cal_c.z_h1, cal_c.z_null.ravel())
    mdf = pd.DataFrame(rw, columns=HEADER)
    for c in ["s_true"] + [f"s_false_{i+1}" for i in range(J)]:
        mdf[c] = mdf[c].astype(float)
    cal_m = av.calibrate(mdf); auc_m = av.auc(cal_m.z_h1, cal_m.z_null.ravel())
    return f"whitAUC={auc_m:.3f} committedAUC={auc_c:.3f} {'OK' if abs(auc_m-auc_c)<0.02 else 'DIFF'}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()
    if not args.probe:
        RAW.mkdir(exist_ok=True); WHIT.mkdir(exist_ok=True)
    for cond in CONDS:
        stem, rr, rw = build(cond, probe=args.probe)
        g = "" if (args.probe and len(rw) < 5) else gate(stem, rw)
        print(f"{stem:32s} n={len(rr):4d}  {g}")
        if args.probe and rw:
            print(f"      id0={rw[0][0]}")
        if not args.probe:
            write_csv(RAW / f"{stem}.csv", rr); write_csv(WHIT / f"{stem}.csv", rw)
    print("DONE")


if __name__ == "__main__":
    main()
