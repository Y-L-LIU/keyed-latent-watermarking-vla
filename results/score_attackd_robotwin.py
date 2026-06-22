#!/usr/bin/env python3
"""Score pi0.5 / RoboTwin Attack-D (direct adversarial) rollouts with the SAME canonical RAW
matched-filter pipeline used everywhere else (ablation_scorer_allcells.score_openpi -> raw
s_true/s_false, then fuse_detectors.all_metrics). Mirror of score_attackd_lingbot.py but for the
openpi pipeline, so the pi0.5/RoboTwin cost-utility points are co-plottable with pi0.5/LIBERO and
both LingBot arms.

Per cell: det@G1 (AUC_1), det@G16 (group AUC |G|=16), DIR@1%FAR, and task success (wm/plain) from
each NPZ's `success`. lambda=0 = the no-attack descendant trained in the same sweep
(attack_d_rt_lam0), so the anchor is internally consistent with the attacked cells.

Rollouts: eval_out/openpi_robotwin_attackd/lam<L>/<task>__s0/.../episode_*_{watermarked,plain}.npz.
Writes per-episode raw CSVs to
per_episode_scores_attackd_robotwin/ + _attackd_robotwin_summary.csv.

Usage: python score_attackd_robotwin.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc  # noqa: E402  (score_openpi, J, DATA)
import fuse_detectors as fd             # noqa: E402

J = asc.J
DATA = asc.DATA
OUT = DATA / "attack_c_data" / "per_episode_scores_attackd_robotwin"
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

ER = "eval_out/openpi_robotwin_attackd"   # attacked rollouts; lam0 included (no-attack anchor)
# (label, attack_strength, [rollout dir]); rglob picks up both wm + plain variants under each.
CONDS = [
    ("lam0",  "0",  [f"{ER}/lam0"]),
    ("lam1",  "1",  [f"{ER}/lam1"]),
    ("lam5",  "5",  [f"{ER}/lam5"]),
    ("lam30", "30", [f"{ER}/lam30"]),
]


def build_cell(label, dirs):
    """Score every NPZ with the RAW openpi matched filter; return (rows, succ_wm, succ_pl, nw, npl)."""
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    rows = []
    succ = {"watermarked": [], "plain": []}
    for p in npz:
        try:
            r = asc.score_openpi(p)            # -> (variant, st_w, sf_w, st_r, sf_r)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, _stw, _sfw, st_r, sf_r = r
        # allow_pickle: trusted local NPZs from our own eval_robotwin_watermark_map rollout.
        d = np.load(p, allow_pickle=True)
        if "success" in d.files:
            succ.setdefault(variant, []).append(bool(d["success"]))
        eid = f"pi05|robotwin_10|attackd_{label}|partial|map|{p.parent.parent.name}_{p.stem}_{variant}"
        rows.append([eid, variant, "pi05", "robotwin_10", "partial", "", "map",
                     f"attackd_{label}", "", 1, f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
    sw = float(np.mean(succ["watermarked"])) if succ["watermarked"] else float("nan")
    sp = float(np.mean(succ["plain"])) if succ["plain"] else float("nan")
    return rows, sw, sp, len(succ["watermarked"]), len(succ["plain"])


def main():
    OUT.mkdir(exist_ok=True)
    print("pi0.5/RoboTwin Attack-D  RAW matched filter (asc.score_openpi)  -> co-plottable with the rest\n")
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
        with open(OUT / f"pi05_robotwin_attackd_{label}.csv", "w", newline="") as f:
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
        out = OUT / "_attackd_robotwin_summary.csv"
        sdf.to_csv(out, index=False)
        print(f"\nwrote per-episode CSVs + summary -> {out}")


if __name__ == "__main__":
    main()
