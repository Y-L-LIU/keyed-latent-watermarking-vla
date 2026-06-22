#!/usr/bin/env python3
"""Score the Attack-D (direct adversarial) pi0.5/LIBERO rollouts with the canonical
work-7 (detect dims 0-6) + GLOBAL lag-search RAW detector, balanced 50wm/50pl,
reusing rescore_work7_global_5050's build_cell + fuse_detectors metrics. Prints
det@G1 (AUC_1), det@G16 (group AUC |G|=16), DIR@1%FAR; writes per-episode CSVs to
per_episode_scores_work7_global_5050/.

Usage: python score_attackd.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rescore_work7_global_5050 as R  # noqa: E402  (build_cell, write_csv, OUT, J)
import build_raw_perep as bp           # noqa: E402  (HEADER, META)
import fuse_detectors as fd            # noqa: E402

ROOT = "attack_c_data/rollouts/openpi_libero_attackd"
_ALL = [
    ("pi0.5/libero_10", "attackd", "lam0",  "pi05_libero_10_attackd_lam0",  [f"{ROOT}/lam0"]),
    ("pi0.5/libero_10", "attackd", "lam1",  "pi05_libero_10_attackd_lam1",  [f"{ROOT}/lam1"]),
    ("pi0.5/libero_10", "attackd", "lam5",  "pi05_libero_10_attackd_lam5",  [f"{ROOT}/lam5"]),
    ("pi0.5/libero_10", "attackd", "lam30", "pi05_libero_10_attackd_lam30", [f"{ROOT}/lam30"]),
    ("pi0.5/libero_10", "attackd", "lam1dc","pi05_libero_10_attackd_lam1dc",[f"{ROOT}/lam1dc"]),
]
# Only score cells whose rollout dir has NPZ.
from pathlib import Path as _P
DATA = _P("/workspace/vla")
CONDS = [c for c in _ALL if any((DATA / d).exists() and list((DATA / d).rglob("*.npz")) for d in c[4])]


def main():
    R.OUT.mkdir(exist_ok=True)
    print("Attack-D pi0.5/LIBERO  work-7 (detect 0-6) + GLOBAL lag-search  @ 50wm/50pl\n")
    hdr = (f"{'cell':28s}{'n(wm/pl)':>10s}{'tau*':>6s} | "
           f"{'det@G1':>8s}{'det@G16':>9s} | {'DIR1%@G16':>11s}")
    print(hdr); print("-" * len(hdr))
    rowsout = []
    for cond in CONDS:
        stem, rows, tstar, short = R.build_cell(cond)
        if not rows:
            print(f"{cond[3]:28s}  (no rows)"); continue
        R.write_csv(R.OUT / f"{stem}.csv", rows)
        df = pd.DataFrame(rows, columns=bp.HEADER)
        for c in ["s_true"] + [f"s_false_{i+1}" for i in range(R.J)]:
            df[c] = df[c].astype(float)
        is_wm, zt, zd = fd.keyed_z(df)
        mm = fd.all_metrics(zt, zd, is_wm)
        nwm, npl = int(is_wm.sum()), int((~is_wm).sum())
        warn = f"  !short:{short}" if short else ""
        print(f"{cond[3]:28s}{f'{nwm}/{npl}':>10s}{tstar:6d} | "
              f"{mm['d1']:8.3f}{mm['d16']:9.3f} | {mm['dir16']:11.3f}{warn}")
        rowsout.append([cond[2], nwm, npl, tstar, mm['d1'], mm['d16'], mm['r1'], mm['r16'], mm['dir16']])
    if rowsout:
        sdf = pd.DataFrame(rowsout, columns=["lam", "n_wm", "n_pl", "tau_star",
                                             "detG1", "detG16", "idR1G1", "idR1G16", "DIR1G16"])
        out = R.OUT / "_attackd_summary.csv"
        sdf.to_csv(out, index=False)
        print(f"\nwrote per-episode CSVs + summary -> {out}")


if __name__ == "__main__":
    main()
