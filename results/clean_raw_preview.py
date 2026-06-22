#!/usr/bin/env python3
"""Preview: full clean tab:main detection columns, whitened (committed) vs raw (cache).

Reports AUC(|G|=1), AUC16, TPR@1%(|G|=16) -- the three tab:main "Clean" columns --
plus the realized plain-FPR@1%(|G|=16) negative-control quantity, for both the
shipped whitened score and the raw matched filter. Shows exactly what switching
the verifier to raw buys (separation) and costs (FPR honesty) before any rewrite.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av
import make_ablation_table as mat  # reuse cell list + loaders

G, Q, NT = 16, 0.01, 20000


def realized_fpr(cal, rng):
    z0, zn = cal.z_h0_plain, cal.z_null
    if len(z0) == 0:
        return float("nan")
    ep = rng.integers(0, zn.shape[0], size=(NT, G)); key = rng.integers(0, 32, size=(NT, G))
    thr = np.quantile(zn[ep, key].sum(1), 1 - Q)
    return float((z0[rng.integers(0, len(z0), size=(NT, G))].sum(1) >= thr).mean())


def cols(df):
    cal = av.calibrate(df)
    rng = np.random.default_rng(0)
    auc1 = av.auc(cal.z_h1, cal.z_null.ravel())
    auc16 = av.auc_group(cal.z_h1, cal.z_null, G, rng)
    tpr = av.tpr_point(cal.z_h1, cal.z_null, G, Q, rng)
    fpr = realized_fpr(cal, np.random.default_rng(0))
    return auc1, auc16, tpr, fpr


def main():
    print("CLEAN tab:main columns -- whitened (shipped/committed) vs raw matched filter\n")
    hdr = f"{'cell':20s}{'scorer':9s}{'AUC':>7s}{'AUC16':>8s}{'TPR@1%':>9s}{'  | realFPR@1%':>14s}"
    print(hdr); print("-" * len(hdr))
    for model, ds, cache, csv, _r1 in mat.CELLS:
        name = f"{model}/{ds}".replace(r"$\pi_{0.5}$", "pi0.5")
        w = cols(mat._df_whitened(csv))
        r = cols(mat._df_raw(cache))
        flag = "  <-- FPR>1.5x nominal" if r[3] > Q * 1.5 else ""
        print(f"{name:20s}{'whitened':9s}{w[0]:7.3f}{w[1]:8.3f}{w[2]:9.3f}{w[3]:14.3f}")
        print(f"{'':20s}{'raw':9s}{r[0]:7.3f}{r[1]:8.3f}{r[2]:9.3f}{r[3]:14.3f}{flag}")
    print("\n(committed tab:main clean cols for reference:")
    print("  LingBot/LIBERO 0.941/1.000/1.000  LingBot/RoboTwin 0.823/1.000/0.998")
    print("  pi0.5/LIBERO   0.883/1.000/1.000  pi0.5/RoboTwin   0.793/0.999/0.984 )")


if __name__ == "__main__":
    main()
