#!/usr/bin/env python3
"""Clean-cell diagnostic: score each verifier block on the metric it CONTROLS.

Detection AUC is rank-based and threshold-free, so it is blind to two of the
three blocks. This measures each block on the right axis (clean LingBot/RoboTwin):

  WMF whitening   -> closed-set identification rank-1 (true key vs 32 decoys),
                     the 33-way decision the whitening is designed for, plus
                     detection AUC for reference.
  false-key calib -> realized FPR at a nominal 1% threshold on PLAIN rollouts
                     (literal H0). Calibration's job is to keep that rate honest.

Re-scores the 200 clean npz once, caches whitened + raw (s_true, s_false[32]) to
results/clean_scores_cache.csv, then computes everything from the cache.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av                 # noqa: E402
import ablation_detection as ab                   # noqa: E402  (reuse score_episode/build_dir)

CACHE = HERE / "clean_scores_cache.csv"
J = ab.J


def build_cache():
    rows = []
    for k, p in enumerate(sorted(ab.CLEAN_DIR.rglob("*.npz"))):
        r = ab.score_episode(p)
        if r is None:
            continue
        rows.append([r["variant"], r["st_w"], r["st_r"]] + r["sf_w"] + r["sf_r"])
    cols = (["variant", "st_w", "st_r"]
            + [f"sfw_{i+1}" for i in range(J)] + [f"sfr_{i+1}" for i in range(J)])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(CACHE, index=False)
    return df


def detection_auc(st, sf, is_wm):
    """|G|=1 calibrated detection AUC for a given scorer's (s_true, s_false)."""
    mu = sf.mean(axis=1); sd = sf.std(axis=1, ddof=1)
    z_true = (st - mu) / (sd + av.EPS)
    # leave-one-out decoy z (the null), all rows -- mirrors av.calibrate
    csum = sf.sum(axis=1, keepdims=True)
    loo_mu = (csum - sf) / (J - 1)
    sq = (sf ** 2).sum(axis=1, keepdims=True)
    loo_var = ((sq - sf ** 2) - (J - 1) * loo_mu ** 2) / (J - 2)
    z_null = (sf - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + av.EPS)
    return av.auc(z_true[is_wm], z_null.ravel())


def ident_rank1(st, sf, is_wm):
    """Closed-set rank-1 (|G|=1): fraction of wm episodes whose true-key score
    beats all 32 decoys. Calibration-invariant, so it isolates the scorer."""
    win = st[is_wm] > sf[is_wm].max(axis=1)
    return float(win.mean())


def realized_fpr(st, sf, is_wm, calibrated, nominal=0.01):
    """FPR on plain rollouts at a threshold set to the nominal quantile of the
    false-key null. calibrated=True applies per-episode Z; False thresholds raw."""
    if calibrated:
        mu = sf.mean(axis=1); sd = sf.std(axis=1, ddof=1)
        score = (st - mu) / (sd + av.EPS)
        csum = sf.sum(axis=1, keepdims=True)
        loo_mu = (csum - sf) / (J - 1)
        sq = (sf ** 2).sum(axis=1, keepdims=True)
        loo_var = ((sq - sf ** 2) - (J - 1) * loo_mu ** 2) / (J - 2)
        null = ((sf - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + av.EPS)).ravel()
    else:
        score = st
        null = sf.ravel()
    thr = np.quantile(null, 1 - nominal)
    return float((score[~is_wm] >= thr).mean())


def main():
    df = pd.read_csv(CACHE) if CACHE.exists() else build_cache()
    is_wm = (df["variant"].to_numpy() == "watermarked")
    st_w = df["st_w"].to_numpy(float); sf_w = df[[f"sfw_{i+1}" for i in range(J)]].to_numpy(float)
    st_r = df["st_r"].to_numpy(float); sf_r = df[[f"sfr_{i+1}" for i in range(J)]].to_numpy(float)
    n_wm, n_pl = int(is_wm.sum()), int((~is_wm).sum())
    print(f"clean LingBot/RoboTwin: {n_wm} wm + {n_pl} plain\n")

    print("  scorer    detAUC   ident-R1   FPR@1%(calib)  FPR@1%(raw-thresh)")
    for name, st, sf in [("whitened", st_w, sf_w), ("raw", st_r, sf_r)]:
        a = detection_auc(st, sf, is_wm)
        r1 = ident_rank1(st, sf, is_wm)
        fc = realized_fpr(st, sf, is_wm, calibrated=True)
        fr = realized_fpr(st, sf, is_wm, calibrated=False)
        print(f"  {name:8s}{a:8.3f}{r1:11.3f}{fc:15.3f}{fr:20.3f}")
    print("\n  (FPR nominal = 0.010; calib should keep it <=nominal, raw-thresh may blow up)")


if __name__ == "__main__":
    main()
