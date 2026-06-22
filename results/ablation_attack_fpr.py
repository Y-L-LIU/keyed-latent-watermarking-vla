#!/usr/bin/env python3
"""Whitening's last stand: realized FPR under attack (LingBot/RoboTwin).

Clean already shows the whitened score is more conservative than raw on this cell
(0.2% vs 2.6% at nominal 1%, |G|=16). The paper notes jitter inflates the realized
FPR because it injects chunk-level correlation into plain rollouts that the
false-key null cannot separate. If the whitening keeps that inflation honest where
the raw score blows up, it earns its place; if both blow up (or raw is fine), it
does not.

For each attack tag: re-score wm+plain npz both ways (one MAP pass), then report
realized plain-FPR at the nominal 1% operating point (|G|=16) and the detection
AUC, whitened vs raw. For jitter, sweep |G| -- the extreme case.
Per-tag score caches make re-runs instant.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av
import ablation_detection as ad           # reuse score_episode (map pass -> whitened+raw)

J = 32
EVAL = ad.EVAL
TAGS = ["clip_1.0", "clip_2.0", "ema_0.3", "ema_0.5", "jitter_0.02", "delay_1", "delay_2", "delay_3"]


def build(tag):
    cache = HERE / f"cache_atk_{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for p in sorted((EVAL / "atk" / tag).rglob("*.npz")):
        try:
            r = ad.score_episode(p)
        except Exception:
            continue
        if r is None:
            continue
        rows.append([r["variant"], r["st_w"], r["st_r"]] + r["sf_w"] + r["sf_r"])
    cols = ["variant", "st_w", "st_r"] + [f"sfw_{i+1}" for i in range(J)] + [f"sfr_{i+1}" for i in range(J)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(cache, index=False)
    return df


def _zcal(st, sf):
    mu = sf.mean(1); sd = sf.std(1, ddof=1)
    zt = (st - mu) / (sd + av.EPS)
    cs = sf.sum(1, keepdims=True); lm = (cs - sf) / (J - 1)
    sq = (sf ** 2).sum(1, keepdims=True)
    lv = ((sq - sf ** 2) - (J - 1) * lm ** 2) / (J - 2)
    zn = (sf - lm) / (np.sqrt(np.clip(lv, 0, None)) + av.EPS)
    return zt, zn


def fpr_auc(st, sf, is_wm, G, Q=0.01, NT=20000):
    zt, zn = _zcal(st, sf)
    z_pl, z_wm = zt[~is_wm], zt[is_wm]
    rng = np.random.default_rng(0)
    ep = rng.integers(0, zn.shape[0], size=(NT, G)); key = rng.integers(0, J, size=(NT, G))
    thr = np.quantile(zn[ep, key].sum(1), 1 - Q)
    fpr = float((z_pl[rng.integers(0, len(z_pl), size=(NT, G))].sum(1) >= thr).mean()) if len(z_pl) else float("nan")
    auc1 = av.auc(z_wm, zn.ravel())
    return fpr, auc1


def cols(df):
    is_wm = df["variant"].to_numpy() == "watermarked"
    W = (df["st_w"].to_numpy(float), df[[f"sfw_{i+1}" for i in range(J)]].to_numpy(float))
    R = (df["st_r"].to_numpy(float), df[[f"sfr_{i+1}" for i in range(J)]].to_numpy(float))
    return is_wm, W, R


def main():
    print("realized plain-FPR @ nominal 1% (|G|=16) and detection AUC(|G|=1), whitened vs raw\n")
    print(f"{'attack':12s}{'n(wm/pl)':>11s} | {'whit FPR':>9s}{'raw FPR':>9s} | {'whit AUC':>9s}{'raw AUC':>9s}")
    print("-" * 72)
    for tag in TAGS:
        df = build(tag)
        is_wm, W, R = cols(df)
        nwm, npl = int(is_wm.sum()), int((~is_wm).sum())
        wf, wa = fpr_auc(*W, is_wm, 16)
        rf, ra = fpr_auc(*R, is_wm, 16)
        flag = "  <- raw worse" if (rf > wf + 0.01) else ("  <- whit worse" if wf > rf + 0.01 else "")
        print(f"{tag:12s}{f'{nwm}/{npl}':>11s} | {wf:9.3f}{rf:9.3f} | {wa:9.3f}{ra:9.3f}{flag}")

    # extreme case: jitter, sweep |G|
    print("\njitter_0.02 -- realized plain-FPR vs |G| (the inflation regime):")
    df = build("jitter_0.02"); is_wm, W, R = cols(df)
    print(f"{'|G|':>5s}{'whit FPR':>10s}{'raw FPR':>10s}")
    for G in (1, 2, 4, 8, 16, 32):
        wf, _ = fpr_auc(*W, is_wm, G); rf, _ = fpr_auc(*R, is_wm, G)
        print(f"{G:5d}{wf:10.3f}{rf:10.3f}")


if __name__ == "__main__":
    main()
