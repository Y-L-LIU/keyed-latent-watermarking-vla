#!/usr/bin/env python3
"""Stitch the all-32 and work-7 (detect 0-6) detectors for pi0.5/LIBERO attacks.

Both score the SAME stored recovered noise over nested dim supports (work-7 in
all-32). work-7 wins on per-sample attacks (drops padding-dim noise); all-32 wins
on the temporal delay attack. We fuse per KEY in z-space:

  z_fused(k) = max( z_work7(k), z_all32(k) )     for the true key AND each decoy k

(also reports the sum/average fusion for contrast). The max is applied
symmetrically to the 32 decoys, so the leave-one-out null absorbs the best-of-2
inflation -- the same honesty device the lag search uses over tau. Pure offline:
reads per_episode_scores_raw/ (all-32) and per_episode_scores_raw_work7/.

Prints all-32 / work-7 / fused(max) for detAUC@{1,16}, identR1@{1,16}, DIR1%@16.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av     # noqa: E402

EPS = av.EPS
J = 32
FALSE = [f"s_false_{i+1}" for i in range(J)]
RAW = Path("/workspace/vla/attack_c_data/per_episode_scores_raw")
W7 = Path("/workspace/vla/attack_c_data/per_episode_scores_raw_work7")


def keyed_z(df):
    """Return (is_wm, z_true(n,), z_decoy(n,32)) -- same z as av.calibrate."""
    sf = df[FALSE].to_numpy(float); st = df["s_true"].to_numpy(float)
    is_wm = (df["variant"].to_numpy() == "watermarked")
    mu = sf.mean(1); sd = sf.std(1, ddof=1)
    zt = (st - mu) / (sd + EPS)
    csum = sf.sum(1, keepdims=True); loo_mu = (csum - sf) / (J - 1)
    sq = (sf ** 2).sum(1, keepdims=True)
    loo_var = ((sq - sf ** 2) - (J - 1) * loo_mu ** 2) / (J - 2)
    zd = (sf - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + EPS)
    return is_wm, zt, zd


def det_auc(zt, zd, is_wm, G, rng):
    z_h1 = zt[is_wm]; z_null = zd  # all rows, matches av.calibrate's z_null_all
    if G == 1:
        return av.auc(z_h1, z_null.ravel())
    return av.auc_group(z_h1, z_null, G, rng)


def ident(zt, zd, is_wm, G, rng, n_trials=20000):
    zt_w = zt[is_wm]; zd_w = zd[is_wm]
    n = len(zt_w); idx = rng.integers(0, n, size=(n_trials, G))
    Tt = zt_w[idx].sum(1); Td = zd_w[idx].sum(1, dtype=float) if G == 1 else zd_w[idx].sum(axis=1)
    Td = zd_w[idx].sum(axis=1)                      # (n_trials, 32)
    r1 = float((Tt >= Td.max(1)).mean())
    return r1


def dir_far(zt, zd, is_wm, G, rng, far=0.01, n_trials=20000):
    zt_w, zd_w = zt[is_wm], zd[is_wm]; zt_p, zd_p = zt[~is_wm], zd[~is_wm]
    if len(zt_p) == 0:
        return float("nan")
    iw = rng.integers(0, len(zt_w), size=(n_trials, G))
    Ttg = zt_w[iw].sum(1); Tdg = zd_w[iw].sum(axis=1)
    rank1 = Ttg >= Tdg.max(1)
    ip = rng.integers(0, len(zt_p), size=(n_trials, G))
    Tti = zt_p[ip].sum(1); Tdi = zd_p[ip].sum(axis=1)
    smax = np.maximum(Tti, Tdi.max(1))
    tau = float(np.quantile(smax, 1 - far))
    return float((rank1 & (Ttg >= tau)).mean())


def all_metrics(zt, zd, is_wm):
    rng = np.random.default_rng(av.RNG_SEED)
    return dict(d1=det_auc(zt, zd, is_wm, 1, rng), d16=det_auc(zt, zd, is_wm, 16, rng),
                r1=ident(zt, zd, is_wm, 1, rng), r16=ident(zt, zd, is_wm, 16, rng),
                dir16=dir_far(zt, zd, is_wm, 16, rng))


CELLS = ["clean", "clip_0.5", "clip_1.0", "clip_2.0", "ema_0.2", "ema_0.5", "ema_0.8",
         "jitter_0.005", "jitter_0.01", "jitter_0.05", "jitter_0.1",
         "delay_1", "delay_2", "delay_3"]


def main():
    print("pi0.5/LIBERO  fuse all-32 + work-7 (per-key max in z-space)\n")
    h = (f"{'attack':13s} | {'detAUC@16':^21s} | {'identR1@16':^21s} | {'DIR1%@16':^21s}")
    print(h); print(f"{'':13s} | {'a32':>6s}{'w7':>7s}{'FUSE':>8s} | {'a32':>6s}{'w7':>7s}{'FUSE':>8s} | {'a32':>6s}{'w7':>7s}{'FUSE':>8s}")
    print("-" * len(h))
    rows = []
    for c in CELLS:
        pa = RAW / f"pi05_libero_10_partial_map_{c}.csv"
        pw = W7 / f"pi05_libero_10_partial_map_{c}.csv"
        if not (pa.exists() and pw.exists()):
            continue
        da = pd.read_csv(pa).set_index("episode_id")
        dw = pd.read_csv(pw).set_index("episode_id")
        common = da.index.intersection(dw.index)
        da = da.loc[common]; dw = dw.loc[common]
        is_wm, zta, zda = keyed_z(da.reset_index())
        _,    ztw, zdw = keyed_z(dw.reset_index())
        zt_f = np.maximum(zta, ztw); zd_f = np.maximum(zda, zdw)
        ma = all_metrics(zta, zda, is_wm)
        mw = all_metrics(ztw, zdw, is_wm)
        mf = all_metrics(zt_f, zd_f, is_wm)
        print(f"{c:13s} | {ma['d16']:6.3f}{mw['d16']:7.3f}{mf['d16']:8.3f} | "
              f"{ma['r16']:6.3f}{mw['r16']:7.3f}{mf['r16']:8.3f} | "
              f"{ma['dir16']:6.3f}{mw['dir16']:7.3f}{mf['dir16']:8.3f}")
        rows.append((c, ma, mw, mf))
    # quick dominance summary
    print("\nFUSE >= max(a32, w7) per metric?  (count of cells where fuse is within 0.005 of the better arm)")
    for key, lbl in [("d16", "detAUC@16"), ("r16", "identR1@16"), ("dir16", "DIR1%@16")]:
        ok = sum(1 for _, ma, mw, mf in rows
                 if not np.isnan(mf[key]) and mf[key] >= max(ma[key], mw[key]) - 0.005)
        print(f"   {lbl:12s}: {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
