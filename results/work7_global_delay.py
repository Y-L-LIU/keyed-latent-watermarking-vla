#!/usr/bin/env python3
"""Apples-to-apples fix: re-score the lag-confounded pi0.5/LIBERO cells with
work-7 + GLOBAL lag-search, matching the committed all-32 baseline's setting.

The committed per_episode_scores_raw/ was built with LAG_SEARCH=1 LAG_MODE=global
(rescore_lag_cells.py). Empirically only delay_1/2/3 and ema_0.2 differ from
lag-0 (the only attacks with a real temporal/group-delay component). My earlier
work-7 table scored those cells at lag-0 -> unfair vs the lag-searched all-32.

Here: estimate ONE deployment tau* per cell from the watermarked episodes'
pooled keyed response over dims 0..6 (asc.estimate_global_tau), then score every
episode + its 32 decoys at that single tau* (asc.lag_raw_at). Compares, per cell:
  a32_lag0 (L0) | a32_global (committed) | w7_lag0 | w7_global (NEW)
on det@G16 / identR1@16 / DIR1%@16. Pure offline (reads stored npz)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc       # noqa: E402
import build_raw_perep as bp                  # noqa: E402
import fuse_detectors as fd                   # noqa: E402
from build_work7_lagsearch import _lag_feats_dims  # noqa: E402  (dim-restricted lag features)

asc.LAG_SEARCH = True
op_wm = asc.op_wm
J = asc.J
DATA = asc.DATA
DIMS = list(range(0, 7))
RAW = DATA / "attack_c_data" / "per_episode_scores_raw"          # committed = a32 global
L0 = DATA / "attack_c_data" / "per_episode_scores_raw_L0"        # a32 lag-0
W7 = DATA / "attack_c_data" / "per_episode_scores_raw_work7"     # w7 lag-0
CELLS = {c[1] + ("_" + c[2] if c[2] else ""): c[4]
         for c in bp.CONDS if c[0] == "pi0.5/libero_10"}
TARGETS = ["ema_0.2", "delay_1", "delay_2", "delay_3"]           # the lag-confounded cells


def w7_global_df(dirs):
    """Two-pass global tau* on dims 0..6: estimate one tau* from the pooled
    watermarked keyed response, then score all episodes + decoys at that tau*."""
    npz = []
    for dd in dirs:
        npz += sorted((DATA / dd).rglob("*.npz"))
    recs = []                                   # (variant, feats)
    for p in npz:
        try:
            d = np.load(p, allow_pickle=True)   # trusted local rollout artifact
            f = _lag_feats_dims(d, DIMS)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if f is None:
            continue
        variant = str(d["variant"]) if "variant" in d.files else "watermarked"
        recs.append((variant, f))
    if not recs:
        return None, None
    tstar = asc.estimate_global_tau(recs)
    rows = []
    for variant, f in recs:
        st, sf = asc.lag_raw_at(f, tstar)
        rows.append([variant, st] + list(sf))
    cols = ["variant", "s_true"] + [f"s_false_{i+1}" for i in range(J)]
    return pd.DataFrame(rows, columns=cols), tstar


def metrics_csv(p):
    is_wm, zt, zd = fd.keyed_z(pd.read_csv(p))
    return fd.all_metrics(zt, zd, is_wm)


def main():
    print("pi0.5/LIBERO  lag-confounded cells: work-7 + GLOBAL lag-search (apples vs committed a32-global)\n")
    h = f"{'attack':10s}{'tau*':>5s} | {'detAUC@16':^30s} | {'identR1@16':^30s} | {'DIR1%@16':^30s}"
    print(h)
    sub = f"{'a32_l0':>7s}{'a32_glb':>8s}{'w7_l0':>7s}{'w7_glb':>8s}"
    print(f"{'':10s}{'':>5s} | {sub} | {sub} | {sub}")
    print("-" * len(h))
    for c in TARGETS:
        a0 = metrics_csv(L0 / f"pi05_libero_10_partial_map_{c}.csv")
        ag = metrics_csv(RAW / f"pi05_libero_10_partial_map_{c}.csv")
        w0 = metrics_csv(W7 / f"pi05_libero_10_partial_map_{c}.csv")
        wdf, tstar = w7_global_df(CELLS[c])
        is_wm, zt, zd = fd.keyed_z(wdf)
        wg = fd.all_metrics(zt, zd, is_wm)

        def quad(k):
            return f"{a0[k]:7.3f}{ag[k]:8.3f}{w0[k]:7.3f}{wg[k]:8.3f}"
        print(f"{c:10s}{tstar:5d} | {quad('d16')} | {quad('r16')} | {quad('dir16')}")


if __name__ == "__main__":
    main()
