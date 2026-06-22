#!/usr/bin/env python3
"""Test the principled stitch: work-7 detector + lag-search (shared tau).

Hypothesis: all-32's only edge on the delay attack is that delay is a TEMPORAL
shift, which the lag search already handles. So work-7 + lag-search should
recover delay WITHOUT the padding dims -- giving one detector that wins
everywhere. On non-temporal attacks tau*~0, so the lag search is a near-noop and
work-7's per-sample wins are preserved.

For each cell, recomputes per-window cosines at every lag (dim-restricted to
0..6), picks the shared deployment tau* from the true key (asc LAG_MODE=shared),
and scores true+decoys at tau*. Reuses asc lag machinery; only the cosine operand
is sliced. Reports det@G16 / identR1@16 / DIR1%@16 for:
  a32-lag0 (committed) | w7-lag0 | w7+lag | a32+lag
Pure offline (reads the stored npz).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc       # noqa: E402
import build_raw_perep as bp                  # noqa: E402
import fuse_detectors as fd                   # noqa: E402  (keyed_z + all_metrics)

asc.LAG_SEARCH = True
asc.LAG_MODE = "shared"                        # universal-viable: one tau* from true key, applied to all
op_wm = asc.op_wm
J = asc.J
DATA = asc.DATA
DIMS = list(range(0, 7))


def _lag_feats_dims(d, dims):
    selected = d["chunk_selected"]; rec = d["chunk_recovered_noise"]; ref = d["chunk_reference"]
    executed = d["chunk_executed_steps"]
    chunk_idx = d["chunk_index"] if "chunk_index" in d.files else d["chunk_chunk_index"]
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    action_dim = int(ref.shape[-1]); horizon = int(ref.shape[1]); lags = asc.lag_set(horizon)
    wins = [i for i in range(len(selected)) if selected[i] and executed[i] > 0][:5]
    if not wins:
        return None
    tv = {t: [] for t in lags}; nm = {t: [[] for _ in range(J)] for t in lags}
    for i in wins:
        rec_i = rec[i]; ref_true = ref[i]
        ctx = op_wm.WatermarkContext(chunk_index=int(chunk_idx[i]), episode_nonce=nonce)
        drefs = [op_wm.generate_keyed_reference(
                    length=horizon, action_dim=action_dim, sample_rate_hz=srate,
                    config=op_wm.InternalNoiseWatermarkConfig(secret_key=key + off, control_freq=srate),
                    context=ctx) for off in range(1, J + 1)]
        for t in lags:
            a = rec_i[t:horizon][:, dims]
            tv[t].append(asc._cosine_sim(a, ref_true[:horizon - t][:, dims]))
            for j in range(J):
                nm[t][j].append(asc._cosine_sim(a, drefs[j][:horizon - t][:, dims]))
    return ({t: np.asarray(tv[t], float) for t in lags}, {t: np.asarray(nm[t], float) for t in lags})


def score_lag(npz, dims=None):
    """dims=None -> all-32 lag-search (asc._openpi_lag_feats); else dim-restricted."""
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else "watermarked"
    feats = asc._openpi_lag_feats(d) if dims is None else _lag_feats_dims(d, dims)
    if feats is None:
        return None
    st_r, sf_r = asc._lag_raw(feats)
    return variant, st_r, sf_r


def cell_df(dirs, dims):
    rows = []
    npz = []
    for dd in dirs:
        npz += sorted((DATA / dd).rglob("*.npz"))
    for p in npz:
        try:
            r = score_lag(p, dims)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, st_r, sf_r = r
        rows.append([variant, st_r] + list(sf_r))
    cols = ["variant", "s_true"] + [f"s_false_{i+1}" for i in range(J)]
    return pd.DataFrame(rows, columns=cols)


def metrics_from_df(df):
    is_wm, zt, zd = fd.keyed_z(df)
    return fd.all_metrics(zt, zd, is_wm)


CELLS = {c[1] + ("_" + c[2] if c[2] else ""): c[4]
         for c in bp.CONDS if c[0] == "pi0.5/libero_10"}
RAW = Path("/workspace/vla/attack_c_data/per_episode_scores_raw")
W7 = Path("/workspace/vla/attack_c_data/per_episode_scores_raw_work7")
TARGETS = ["clean", "ema_0.5", "delay_1", "delay_2", "delay_3"]


def main():
    print("pi0.5/LIBERO  work-7 + lag-search (shared tau)  vs  lag-0 baselines\n")
    h = f"{'attack':10s} | {'detAUC@16':^28s} | {'identR1@16':^28s} | {'DIR1%@16':^28s}"
    print(h)
    sub = f"{'a32_l0':>7s}{'w7_l0':>7s}{'w7+lag':>7s}{'a32+lag':>8s}"
    print(f"{'':10s} | {sub} | {sub} | {sub}")
    print("-" * len(h))
    for c in TARGETS:
        dirs = CELLS[c]
        # lag-0 baselines straight from committed CSVs
        a0 = metrics_from_df(pd.read_csv(RAW / f"pi05_libero_10_partial_map_{c}.csv"))
        w0 = metrics_from_df(pd.read_csv(W7 / f"pi05_libero_10_partial_map_{c}.csv"))
        # lag-search: recompute from npz
        wl = metrics_from_df(cell_df(dirs, DIMS))
        al = metrics_from_df(cell_df(dirs, None))
        def trip(k):
            return f"{a0[k]:7.3f}{w0[k]:7.3f}{wl[k]:7.3f}{al[k]:8.3f}"
        print(f"{c:10s} | {trip('d16')} | {trip('r16')} | {trip('dir16')}")


if __name__ == "__main__":
    main()
