#!/usr/bin/env python3
"""Raw matched filter vs. whitened WMF across ALL FOUR cells.

For each (model, dataset) cell, re-score the clean rollouts two ways from the
SAME recovered noise + decoy references:
  whitened : the pipeline's shipped WMF (PCA-whitened subspace projection)
  raw      : sum_d (feature_d - mu_decoy_d)  (no whitening, no subspace)

and report the metric each detection block governs, at |G|=1:
  detection AUC   (separability)
  identification rank-1 (true key vs 32 decoys; calibration-invariant)
  realized FPR @ nominal 1% on plain rollouts (literal H0), calib vs raw-thresh

A reproduction gate checks the whitened detection AUC against the committed CSV
(== the table) before trusting the raw comparison. Per-cell score caches let it
re-run instantly.

Two pipelines:
  lingbot  -> wan_va.wm.scoring (preset robotwin / libero)
  openpi   -> scripts/attacks/build_cost_utility_table (_score_episode/_null/_wmf)
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402

# --- lingbot scoring ------------------------------------------------------- #
LV = Path("/workspace/vla/lingbot-va/wan_va")
for _n, _p in [("wan_va", str(LV)), ("wan_va.wm", str(LV / "wm"))]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); _m.__path__ = [_p]; sys.modules[_n] = _m
from wan_va.wm.scoring import build_score_vector_from_noise, wmf_score_from_vectors  # noqa: E402
from wan_va.wm.watermark import InternalNoiseWatermarkConfig as LCfg, WatermarkContext as LCtx  # noqa: E402

# --- openpi scoring -------------------------------------------------------- #
OP = Path("/workspace/vla/openpi")
sys.path.insert(0, str(OP / "src"))
sys.path.insert(0, str(OP / "scripts" / "attacks"))
from build_cost_utility_table import _score_episode, _null_episode, _wmf_score, _cosine_sim  # noqa: E402
from openpi.policies import watermark as op_wm  # noqa: E402
from wan_va.wm.watermark import generate_keyed_reference as lb_genref  # noqa: E402

J = 32
DATA = Path("/workspace/vla")
SCORE_CSV = DATA / "attack_c_data" / "per_episode_scores"

# Synchronization (lag) search, paper-wide PRODUCTION DEFAULT (LAG_SEARCH=1, LAG_MODE=global).
# The raw matched filter assumes hat-z and r are time-aligned; the delay branch of the controller
# (and residual MAP/windowing offset, and the group delay of heavy EMA smoothing) shifts hat-z, so
# a lag-0 score understates the evidence. We search a small set of lags and keep the peak, applied
# symmetrically to the true key AND the J decoys so the false-key calibration absorbs the
# best-of-(L+1) inflation. Depth L=max(ceil(0.1 H),4): the horizon term caps overlap-loss on long
# chunks, the floor reaches a multi-step delay on short chunks. On non-temporal attacks the
# estimated tau* resolves to 0, so those cells stay byte-identical to lag-0; only the temporal cells
# (delay, heavy EMA) move. Set LAG_SEARCH=0 in the env to restore the lag-0 baseline (per_episode_
# scores_raw_L0 / tab_lagsearch off-column are built that way).
import os  # noqa: E402
LAG_SEARCH = os.environ.get("LAG_SEARCH", "1") == "1"
LAG_FLOOR = 4
# LAG_MODE (default "global"): one deployment tau* per condition, estimated once from the pooled
# watermarked keyed response (estimate_global_tau) and applied to every episode + its decoys by the
# builder (build_raw_perep). Other modes: "max" = symmetric per-key max over tau (honest null, but
# the decoy best-of-L inflation costs clean detection on short horizons); "shared" = one tau* from
# the true key per episode applied to true+all decoys (clean tau*~0 preserved, delay tau*~k
# recovered); "gated" = "shared" but accept tau* only if the true-key gain beats every decoy's.
LAG_MODE = os.environ.get("LAG_MODE", "global")


def lag_set(horizon: int):
    return list(range(0, max(int(np.ceil(horizon * 0.1)), LAG_FLOOR) + 1))

PRESETS = {
    "robotwin": dict(ACTIVE=list(range(0, 7)) + [28] + list(range(7, 14)) + [29], F=2, H=16),
    "libero":   dict(ACTIVE=list(range(0, 7)), F=4, H=4),
}


def raw_score(feature, null_mat):
    feature = np.asarray(feature, float)
    null_mat = np.asarray(null_mat, float)
    if feature.size == 0 or null_mat.size == 0:
        return 0.0
    return float(np.sum(feature - null_mat.mean(axis=0)))


def _finish(variant, tv, nm, wmf):
    st_w = wmf(tv, nm, subspace_rank=3)
    sf_w = [wmf(nm[i], np.delete(nm, i, axis=0), subspace_rank=3) for i in range(len(nm))]
    st_r = raw_score(tv, nm)
    sf_r = [raw_score(nm[i], np.delete(nm, i, axis=0)) for i in range(len(nm))]
    return variant, float(st_w), [float(x) for x in sf_w], float(st_r), [float(x) for x in sf_r]


def _lcfg(key, cf):
    return LCfg(secret_key=key, control_freq=cf, beta=1.0, freq_range=(0.5, 3.0), n_tones=4,
               reference_mode="gaussian", chunk_selection_strategy="stateful_online",
               chunk_selection_period=6, chunk_selection_count=5, chunk_start_min=2)


# --- lag-search feature builders (per-tau tv vectors + nm decoy matrices) --- #
# At tau=0 these reproduce score_openpi / score_lingbot exactly (same windows, decoys, cosine),
# so LAG_SEARCH=0 is a no-op; with LAG_SEARCH=1 the raw score is max_tau over lag_set(H).
def _openpi_lag_feats(d):
    selected = d["chunk_selected"]; rec = d["chunk_recovered_noise"]; ref = d["chunk_reference"]
    executed = d["chunk_executed_steps"]
    chunk_idx = d["chunk_index"] if "chunk_index" in d.files else d["chunk_chunk_index"]
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    action_dim = int(ref.shape[-1]); horizon = int(ref.shape[1]); lags = lag_set(horizon)
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
            a = rec_i[t:horizon]
            tv[t].append(_cosine_sim(a, ref_true[:horizon - t]))
            for j in range(J):
                nm[t][j].append(_cosine_sim(a, drefs[j][:horizon - t]))
    return ({t: np.asarray(tv[t], float) for t in lags}, {t: np.asarray(nm[t], float) for t in lags})


def _lingbot_lag_feats(d, preset):
    cf = float(preset["F"] * preset["H"]); length0 = preset["F"] * preset["H"]
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 42
    wm_idx = np.where(np.asarray(d["chunk_watermarked_flags"], bool))[0]
    map_z = d["map_z"] if "map_z" in d.files else None
    wm_noises = d["chunk_wm_noises"]
    if len(wm_idx) == 0:
        return None
    action_dim = len(preset["ACTIVE"]); lags = lag_set(length0)
    tv = {t: [] for t in lags}; nm = {t: [[] for _ in range(J)] for t in lags}
    for i, ci in enumerate(wm_idx):   # ALL wm chunks (uncapped, matches score_lingbot)
        rec = map_z[i] if (map_z is not None and i < len(map_z)) else wm_noises[ci]
        noise = np.asarray(rec, np.float32)
        while noise.ndim > 3:
            noise = noise[..., 0]
        active = noise[preset["ACTIVE"]].reshape(action_dim, length0)
        ctx = LCtx(chunk_index=int(ci), episode_nonce=nonce)
        ref_true = lb_genref(length=length0, action_dim=action_dim, sample_rate_hz=cf,
                             config=_lcfg(key, cf), context=ctx).T
        drefs = [lb_genref(length=length0, action_dim=action_dim, sample_rate_hz=cf,
                           config=_lcfg(key + 1000 + j, cf), context=ctx).T for j in range(J)]
        for t in lags:
            tv[t].append(np.asarray([_cosine_sim(active[dd][t:length0], ref_true[dd][:length0 - t])
                                     for dd in range(action_dim)], float))
            for j in range(J):
                nm[t][j].append(np.asarray([_cosine_sim(active[dd][t:length0], drefs[j][dd][:length0 - t])
                                            for dd in range(action_dim)], float))
    return ({t: np.concatenate(tv[t]) for t in lags},
            {t: np.stack([np.concatenate(p) for p in nm[t]]) for t in lags})


# --- condition-level GLOBAL tau* (the deployment lag, estimated once per suspect group) --- #
def lag_feats_for(d, pipe, preset=None):
    return _lingbot_lag_feats(d, preset) if pipe == "lingbot" else _openpi_lag_feats(d)


def estimate_global_tau(feats_list):
    # feats_list: [(variant, (tv, nm)), ...]; estimate one tau* by maximizing the aggregate
    # keyed (true-key, decoy-centered) response over the watermarked episodes of the group.
    taus = sorted(feats_list[0][1][0].keys())
    wm = [f for var, f in feats_list if var == "watermarked"]
    pool = wm if wm else [f for _, f in feats_list]
    T = {t: sum(raw_score(tv[t], nm[t]) for (tv, nm) in pool) for t in taus}
    return max(T, key=lambda t: T[t])


def lag_raw_at(feats, tstar):
    tv, nm = feats
    st = raw_score(tv[tstar], nm[tstar])
    sf = [raw_score(nm[tstar][j], np.delete(nm[tstar], j, axis=0)) for j in range(J)]
    return float(st), [float(x) for x in sf]


def _lag_raw(feats):
    tv, nm = feats; taus = list(tv.keys())
    if LAG_MODE in ("shared", "gated"):
        true_by_t = {t: raw_score(tv[t], nm[t]) for t in taus}
        tstar = max(taus, key=lambda t: true_by_t[t])
        if LAG_MODE == "gated":
            # accept tau* only if the true key's lag-search gain exceeds EVERY decoy's gain
            # (a real temporal shift benefits the true key far more than any wrong key; on
            # clean/plain the gains are all noise -> reject -> fall back to tau=0 -> FPR honest).
            true_gain = true_by_t[tstar] - true_by_t[0]
            decoy_gain_max = 0.0
            for j in range(J):
                d0 = raw_score(nm[0][j], np.delete(nm[0], j, axis=0))
                dstar = max(raw_score(nm[t][j], np.delete(nm[t], j, axis=0)) for t in taus)
                decoy_gain_max = max(decoy_gain_max, dstar - d0)
            if true_gain <= decoy_gain_max:
                tstar = 0
        st = raw_score(tv[tstar], nm[tstar])
        sf = [raw_score(nm[tstar][j], np.delete(nm[tstar], j, axis=0)) for j in range(J)]
        return float(st), [float(x) for x in sf]
    # default "max": symmetric per-key max over tau
    st = max(raw_score(tv[t], nm[t]) for t in taus)
    sf = [max(raw_score(nm[t][j], np.delete(nm[t], j, axis=0)) for t in taus) for j in range(J)]
    return float(st), [float(x) for x in sf]


def score_lingbot(npz, preset):
    # allow_pickle: trusted local artifacts written by our own rollout pipeline.
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else (
        "watermarked" if float(d["beta"]) > 0 else "plain")
    cf = float(preset["F"] * preset["H"])
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 42
    wm_idx = np.where(np.asarray(d["chunk_watermarked_flags"], bool))[0]
    map_z = d["map_z"] if "map_z" in d.files else None
    wm_noises = d["chunk_wm_noises"]
    tp, npr = [], [[] for _ in range(J)]
    for i, ci in enumerate(wm_idx):
        rec = map_z[i] if (map_z is not None and i < len(map_z)) else wm_noises[ci]
        ctx = LCtx(chunk_index=int(ci), episode_nonce=nonce)
        vec = lambda kk: build_score_vector_from_noise(
            np.asarray(rec, np.float32), config=_lcfg(kk, cf), context=ctx, sample_rate_hz=cf,
            active_channel_ids=preset["ACTIVE"], frame_chunk_size=preset["F"], action_per_frame=preset["H"])
        tp.append(vec(key))
        for j in range(J):
            npr[j].append(vec(key + 1000 + j))
    if not tp:
        return None
    tv = np.concatenate(tp).astype(np.float64)
    nm = np.stack([np.concatenate(p) for p in npr]).astype(np.float64)
    res = _finish(variant, tv, nm, wmf_score_from_vectors)
    if LAG_SEARCH and LAG_MODE != "global":   # global tau* is condition-level, applied by the builder
        feats = _lingbot_lag_feats(d, preset)
        if feats is not None:
            st_r, sf_r = _lag_raw(feats)
            res = (res[0], res[1], res[2], st_r, sf_r)
    return res


def score_openpi(npz):
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else "watermarked"
    key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    tv = _score_episode(d, score_step_scope="full_chunk", max_windows=5)
    if tv.shape[0] == 0:
        return None
    nm = np.stack([_null_episode(d, off + 1, score_step_scope="full_chunk", max_windows=5,
                                 secret_key=key, sample_rate_hz=srate) for off in range(J)])
    res = _finish(variant, tv, nm, _wmf_score)
    if LAG_SEARCH and LAG_MODE != "global":   # global tau* is condition-level, applied by the builder
        feats = _openpi_lag_feats(d)
        if feats is not None:
            st_r, sf_r = _lag_raw(feats)
            res = (res[0], res[1], res[2], st_r, sf_r)
    return res


def build_cell(cell):
    cache = HERE / f"cache_{cell['key']}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for d in cell["dirs"]:
        for p in sorted((DATA / d).rglob("*.npz")):
            try:
                r = score_lingbot(p, PRESETS[cell["preset"]]) if cell["pipe"] == "lingbot" else score_openpi(p)
            except Exception as e:                       # skip malformed npz, keep going
                print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
            if r is None:
                continue
            variant, st_w, sf_w, st_r, sf_r = r
            rows.append([variant, st_w, st_r] + sf_w + sf_r)
    cols = (["variant", "st_w", "st_r"] + [f"sfw_{i+1}" for i in range(J)] + [f"sfr_{i+1}" for i in range(J)])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(cache, index=False)
    return df


# --- metrics (each block on the axis it governs, |G|=1) -------------------- #
def _zcal(st, sf):
    mu = sf.mean(axis=1); sd = sf.std(axis=1, ddof=1)
    z_true = (st - mu) / (sd + av.EPS)
    csum = sf.sum(axis=1, keepdims=True)
    loo_mu = (csum - sf) / (J - 1)
    sq = (sf ** 2).sum(axis=1, keepdims=True)
    loo_var = ((sq - sf ** 2) - (J - 1) * loo_mu ** 2) / (J - 2)
    z_null = (sf - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + av.EPS)
    return z_true, z_null


def det_auc(st, sf, is_wm):
    z_true, z_null = _zcal(st, sf)
    return av.auc(z_true[is_wm], z_null.ravel())


def ident_r1(st, sf, is_wm):
    return float((st[is_wm] > sf[is_wm].max(axis=1)).mean())


def realized_fpr(st, sf, is_wm, calibrated, nominal=0.01):
    if calibrated:
        z_true, z_null = _zcal(st, sf)
        score, null = z_true, z_null.ravel()
    else:
        score, null = st, sf.ravel()
    thr = np.quantile(null, 1 - nominal)
    return float((score[~is_wm] >= thr).mean())


CELLS = [
    dict(name="LingBot/RoboTwin", key="lingbot_rt", pipe="lingbot", preset="robotwin",
         dirs=["eval_out/lingbot_rt_fix/clean"], csv="lingbot_robotwin_partial_map_clean.csv"),
    dict(name="LingBot/LIBERO-10", key="lingbot_libero", pipe="lingbot", preset="libero",
         dirs=["attack_c_data/rollouts/lingbot_libero/normal", "attack_c_data/rollouts/lingbot_libero/plain"],
         csv="lingbot_libero_10_partial_map_clean.csv"),
    dict(name="pi0.5/LIBERO-10", key="pi05_libero", pipe="openpi",
         dirs=["eval_out/base/libero_10/rollouts/none/task_rollout"],
         csv="pi05_libero_10_partial_map_clean.csv"),
    dict(name="pi0.5/RoboTwin", key="pi05_rt", pipe="openpi",
         dirs=["eval_out/openpi_robotwin_topup"], csv="pi05_robotwin10_partial_map_clean.csv"),
]


def csv_det_auc(cell):
    """Committed-CSV whitened detection AUC (== the table) for the repro gate."""
    p = SCORE_CSV / cell["csv"]
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    cal = av.calibrate(df)
    return av.auc(cal.z_h1, cal.z_null.ravel())


def main():
    print(f"{'cell':20s}{'n(wm/pl)':>12s} | {'scorer':9s}{'detAUC':>8s}{'identR1':>9s}"
          f"{'FPRcal':>8s}{'FPRraw':>8s} | gate(csv whitened)")
    print("-" * 100)
    summary = []
    for cell in CELLS:
        df = build_cell(cell)
        is_wm = (df["variant"].to_numpy() == "watermarked")
        st_w = df["st_w"].to_numpy(float); sf_w = df[[f"sfw_{i+1}" for i in range(J)]].to_numpy(float)
        st_r = df["st_r"].to_numpy(float); sf_r = df[[f"sfr_{i+1}" for i in range(J)]].to_numpy(float)
        nwm, npl = int(is_wm.sum()), int((~is_wm).sum())
        gate = csv_det_auc(cell)
        for tag, st, sf in [("whitened", st_w, sf_w), ("raw", st_r, sf_r)]:
            a = det_auc(st, sf, is_wm); r1 = ident_r1(st, sf, is_wm)
            fc = realized_fpr(st, sf, is_wm, True); fr = realized_fpr(st, sf, is_wm, False)
            g = f"  csv={gate:.3f} d={abs(gate-a):.3f}" if tag == "whitened" else ""
            head = f"{cell['name']:20s}{f'{nwm}/{npl}':>12s}" if tag == "whitened" else " " * 32
            print(f"{head} | {tag:9s}{a:8.3f}{r1:9.3f}{fc:8.3f}{fr:8.3f} |{g}")
            summary.append((cell["name"], tag, a, r1, fc, fr))
        print()
    pd.DataFrame(summary, columns=["cell", "scorer", "detAUC", "identR1", "FPRcal", "FPRraw"]).to_csv(
        HERE / "ablation_allcells_summary.csv", index=False)
    print(f"wrote {HERE/'ablation_allcells_summary.csv'}")


if __name__ == "__main__":
    main()
