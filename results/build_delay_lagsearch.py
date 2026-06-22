#!/usr/bin/env python3
"""Lag-search detection for the DELAY attack, across all cells that have delay data.

The delay controller FIFO-shifts the executed action stream by k steps, so the
recovered per-chunk noise is time-shifted by ~k samples relative to the keyed
reference's phase -> the lag-0 matched-filter cosine decorrelates. We add a
synchronization search: score the recovered noise against the reference shifted
over candidate lags tau in {0..4} and take the peak.

Design:
  - PER-EPISODE SHARED lag: the physical delay is constant within an episode, so
    one tau is applied to ALL selected chunks; we take the episode-level argmax.
  - ONE-SIDED {0..4}: delay only shifts the signal later (positive lag).
  - SYMMETRIC null: the identical max-over-tau is applied to the true key AND to
    each of the 32 decoys independently, so the null also rises -- the search only
    helps if the true key has a genuine peak at the delay offset decoys lack.

Alignment at lag tau: recovered[t] ~ intended[t-tau], so compare
    recovered[tau:]  vs  reference[:length-tau]                 (linear, valid overlap).

For each cell+delay strength we emit a baseline (lag0) and a lag-search per-episode
CSV (build_raw_perep schema) and print the |G|=1 detection AUC, after GATING the
lag0 baseline against the committed delay CSV (must reproduce it).

NPZs are our own pipeline-written rollout artifacts (trusted local) -> allow_pickle
matches the existing scorers (ablation_scorer_allcells).
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av            # noqa: E402
import ablation_scorer_allcells as asc       # noqa: E402  (sets up openpi + lingbot paths)
from build_cost_utility_table import _cosine_sim  # noqa: E402
from openpi.policies import watermark as op_wm     # noqa: E402
from wan_va.wm.watermark import generate_keyed_reference as lb_genref  # noqa: E402

J = asc.J
# Lag range. ADAPTIVE (default): L = max(ceil(0.1*horizon), LAG_FLOOR). The horizon term keeps
# the linear valid-overlap loss (~L/horizon) near 10% on long chunks (RoboTwin h=50 -> 5); the
# floor of 4 guarantees the window can still reach a multi-step delay on short chunks (LIBERO
# h=10 would otherwise give L=1, which only aligns delay-1 and collapses on delay-2/3). The
# per-episode overlap cost the floor adds on short horizons (LIBERO clean/compression dropping
# ~0.1 AUC at |G|=1) washes out under |G|=16 aggregation, while the delay-3 cell it rescues does
# not (|G|=16 0.57 -> 0.84). Fixed {0..4} (ADAPTIVE=False) kept for the ablation comparison.
ADAPTIVE = True
LAG_FLOOR = 4
FIXED_LAGS = list(range(0, 5))


def lag_set(horizon: int):
    if not ADAPTIVE:
        return FIXED_LAGS
    L = max(int(np.ceil(horizon * 0.1)), LAG_FLOOR)
    return list(range(0, L + 1))


DATA = Path("/workspace/vla")
RAW = DATA / "attack_c_data" / "per_episode_scores_raw"
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])


# --------------------------------------------------------------------------- #
# Per-episode lag features: tv[tau] (feature vector), nm[tau] (J x feature dim) #
# --------------------------------------------------------------------------- #
def openpi_lag_features(d):
    selected = d["chunk_selected"]; rec = d["chunk_recovered_noise"]; ref = d["chunk_reference"]
    executed = d["chunk_executed_steps"]
    chunk_idx = d["chunk_index"] if "chunk_index" in d.files else d["chunk_chunk_index"]
    nonce = int(d["episode_nonce"])
    key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    action_dim = int(ref.shape[-1]); horizon = int(ref.shape[1])
    lags = lag_set(horizon)
    wins = [i for i in range(len(selected)) if selected[i] and executed[i] > 0][:5]
    if not wins:
        return None
    tv = {t: [] for t in lags}; nm = {t: [[] for _ in range(J)] for t in lags}
    for i in wins:
        rec_i = rec[i]; ref_true = ref[i]
        ctx = op_wm.WatermarkContext(chunk_index=int(chunk_idx[i]), episode_nonce=nonce)
        drefs = []
        for off in range(1, J + 1):
            cfg = op_wm.InternalNoiseWatermarkConfig(secret_key=key + off, control_freq=srate)
            drefs.append(op_wm.generate_keyed_reference(length=horizon, action_dim=action_dim,
                                                       sample_rate_hz=srate, config=cfg, context=ctx))
        for t in lags:
            a = rec_i[t:horizon]
            tv[t].append(_cosine_sim(a, ref_true[:horizon - t]))
            for j in range(J):
                nm[t][j].append(_cosine_sim(a, drefs[j][:horizon - t]))
    return ({t: np.asarray(tv[t], float) for t in lags},
            {t: np.asarray(nm[t], float) for t in lags})


def _lb_active(rec, preset):
    length = preset["F"] * preset["H"]; action_dim = len(preset["ACTIVE"])
    noise = np.asarray(rec, np.float32)
    while noise.ndim > 3:           # [C,F,H,1] (or extra trailing) -> [C,F,H]
        noise = noise[..., 0]
    return noise[preset["ACTIVE"]].reshape(action_dim, length), length, action_dim


def lingbot_lag_features(d, preset):
    variant = str(d["variant"]) if "variant" in d.files else ("watermarked" if float(d["beta"]) > 0 else "plain")
    cf = float(preset["F"] * preset["H"])
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 42
    wm_idx = np.where(np.asarray(d["chunk_watermarked_flags"], bool))[0]
    map_z = d["map_z"] if "map_z" in d.files else None
    wm_noises = d["chunk_wm_noises"]
    wins = list(wm_idx)[:5]
    if not wins:
        return None, variant
    length0 = preset["F"] * preset["H"]
    lags = lag_set(length0)
    tv = {t: [] for t in lags}; nm = {t: [[] for _ in range(J)] for t in lags}
    for i, ci in enumerate(wins):
        rec = map_z[i] if (map_z is not None and i < len(map_z)) else wm_noises[ci]
        active, length, action_dim = _lb_active(rec, preset)   # [action_dim, length]
        ctx = asc.LCtx(chunk_index=int(ci), episode_nonce=nonce)
        ref_true = lb_genref(length=length, action_dim=action_dim, sample_rate_hz=cf,
                             config=asc._lcfg(key, cf), context=ctx).T          # [action_dim, length]
        drefs = [lb_genref(length=length, action_dim=action_dim, sample_rate_hz=cf,
                           config=asc._lcfg(key + 1000 + j, cf), context=ctx).T for j in range(J)]
        for t in lags:
            tv[t].append(np.asarray([_cosine_sim(active[dd][t:length], ref_true[dd][:length - t])
                                     for dd in range(action_dim)], float))
            for j in range(J):
                nm[t][j].append(np.asarray([_cosine_sim(active[dd][t:length], drefs[j][dd][:length - t])
                                            for dd in range(action_dim)], float))
    feats = ({t: np.concatenate(tv[t]) for t in lags},
             {t: np.stack([np.concatenate(p) for p in nm[t]]) for t in lags})
    return feats, variant


# --------------------------------------------------------------------------- #
def episode_scores(feats, lagsearch: bool):
    tv, nm = feats
    taus = list(tv.keys()) if lagsearch else [0]
    s_true = max(asc.raw_score(tv[t], nm[t]) for t in taus)
    s_false = [max(asc.raw_score(nm[t][j], np.delete(nm[t], j, axis=0)) for t in taus) for j in range(J)]
    return float(s_true), [float(x) for x in s_false]


def det_auc(rows):
    if not rows:
        return float("nan"), 0, 0
    var = np.array([r[1] for r in rows]); st = np.array([float(r[10]) for r in rows])
    sf = np.array([[float(x) for x in r[11:11 + J]] for r in rows]); is_wm = var == "watermarked"
    z_true, z_null = asc._zcal(st, sf)
    return av.auc(z_true[is_wm], z_null.ravel()), int(is_wm.sum()), int((~is_wm).sum())


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)


def committed_auc(stem):
    p = RAW / f"{stem}.csv"
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    st = df["s_true"].to_numpy(float); sf = df[[f"s_false_{i+1}" for i in range(J)]].to_numpy(float)
    is_wm = (df["variant"] == "watermarked").to_numpy()
    zt, zn = asc._zcal(st, sf)
    return av.auc(zt[is_wm], zn.ravel())


# (name, pipe, preset, model, dataset, {tag: (dir, committed_stem)})
CELLS = [
    ("pi0.5/RoboTwin", "openpi", None, "pi0.5", "robotwin10", {
        "1": ("attack_c_data/rollouts/openpi_robotwin/delay_1", "pi05_robotwin10_partial_map_delay_1"),
        "2": ("attack_c_data/rollouts/openpi_robotwin/delay_2", "pi05_robotwin10_partial_map_delay_2"),
        "3": ("attack_c_data/rollouts/openpi_robotwin/delay_3", "pi05_robotwin10_partial_map_delay_3")}),
    ("pi0.5/LIBERO", "openpi", None, "pi0.5", "libero_10", {
        "1": ("attack_c_data/rollouts/openpi_libero/libero_10_delay_1", "pi05_libero_10_partial_map_delay_1"),
        "2": ("attack_c_data/rollouts/openpi_libero/libero_10_delay_2", "pi05_libero_10_partial_map_delay_2"),
        "3": ("attack_c_data/rollouts/openpi_libero/libero_10_delay_3", "pi05_libero_10_partial_map_delay_3")}),
    ("LingBot/RoboTwin", "lingbot", "robotwin", "lingbot", "robotwin10", {
        "1": ("eval_out/lingbot_rt_fix/atk/delay_1", "lingbot_robotwin10_partial_map_delay_1"),
        "2": ("eval_out/lingbot_rt_fix/atk/delay_2", "lingbot_robotwin10_partial_map_delay_2"),
        "3": ("eval_out/lingbot_rt_fix/atk/delay_3", "lingbot_robotwin10_partial_map_delay_3")}),
    ("LingBot/LIBERO", "lingbot", "libero", "lingbot", "libero_10", {
        "1": ("attack_c_data/rollouts/lingbot_libero/robust/libero_10/controller_delay_1",
              "lingbot_libero_10_partial_map_delay_1"),
        "2": ("attack_c_data/rollouts/lingbot_libero/robust/libero_10/controller_delay_2",
              "lingbot_libero_10_partial_map_delay_2"),
        "3": ("attack_c_data/rollouts/lingbot_libero/robust/libero_10/controller_delay_3",
              "lingbot_libero_10_partial_map_delay_3")}),
]


def build_cond(cell, cond_dir):
    name, pipe, preset_key, model, dataset, _ = cell
    preset = asc.PRESETS[preset_key] if preset_key else None
    rows_base, rows_lag = [], []
    for p in sorted((DATA / cond_dir).rglob("*.npz")):
        try:
            d = np.load(p, allow_pickle=True)
            if pipe == "openpi":
                feats = openpi_lag_features(d)
                variant = str(d["variant"]) if "variant" in d.files else ("watermarked" if "_wm" in p.name else "plain")
            else:
                feats, variant = lingbot_lag_features(d, preset)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if feats is None:
            continue
        eid = p.stem
        m = len(feats[0][0])
        base = [eid, variant, model, dataset, "", "", "partial_map", "delay", None, m]
        st0, sf0 = episode_scores(feats, False)
        stL, sfL = episode_scores(feats, True)
        rows_base.append(base + [f"{st0:.8f}"] + [f"{x:.8f}" for x in sf0])
        rows_lag.append(base + [f"{stL:.8f}"] + [f"{x:.8f}" for x in sfL])
    return rows_base, rows_lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    print(f"{'cell':18}{'delay':>6}{'nWM/nPL':>10}{'committed':>11}{'base(lag0)':>12}"
          f"{'lagsearch':>11}{'delta':>8}{'gate':>7}")
    print("-" * 84)
    for cell in CELLS:
        name, pipe, preset_key, model, dataset, conds = cell
        for tag, (cond_dir, stem) in conds.items():
            if not (DATA / cond_dir).exists():
                print(f"{name:18}{tag:>6}  (missing {cond_dir})"); continue
            rb, rl = build_cond(cell, cond_dir)
            for r in rb: r[8] = tag
            for r in rl: r[8] = tag
            ab, nwm, npl = det_auc(rb); al, _, _ = det_auc(rl)
            comm = committed_auc(stem)
            gate = "ok" if (np.isnan(comm) or abs(comm - ab) < 0.02) else f"!{comm-ab:+.3f}"
            print(f"{name:18}{tag:>6}{f'{nwm}/{npl}':>10}{comm:>11.3f}{ab:>12.3f}"
                  f"{al:>11.3f}{al-ab:>+8.3f}{gate:>7}")
            if args.write:
                write_csv(RAW / f"{stem}_lagsearch.csv", rl)
        print()


if __name__ == "__main__":
    main()
