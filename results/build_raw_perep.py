#!/usr/bin/env python3
"""Per-episode score CSVs (committed schema) with RAW and WHITENED scores.

raw      = sum(feature - decoy_mean)              [Sigma=I, no whitening]  <- switched scorer
whitened = shipped rank-3 WMF                     [reproduction gate vs committed]

Reuses the gated scoring in ablation_scorer_allcells (score_lingbot/score_openpi,
which return BOTH st_w/sf_w and st_r/sf_r from the same recovered noise), and adds
the committed episode_id so analyze_verification + analyze_identification run
unchanged on the output. Writes per_episode_scores_raw/ (s_true=raw) and
per_episode_scores_whit_repro/ (s_true=whitened). pi0.5/LIBERO attacks are omitted
(npz not retained) -> that robust column stays blank, to be filled later.

Usage: build_raw_perep.py [--probe N]   (probe: only first N npz/condition, print id+gate)
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc  # noqa: E402

J = asc.J
DATA = asc.DATA
RAW = DATA / "attack_c_data" / "per_episode_scores_raw"
WHIT = DATA / "attack_c_data" / "per_episode_scores_whit_repro"
COMMITTED = DATA / "attack_c_data" / "per_episode_scores"
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

META = {
    "lingbot/libero_10":  dict(model="lingbot", dataset="libero_10",  obs_ratio="0.2333", pipe="lingbot", preset="libero"),
    "lingbot/robotwin10": dict(model="lingbot", dataset="robotwin10", obs_ratio="0.5333", pipe="lingbot", preset="robotwin"),
    "pi0.5/libero_10":    dict(model="pi0.5",   dataset="libero_10",  obs_ratio="0.2188", pipe="openpi"),
    "pi0.5/robotwin10":   dict(model="pi0.5",   dataset="robotwin10", obs_ratio="0.4375", pipe="openpi"),
}

R = "attack_c_data/rollouts/lingbot_libero/robust/libero_10"
# plain (beta=0) policy rolled out under each controller attack -> open-set DIR@FAR impostor pool
# for LingBot/LIBERO (the watermarked-only attack dirs had no plain rows -> D was "--"). Same
# run-tag naming as R, separate tree so the eval did not skip on existing watermarked npz.
# Re-rolled into a separate plain-attack tree.
RP = "attack_c_data/rollouts/lingbot_libero/robust_plain/libero_10"
A = "eval_out/lingbot_rt_fix/atk"
# (cell, attack, strength, filestem, [dirs])
CONDS = [
    ("lingbot/libero_10",  "clean",  "",    "lingbot_libero_10_partial_map_clean",
        ["attack_c_data/rollouts/lingbot_libero/normal", "attack_c_data/rollouts/lingbot_libero/plain"]),
    # each attack cell pairs its watermarked dir with the matching robust_plain dir (RP) so the
    # build pools watermarked + plain -> open-set DIR@FAR impostor pool. tau* is estimated from the
    # watermarked episodes only (estimate_global_tau) and applied to both, so plain scored at same tau*.
    ("lingbot/libero_10",  "clip",   "1.0", "lingbot_libero_10_partial_map_clip_1.0",  [f"{R}/controller_clip_1",     f"{RP}/controller_clip_1"]),
    ("lingbot/libero_10",  "ema",    "0.5", "lingbot_libero_10_partial_map_ema_0.5",   [f"{R}/controller_smooth_0.5", f"{RP}/controller_smooth_0.5"]),
    ("lingbot/libero_10",  "jitter", "0.01","lingbot_libero_10_partial_map_jitter_0.01",[f"{R}/controller_jitter_0.01",f"{RP}/controller_jitter_0.01"]),
    ("lingbot/libero_10",  "delay",  "1",   "lingbot_libero_10_partial_map_delay_1",   [f"{R}/controller_delay_1",    f"{RP}/controller_delay_1"]),
    # delay tau=2,3 rolled with osmesa (the egl sweep was unstable past tau=1)
    ("lingbot/libero_10",  "delay",  "2",   "lingbot_libero_10_partial_map_delay_2",   [f"{R}/controller_delay_2",    f"{RP}/controller_delay_2"]),
    ("lingbot/libero_10",  "delay",  "3",   "lingbot_libero_10_partial_map_delay_3",   [f"{R}/controller_delay_3",    f"{RP}/controller_delay_3"]),
    # off-canonical strengths re-rolled on this node (merged sweep) to give the LingBot/LIBERO row a
    # full strength sweep. Note dir naming: clip 2.0 -> controller_clip_2 (drops .0); ema -> smooth.
    ("lingbot/libero_10",  "clip",   "0.5", "lingbot_libero_10_partial_map_clip_0.5",  [f"{R}/controller_clip_0.5",   f"{RP}/controller_clip_0.5"]),
    ("lingbot/libero_10",  "clip",   "2.0", "lingbot_libero_10_partial_map_clip_2.0",  [f"{R}/controller_clip_2",     f"{RP}/controller_clip_2"]),
    ("lingbot/libero_10",  "ema",    "0.3", "lingbot_libero_10_partial_map_ema_0.3",   [f"{R}/controller_smooth_0.3", f"{RP}/controller_smooth_0.3"]),
    ("lingbot/libero_10",  "ema",    "0.7", "lingbot_libero_10_partial_map_ema_0.7",   [f"{R}/controller_smooth_0.7", f"{RP}/controller_smooth_0.7"]),
    ("lingbot/libero_10",  "jitter", "0.005","lingbot_libero_10_partial_map_jitter_0.005",[f"{R}/controller_jitter_0.005",f"{RP}/controller_jitter_0.005"]),
    ("lingbot/libero_10",  "jitter", "0.02","lingbot_libero_10_partial_map_jitter_0.02",[f"{R}/controller_jitter_0.02",f"{RP}/controller_jitter_0.02"]),
    ("lingbot/robotwin10", "clean",  "",    "lingbot_robotwin_partial_map_clean",      ["eval_out/lingbot_rt_fix/clean"]),
    ("lingbot/robotwin10", "clip",   "1.0", "lingbot_robotwin_partial_map_clip_1.0",   [f"{A}/clip_1.0"]),
    ("lingbot/robotwin10", "ema",    "0.5", "lingbot_robotwin_partial_map_ema_0.5",    [f"{A}/ema_0.5"]),
    ("lingbot/robotwin10", "jitter", "0.01","lingbot_robotwin_partial_map_jitter_0.01",[f"{A}/jitter_0.01"]),
    ("lingbot/robotwin10", "delay",  "1",   "lingbot_robotwin_partial_map_delay_1",    [f"{A}/delay_1"]),
    ("lingbot/robotwin10", "delay",  "2",   "lingbot_robotwin_partial_map_delay_2",    [f"{A}/delay_2"]),
    ("lingbot/robotwin10", "delay",  "3",   "lingbot_robotwin_partial_map_delay_3",    [f"{A}/delay_3"]),
    # off-canonical strengths: rollouts already on disk from the 262-job SNR05 grid (120 npz ea),
    # just never scored -> wire them in for the LingBot/RoboTwin strength sweep (zero re-roll).
    ("lingbot/robotwin10", "clip",   "0.5", "lingbot_robotwin_partial_map_clip_0.5",   [f"{A}/clip_0.5"]),
    ("lingbot/robotwin10", "clip",   "2.0", "lingbot_robotwin_partial_map_clip_2.0",   [f"{A}/clip_2.0"]),
    ("lingbot/robotwin10", "ema",    "0.3", "lingbot_robotwin_partial_map_ema_0.3",    [f"{A}/ema_0.3"]),
    ("lingbot/robotwin10", "ema",    "0.7", "lingbot_robotwin_partial_map_ema_0.7",    [f"{A}/ema_0.7"]),
    ("lingbot/robotwin10", "jitter", "0.005","lingbot_robotwin_partial_map_jitter_0.005",[f"{A}/jitter_0.005"]),
    ("lingbot/robotwin10", "jitter", "0.02","lingbot_robotwin_partial_map_jitter_0.02",[f"{A}/jitter_0.02"]),
    ("pi0.5/libero_10",    "clean",  "",    "pi05_libero_10_partial_map_clean",
        ["eval_out/base/libero_10/rollouts/none/task_rollout"]),
    # pi0.5/LIBERO attacks. ema maps to the smooth_0.5 rollout dir.
    ("pi0.5/libero_10",    "clip",   "1.0", "pi05_libero_10_partial_map_clip_1.0",
        ["attack_c_data/rollouts/openpi_libero/libero_10_clip_1.0/rollouts/clip_1.0/task_rollout"]),
    ("pi0.5/libero_10",    "ema",    "0.5", "pi05_libero_10_partial_map_ema_0.5",
        ["attack_c_data/rollouts/openpi_libero/libero_10_smooth_0.5/rollouts/smooth_0.5/task_rollout"]),
    ("pi0.5/libero_10",    "jitter", "0.01","pi05_libero_10_partial_map_jitter_0.01",
        ["attack_c_data/rollouts/openpi_libero/libero_10_jitter_0.01/rollouts/jitter_0.01/task_rollout"]),
    # off-canonical strengths re-rolled to restore the full sweep the original committed data had
    # (clip 0.5/2.0, ema 0.2/0.8, jitter 0.005/0.05/0.1) -> for fig_attack_combined strength curves.
    ("pi0.5/libero_10",    "clip",   "0.5", "pi05_libero_10_partial_map_clip_0.5",
        ["attack_c_data/rollouts/openpi_libero/libero_10_clip_0.5/rollouts/clip_0.5/task_rollout"]),
    ("pi0.5/libero_10",    "clip",   "2.0", "pi05_libero_10_partial_map_clip_2.0",
        ["attack_c_data/rollouts/openpi_libero/libero_10_clip_2.0/rollouts/clip_2.0/task_rollout"]),
    ("pi0.5/libero_10",    "ema",    "0.2", "pi05_libero_10_partial_map_ema_0.2",
        ["attack_c_data/rollouts/openpi_libero/libero_10_smooth_0.2/rollouts/smooth_0.2/task_rollout"]),
    ("pi0.5/libero_10",    "ema",    "0.8", "pi05_libero_10_partial_map_ema_0.8",
        ["attack_c_data/rollouts/openpi_libero/libero_10_smooth_0.8/rollouts/smooth_0.8/task_rollout"]),
    ("pi0.5/libero_10",    "jitter", "0.005","pi05_libero_10_partial_map_jitter_0.005",
        ["attack_c_data/rollouts/openpi_libero/libero_10_jitter_0.005/rollouts/jitter_0.005/task_rollout"]),
    ("pi0.5/libero_10",    "jitter", "0.05","pi05_libero_10_partial_map_jitter_0.05",
        ["attack_c_data/rollouts/openpi_libero/libero_10_jitter_0.05/rollouts/jitter_0.05/task_rollout"]),
    ("pi0.5/libero_10",    "jitter", "0.1", "pi05_libero_10_partial_map_jitter_0.1",
        ["attack_c_data/rollouts/openpi_libero/libero_10_jitter_0.1/rollouts/jitter_0.1/task_rollout"]),
    ("pi0.5/robotwin10",   "clean",  "",    "pi05_robotwin10_partial_map_clean",       ["eval_out/openpi_robotwin_topup"]),
    ("pi0.5/robotwin10",   "clip",   "1.0", "pi05_robotwin10_partial_map_clip_1.0",    ["eval_out/openpi_robotwin_clip_1.0"]),
    ("pi0.5/robotwin10",   "ema",    "0.5", "pi05_robotwin10_partial_map_ema_0.5",     ["eval_out/openpi_robotwin_ema_0.5"]),
    ("pi0.5/robotwin10",   "jitter", "0.01","pi05_robotwin10_partial_map_jitter_0.01", ["eval_out/openpi_robotwin_jitter_0.01"]),
    # --- full pi0.5/RoboTwin strength sweep (for fig:attack-combined) ---
    ("pi0.5/robotwin10",   "clip",   "0.5", "pi05_robotwin10_partial_map_clip_0.5",    ["eval_out/openpi_robotwin_clip_0.5"]),
    ("pi0.5/robotwin10",   "clip",   "2.0", "pi05_robotwin10_partial_map_clip_2.0",    ["eval_out/openpi_robotwin_clip_2.0"]),
    ("pi0.5/robotwin10",   "ema",    "0.2", "pi05_robotwin10_partial_map_ema_0.2",     ["eval_out/openpi_robotwin_ema_0.2"]),
    ("pi0.5/robotwin10",   "ema",    "0.8", "pi05_robotwin10_partial_map_ema_0.8",     ["eval_out/openpi_robotwin_ema_0.8"]),
    ("pi0.5/robotwin10",   "jitter", "0.005","pi05_robotwin10_partial_map_jitter_0.005",["eval_out/openpi_robotwin_jitter_0.005"]),
    ("pi0.5/robotwin10",   "jitter", "0.05","pi05_robotwin10_partial_map_jitter_0.05", ["eval_out/openpi_robotwin_jitter_0.05"]),
    ("pi0.5/robotwin10",   "jitter", "0.1", "pi05_robotwin10_partial_map_jitter_0.1",  ["eval_out/openpi_robotwin_jitter_0.1"]),
    # --- pi0.5 delay sweeps (rollouts survived; never scored into the paper before) ---
    ("pi0.5/robotwin10",   "delay",  "1",   "pi05_robotwin10_partial_map_delay_1",     ["attack_c_data/rollouts/openpi_robotwin/delay_1"]),
    ("pi0.5/robotwin10",   "delay",  "2",   "pi05_robotwin10_partial_map_delay_2",     ["attack_c_data/rollouts/openpi_robotwin/delay_2"]),
    ("pi0.5/robotwin10",   "delay",  "3",   "pi05_robotwin10_partial_map_delay_3",     ["attack_c_data/rollouts/openpi_robotwin/delay_3"]),
    ("pi0.5/libero_10",    "delay",  "1",   "pi05_libero_10_partial_map_delay_1",      ["attack_c_data/rollouts/openpi_libero/libero_10_delay_1"]),
    ("pi0.5/libero_10",    "delay",  "2",   "pi05_libero_10_partial_map_delay_2",      ["attack_c_data/rollouts/openpi_libero/libero_10_delay_2"]),
    ("pi0.5/libero_10",    "delay",  "3",   "pi05_libero_10_partial_map_delay_3",      ["attack_c_data/rollouts/openpi_libero/libero_10_delay_3"]),
    # --- grid-align cells so every family covers the target
    #     grid ema{0.3,0.5,0.7}/jitter{0.005,0.01,0.02,0.05}. pi0.5/libero rows are read
    #     by rescore_work7_global_5050 (the loop guard skips them for the raw scorer). ---
    ("lingbot/libero_10",  "jitter", "0.05","lingbot_libero_10_partial_map_jitter_0.05",[f"{R}/controller_jitter_0.05",f"{RP}/controller_jitter_0.05"]),
    ("lingbot/robotwin10", "jitter", "0.05","lingbot_robotwin_partial_map_jitter_0.05",[f"{A}/jitter_0.05"]),
    ("pi0.5/robotwin10",   "ema",    "0.3", "pi05_robotwin10_partial_map_ema_0.3",     ["eval_out/openpi_robotwin_ema_0.3"]),
    ("pi0.5/robotwin10",   "ema",    "0.7", "pi05_robotwin10_partial_map_ema_0.7",     ["eval_out/openpi_robotwin_ema_0.7"]),
    ("pi0.5/robotwin10",   "jitter", "0.02","pi05_robotwin10_partial_map_jitter_0.02", ["eval_out/openpi_robotwin_jitter_0.02"]),
    ("pi0.5/libero_10",    "ema",    "0.3", "pi05_libero_10_partial_map_ema_0.3",
        ["attack_c_data/rollouts/openpi_libero/libero_10_smooth_0.3/rollouts/smooth_0.3/task_rollout"]),
    ("pi0.5/libero_10",    "ema",    "0.7", "pi05_libero_10_partial_map_ema_0.7",
        ["attack_c_data/rollouts/openpi_libero/libero_10_smooth_0.7/rollouts/smooth_0.7/task_rollout"]),
    ("pi0.5/libero_10",    "jitter", "0.02","pi05_libero_10_partial_map_jitter_0.02",
        ["attack_c_data/rollouts/openpi_libero/libero_10_jitter_0.02/rollouts/jitter_0.02/task_rollout"]),
]


def episode_id(cell, attack, p, variant):
    m = META[cell]
    if m["pipe"] == "lingbot":
        return f"lingbot|{m['dataset']}|{attack}|partial|map|{p.parent.name}_{p.stem}_{variant}"
    return f"pi0.5|{m['dataset']}|{attack}|partial|map|{p.stem}"


def score_one(cell, p):
    m = META[cell]
    if m["pipe"] == "lingbot":
        return asc.score_lingbot(p, asc.PRESETS[m["preset"]])
    return asc.score_openpi(p)


def build(cond, probe=0):
    cell, attack, strength, stem, dirs = cond
    m = META[cell]
    rows_raw, rows_whit = [], []
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    if probe:
        npz = npz[:probe]
    if asc.LAG_SEARCH and asc.LAG_MODE == "global":
        # two-pass: estimate ONE deployment lag tau* from the pooled keyed response,
        # then score every episode (and its decoys) at that single tau*. score_one gives the
        # L=0 whitened gate column + variant; the raw column is overridden at tau*.
        preset = asc.PRESETS[m["preset"]] if m["pipe"] == "lingbot" else None
        recs = []
        for p in npz:
            try:
                r = score_one(cell, p)
                if r is None:
                    continue
                d = np.load(p, allow_pickle=True)
                f = asc.lag_feats_for(d, m["pipe"], preset)
            except Exception as e:
                print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
            if f is None:
                continue
            recs.append((p, r[0], r[1], r[2], f))   # path, variant, st_w, sf_w, feats
        if not recs:
            return stem, rows_raw, rows_whit
        tstar = asc.estimate_global_tau([(var, f) for (_, var, _, _, f) in recs])
        for (p, variant, st_w, sf_w, f) in recs:
            st_r, sf_r = asc.lag_raw_at(f, tstar)
            eid = episode_id(cell, attack, p, variant)
            base = [eid, variant, m["model"], m["dataset"], "partial", m["obs_ratio"], "map", attack, strength, 1]
            rows_raw.append(base + [f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
            rows_whit.append(base + [f"{st_w:.8f}"] + [f"{x:.8f}" for x in sf_w])
        print(f"   [{stem}] global tau*={tstar} (n={len(recs)})", file=sys.stderr)
        return stem, rows_raw, rows_whit
    for p in npz:
        try:
            r = score_one(cell, p)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, st_w, sf_w, st_r, sf_r = r
        eid = episode_id(cell, attack, p, variant)
        base = [eid, variant, m["model"], m["dataset"], "partial", m["obs_ratio"], "map", attack, strength, 1]
        rows_raw.append(base + [f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
        rows_whit.append(base + [f"{st_w:.8f}"] + [f"{x:.8f}" for x in sf_w])
    return stem, rows_raw, rows_whit


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)


def gate(stem, rows_whit):
    """Join my whitened s_true onto committed on episode_id; report max |diff|."""
    cpath = COMMITTED / f"{stem}.csv"
    if not cpath.exists():
        return f"(no committed CSV: {stem})"
    cdf = pd.read_csv(cpath)[["episode_id", "s_true"]].rename(columns={"s_true": "s_committed"})
    mine = pd.DataFrame([(r[0], float(r[10])) for r in rows_whit], columns=["episode_id", "s_mine"])
    j = cdf.merge(mine, on="episode_id", how="inner")
    if len(j) == 0:
        return f"NO ID MATCH (committed n={len(cdf)}, mine n={len(mine)}) ex mine={mine['episode_id'].iloc[0] if len(mine) else '-'}"
    md = float((j["s_committed"] - j["s_mine"]).abs().max())
    return f"matched {len(j)}/{len(cdf)} ids, max|whit-committed|={md:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip conditions whose RAW CSV already exists (idempotent top-up)")
    args = ap.parse_args()
    if not args.probe:
        RAW.mkdir(exist_ok=True); WHIT.mkdir(exist_ok=True)
    for cond in CONDS:
        # pi0.5/LIBERO is detected with work-7+global now; its canonical per-episode
        # scores live in per_episode_scores_work7_global_5050/ (rescore_work7_global_5050.py).
        # The raw all-32 scores for this family were superseded by the work-7+global detector,
        # so do NOT regenerate them here (avoids picking up the weaker detector by accident).
        if cond[0] == "pi0.5/libero_10":
            continue
        stem_only = cond[3]
        if args.skip_existing and (RAW / f"{stem_only}.csv").exists():
            print(f"{stem_only:46s} SKIP (exists)")
            continue
        stem, rr, rw = build(cond, probe=args.probe)
        g = gate(stem, rw)
        print(f"{stem:46s} n={len(rr):4d}  GATE: {g}")
        if args.probe and rw:
            print(f"      id[0]={rw[0][0]}")
        if not args.probe:
            write_csv(RAW / f"{stem}.csv", rr)
            write_csv(WHIT / f"{stem}.csv", rw)
    print("DONE" + ("" if args.probe else f" -> {RAW}"))


if __name__ == "__main__":
    main()
