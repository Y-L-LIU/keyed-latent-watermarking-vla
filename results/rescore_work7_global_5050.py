#!/usr/bin/env python3
"""Canonical re-score: ALL pi0.5/LIBERO attack cells with the winning detector
(work-7 detect 0-6 + GLOBAL lag-search), standardized to 50 wm + 50 plain each.

Per cell: subsample a balanced 50/50 (5 wm + 5 plain per task x 10 tasks,
deterministic by episode index), compute the dim-restricted (0..6) per-window
cosines at every lag, estimate ONE deployment tau* from the 50 watermarked
episodes' pooled keyed response (asc.estimate_global_tau), and score every
episode + its 32 decoys at that tau* (asc.lag_raw_at). Writes per-episode CSVs
in the committed schema to per_episode_scores_work7_global_5050/ (consumable by
analyze_verification / analyze_identification) and prints the metrics table.

Pure offline: reads the stored inject-all-32 rollout npz; no rollouts run.
"""
from __future__ import annotations
import sys, csv, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc                  # noqa: E402
import build_raw_perep as bp                             # noqa: E402
import fuse_detectors as fd                              # noqa: E402
from build_work7_lagsearch import _lag_feats_dims        # noqa: E402

asc.LAG_SEARCH = True
asc.LAG_MODE = "global"
J = asc.J
DATA = asc.DATA
DIMS = list(range(0, 7))
N_PER_TASK = 5                                            # 5 wm + 5 plain per task * 10 tasks = 50/50
OUT = DATA / "attack_c_data" / "per_episode_scores_work7_global_5050"
NAME_RE = re.compile(r"task_(\d+)_episode_(\d+)_(watermarked|plain)\.npz$")
CONDS = [c for c in bp.CONDS if c[0] == "pi0.5/libero_10"]


def balanced_5050(dirs):
    """Return [(variant, path)] with 5 wm + 5 plain per task (deterministic by
    episode index). Skips *_extra_modes.npz (no matching task/ep/variant)."""
    buckets: dict[tuple[str, int], list[tuple[int, Path]]] = {}
    for d in dirs:
        for p in (DATA / d).rglob("*.npz"):
            m = NAME_RE.search(p.name)
            if not m:
                continue
            task, ep, variant = int(m.group(1)), int(m.group(2)), m.group(3)
            buckets.setdefault((variant, task), []).append((ep, p))
    chosen = []
    short = []
    tasks = sorted({t for (_, t) in buckets})
    for variant in ("watermarked", "plain"):
        for t in tasks:
            eps = sorted(buckets.get((variant, t), []))[:N_PER_TASK]
            if len(eps) < N_PER_TASK:
                short.append((variant, t, len(eps)))
            chosen += [(variant, p) for _, p in eps]
    return chosen, short


def build_cell(cond):
    cell, attack, strength, stem, dirs = cond
    m = bp.META[cell]
    chosen, short = balanced_5050(dirs)
    recs = []                                             # (variant, feats, path)
    for variant, p in chosen:
        try:
            d = np.load(p, allow_pickle=True)             # trusted local rollout artifact
            f = _lag_feats_dims(d, DIMS)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if f is None:
            continue
        recs.append((variant, f, p))
    if not recs:
        return stem, [], None, short
    tstar = asc.estimate_global_tau([(v, f) for v, f, _ in recs])
    rows = []
    for variant, f, p in recs:
        st, sf = asc.lag_raw_at(f, tstar)
        eid = bp.episode_id(cell, attack, p, variant)
        base = [eid, variant, m["model"], m["dataset"], "partial", m["obs_ratio"], "map", attack, strength, 1]
        rows.append(base + [f"{st:.8f}"] + [f"{x:.8f}" for x in sf])
    return stem, rows, tstar, short


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(bp.HEADER); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    print("pi0.5/LIBERO  work-7 (detect 0-6) + GLOBAL lag-search  @ 50wm/50pl per cell\n")
    hdr = (f"{'attack':14s}{'n(wm/pl)':>10s}{'tau*':>5s} | "
           f"{'det@G1':>8s}{'det@G16':>9s} | {'idR1@G1':>9s}{'idR1@G16':>10s} | {'DIR1%@G16':>11s}")
    print(hdr); print("-" * len(hdr))
    summary = []
    for cond in CONDS:
        stem, rows, tstar, short = build_cell(cond)
        lbl = f"{cond[1]} {cond[2]}".strip()
        if not rows:
            print(f"{lbl:14s}  (no rows)"); continue
        write_csv(OUT / f"{stem}.csv", rows)
        df = pd.DataFrame(rows, columns=bp.HEADER)
        for c in ["s_true"] + [f"s_false_{i+1}" for i in range(J)]:
            df[c] = df[c].astype(float)
        is_wm, zt, zd = fd.keyed_z(df)
        mm = fd.all_metrics(zt, zd, is_wm)
        nwm = int(is_wm.sum()); npl = int((~is_wm).sum())
        warn = f"  !short:{short}" if short else ""
        print(f"{lbl:14s}{f'{nwm}/{npl}':>10s}{tstar:5d} | "
              f"{mm['d1']:8.3f}{mm['d16']:9.3f} | {mm['r1']:9.3f}{mm['r16']:10.3f} | {mm['dir16']:11.3f}{warn}")
        summary.append([cond[1], cond[2], nwm, npl, tstar, mm['d1'], mm['d16'], mm['r1'], mm['r16'], mm['dir16']])
    sdf = pd.DataFrame(summary, columns=["attack", "strength", "n_wm", "n_pl", "tau_star",
                                         "detG1", "detG16", "idR1G1", "idR1G16", "DIR1G16"])
    sdf.to_csv(OUT / "_summary.csv", index=False)
    print(f"\nwrote {len(summary)} cells -> {OUT}\n  summary -> {OUT/'_summary.csv'}")


if __name__ == "__main__":
    main()
