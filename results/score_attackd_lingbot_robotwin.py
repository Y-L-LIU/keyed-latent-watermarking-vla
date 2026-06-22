#!/usr/bin/env python3
"""Score LingBot Attack-D RoboTwin rollouts with the canonical RAW matched filter."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc  # noqa: E402
import fuse_detectors as fd             # noqa: E402

J = asc.J
DATA = asc.DATA
OUT = DATA / "attack_c_data" / "per_episode_scores_attackd_lingbot_robotwin"
if os_out := __import__("os").environ.get("ATTACKD_ROBOTWIN_OUT"):
    OUT = Path(os_out)
HEADER = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
           "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])

ER = __import__("os").environ.get("ATTACKD_ROBOTWIN_ER", "eval_out/lingbot_robotwin_attackd")
CONDS = [
    ("lam0", "0", [f"{ER}/lam0_watermarked", f"{ER}/lam0_plain"]),
    ("tw100", "100", [f"{ER}/tw100_watermarked", f"{ER}/tw100_plain"]),
    ("tw10", "10", [f"{ER}/tw10_watermarked", f"{ER}/tw10_plain"]),
    ("tw1", "1", [f"{ER}/tw1_watermarked", f"{ER}/tw1_plain"]),
    ("tw0", "inf", [f"{ER}/tw0_watermarked", f"{ER}/tw0_plain"]),
]


def build_cell(label, dirs):
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    rows = []
    succ = {"watermarked": [], "plain": []}
    missing_plain_map = 0
    for p in npz:
        try:
            r = asc.score_lingbot(p, asc.PRESETS["robotwin"])
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        variant, _stw, _sfw, st_r, sf_r = r
        d = np.load(p, allow_pickle=True)
        if variant == "plain" and "map_z" not in d.files:
            missing_plain_map += 1
        if "success" in d.files:
            succ.setdefault(variant, []).append(bool(d["success"]))
        task = str(d["task_name"]) if "task_name" in d.files else p.parent.name
        eid = f"lingbot|robotwin|attackd_{label}|partial|map|{task}_{p.stem}_{variant}"
        rows.append([eid, variant, "lingbot", "robotwin", "partial", "0.2333", "map",
                     f"attackd_{label}", "", 1, f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
    if missing_plain_map:
        print(f"   WARNING {label}: {missing_plain_map} plain NPZs have no map_z; scorer fell back to chunk_wm_noises",
              file=sys.stderr)
    sw = float(np.mean(succ["watermarked"])) if succ["watermarked"] else float("nan")
    sp = float(np.mean(succ["plain"])) if succ["plain"] else float("nan")
    return rows, sw, sp, len(succ["watermarked"]), len(succ["plain"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("LingBot/RoboTwin Attack-D RAW matched filter (preset=robotwin)\n")
    hdr = (f"{'cell':10s}{'atk_str':>8s}{'n(wm/pl)':>10s} | {'succ_wm':>8s}{'succ_pl':>8s} | "
           f"{'det@G1':>8s}{'det@G16':>9s}{'DIR1%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    summ = []
    for label, atk_str, dirs in CONDS:
        if not any((DATA / d).exists() and list((DATA / d).rglob("*.npz")) for d in dirs):
            print(f"{label:10s}{atk_str:>8s}  (no rollouts yet)")
            continue
        rows, sw, sp, nw, npl = build_cell(label, dirs)
        if not rows:
            print(f"{label:10s}  (no scored rows)")
            continue
        with open(OUT / f"lingbot_robotwin_attackd_{label}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        df = pd.DataFrame(rows, columns=HEADER)
        for c in ["s_true"] + [f"s_false_{i+1}" for i in range(J)]:
            df[c] = df[c].astype(float)
        is_wm, zt, zd = fd.keyed_z(df)
        mm = fd.all_metrics(zt, zd, is_wm)
        print(f"{label:10s}{atk_str:>8s}{f'{int(is_wm.sum())}/{int((~is_wm).sum())}':>10s} | "
              f"{sw:8.3f}{sp:8.3f} | {mm['d1']:8.3f}{mm['d16']:9.3f}{mm['dir16']:8.3f}")
        summ.append([label, atk_str, nw, npl, sw, sp, mm["d1"], mm["d16"], mm["r1"], mm["r16"], mm["dir16"]])
    if summ:
        sdf = pd.DataFrame(summ, columns=["cell", "atk_strength", "n_wm", "n_pl", "succ_wm", "succ_pl",
                                          "detG1", "detG16", "idR1G1", "idR1G16", "DIR1G16"])
        out = OUT / "_summary.csv"
        sdf.to_csv(out, index=False)
        print(f"\nwrote per-episode CSVs + summary -> {out}")


if __name__ == "__main__":
    main()
