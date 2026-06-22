"""Export utility metrics (task success rate + mean steps) per config.

Plain vs fingerprinted (watermarked), straight from the rollout NPZs' `success`
and step fields. One CSV row per (model, dataset, attack, attack_strength).
GPU-free; complements the per-episode score CSVs.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np


def _steps(d):
    for k in ("steps", "total_steps"):
        if k in d.files:
            return int(d[k])
    return int(d["executed_actions"].shape[0]) if "executed_actions" in d.files else -1


def summarize(rollout_dir: pathlib.Path):
    rows = {"plain": {"succ": [], "steps": []}, "watermarked": {"succ": [], "steps": []}}
    for npz in sorted(rollout_dir.rglob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        variant = str(d["variant"]) if "variant" in d.files else (
            "watermarked" if ("beta" in d.files and float(d["beta"]) > 0) else "plain")
        if variant not in rows:
            continue
        rows[variant]["succ"].append(bool(d["success"]) if "success" in d.files else False)
        rows[variant]["steps"].append(_steps(d))
    out = {}
    for v in ("plain", "watermarked"):
        s = rows[v]["succ"]; st = [x for x in rows[v]["steps"] if x >= 0]
        out[v] = {
            "n": len(s),
            "sr": float(np.mean(s)) if s else float("nan"),
            "mean_steps": float(np.mean(st)) if st else float("nan"),
            "mean_steps_success": float(np.mean([t for t, ok in zip(rows[v]["steps"], s) if ok]))
                                  if any(s) else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # each --config: model dataset attack strength rollout_dir
    ap.add_argument("--config", action="append", nargs=5, metavar=
                    ("MODEL", "DATASET", "ATTACK", "STRENGTH", "DIR"), required=True)
    args = ap.parse_args()

    header = ["model", "dataset", "attack", "attack_strength",
              "n_plain", "n_wm", "sr_plain", "sr_wm",
              "mean_steps_plain", "mean_steps_wm",
              "mean_steps_success_plain", "mean_steps_success_wm"]
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for model, dataset, attack, strength, d in args.config:
            rd = pathlib.Path(d)
            if not rd.exists():
                print(f"[utility] SKIP missing {rd}", file=sys.stderr); continue
            s = summarize(rd)
            w.writerow([model, dataset, attack, strength,
                        s["plain"]["n"], s["watermarked"]["n"],
                        f"{s['plain']['sr']:.4f}", f"{s['watermarked']['sr']:.4f}",
                        f"{s['plain']['mean_steps']:.2f}", f"{s['watermarked']['mean_steps']:.2f}",
                        f"{s['plain']['mean_steps_success']:.2f}", f"{s['watermarked']['mean_steps_success']:.2f}"])
            print(f"[utility] {model}/{dataset}/{attack}{strength}: "
                  f"SR pl={s['plain']['sr']:.2f} wm={s['watermarked']['sr']:.2f}", file=sys.stderr)
    print(f"[utility] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
