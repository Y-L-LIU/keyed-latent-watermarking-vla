"""Re-score the re-rolled under-saturated pi0.5/LIBERO-10 attack cells at the new
(larger) episode count, using build_raw_perep's per-condition scorer.

Run TWICE with different env (asc reads LAG_SEARCH/LAG_MODE at import):

  # global tau* (production default) -> per_episode_scores_raw, all 8 cells
  LAG_SEARCH=1 LAG_MODE=global OUT_DIR=.../per_episode_scores_raw \
      STEMS=delay_1,delay_2,delay_3,ema_0.5,ema_0.2,jitter_0.1,jitter_0.01,clip_0.5 \
      python results/rescore_lag_cells.py

  # lag-zero baseline -> per_episode_scores_raw_L0, delay only (tab_lagsearch off-column)
  LAG_SEARCH=0 OUT_DIR=.../per_episode_scores_raw_L0 \
      STEMS=delay_1,delay_2,delay_3 python results/rescore_lag_cells.py

STEMS are matched as suffixes of the full CSV stem (pi05_libero_10_partial_map_<X>),
so `delay_1` selects pi05_libero_10_partial_map_delay_1. Existing CSVs are backed
up to <name>.bak before overwrite. Prints n and the global tau* per cell.
"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path

sys.path.insert(0, "results")
import build_raw_perep as brp  # noqa: E402  (imports asc, honoring LAG_SEARCH/LAG_MODE env)

OUT = Path(os.environ["OUT_DIR"])
SUFFIXES = [s for s in os.environ.get("STEMS", "").split(",") if s]
PREFIX = "pi05_libero_10_partial_map_"


def selected(stem):
    return any(stem == PREFIX + suf for suf in SUFFIXES)


def main():
    import build_raw_perep as _b  # local alias for clarity
    import ablation_scorer_allcells as asc
    print(f"OUT_DIR={OUT}  LAG_SEARCH={asc.LAG_SEARCH}  LAG_MODE={asc.LAG_MODE}")
    print(f"selecting suffixes: {SUFFIXES}")
    OUT.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for cond in brp.CONDS:
        stem = cond[3]
        if not selected(stem):
            continue
        s, rr, rw = brp.build(cond)
        dst = OUT / f"{s}.csv"
        if dst.exists():
            shutil.copy2(dst, dst.with_suffix(".csv.bak"))
        brp.write_csv(dst, rr)
        n_wm = sum(1 for r in rr if r[1] == "watermarked")
        print(f"  {s:46s} n={len(rr):4d}  n_wm={n_wm:4d}  -> {dst.name}")
        n_done += 1
    print(f"DONE: {n_done} cell(s) re-scored into {OUT}")
    if n_done != len(SUFFIXES):
        print(f"WARNING: matched {n_done} but expected {len(SUFFIXES)} suffixes",
              file=sys.stderr)


if __name__ == "__main__":
    main()
