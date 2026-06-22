#!/usr/bin/env python3
"""Build tab_main + tab_identification from a given per-episode score dir.

No PDF writes, no PAPER_DIR writes -- emits only <out_prefix>_tab_main.tex and
<out_prefix>_tab_identification.tex so collaborator-owned figures are untouched.
Used to (a) gate the whitened re-score against the committed tables, and
(b) produce the raw tables.

Usage: run_tables.py <score_dir> <out_prefix> [<desc_dir>]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av
import analyze_identification as ai

score_dir = Path(sys.argv[1])
out_prefix = sys.argv[2]
desc_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ai.DESC_DIR

# ---- tab_main (verification) ----------------------------------------------- #
av.SCORE_DIR = score_dir
rng = np.random.default_rng(av.RNG_SEED)
rng_auc = np.random.default_rng(av.RNG_SEED + 1)
conds = av.discover()
util = av.load_utility()
rows = []
for c in conds:
    cal = av.calibrate(c["df"])
    rows.append(dict(
        family=c["family"], model=c["model"], dataset=c["dataset"],
        attack=c["attack"], strength=c["strength"], obs_ratio=c["obs_ratio"], n_h1=len(cal.z_h1),
        auc=av.auc(cal.z_h1, cal.z_null.ravel()),
        tpr1_g1=av.tpr_point(cal.z_h1, cal.z_null, 1, av.FPR_MAIN, rng),
        tpr1_gT=av.tpr_point(cal.z_h1, cal.z_null, av.G_TABLE, av.FPR_MAIN, rng),
        auc_gT=av.auc_group(cal.z_h1, cal.z_null, av.G_TABLE, rng_auc),
    ))
M = pd.DataFrame(rows)
av.write_main_table(M, util, Path(f"{out_prefix}_tab_main.tex"))
print("=== verification metrics ===")
print(M[["family", "attack", "strength", "n_h1", "auc", "auc_gT", "tpr1_gT"]].to_string(index=False))

# ---- tab_identification ----------------------------------------------------- #
ai.SCORE_DIR = score_dir
ai.DESC_DIR = desc_dir
rng2 = np.random.default_rng(ai.RNG_SEED)
conds2 = ai.discover()
rows2 = []
for c in conds2:
    cal = ai.calibrate(c["df"])
    for G in ai.G_GRID:
        cmc, _, _, _ = ai.cmc_curve(cal, G, rng2)
        dirs = ai.dir_at_far(cal, G, ai.FAR_TABLE, rng2)
        rows2.append(dict(
            family=c["family"], model=c["model"], dataset=c["dataset"], attack=c["attack"],
            strength=c["strength"], group=c["group"], n_wm=c["n_wm"], n_pl=c["n_pl"],
            gallery=ai.GALLERY, G=G, rank1=cmc[0], rank5=cmc[4],
            dir_far01=dirs[0.01], dir_far10=dirs[0.10],
        ))
M2 = pd.DataFrame(rows2)
ai.write_identification_table(M2, Path(f"{out_prefix}_tab_identification.tex"))
print(f"\nwrote {out_prefix}_tab_main.tex + {out_prefix}_tab_identification.tex")
