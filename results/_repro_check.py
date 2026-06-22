#!/usr/bin/env python3
"""Does analyze_verification reproduce the committed whitened tab:main?

Drives only the metric build + write_main_table (no figure writes, so paper/
fig_attack_combined.pdf is untouched). Prints the per-condition metrics it finds
and writes tab_main to a temp path for diffing against paper/tab_main.tex.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av

rng = np.random.default_rng(av.RNG_SEED)
rng_auc = np.random.default_rng(av.RNG_SEED + 1)
conds = av.discover()
util = av.load_utility()
print(f"discovered {len(conds)} conditions:")
rows = []
for c in conds:
    cal = av.calibrate(c["df"])
    rows.append(dict(
        family=c["family"], model=c["model"], dataset=c["dataset"],
        attack=c["attack"], strength=c["strength"], obs_ratio=c["obs_ratio"],
        n_h1=len(cal.z_h1),
        auc=av.auc(cal.z_h1, cal.z_null.ravel()),
        tpr1_g1=av.tpr_point(cal.z_h1, cal.z_null, 1, av.FPR_MAIN, rng),
        tpr1_gT=av.tpr_point(cal.z_h1, cal.z_null, av.G_TABLE, av.FPR_MAIN, rng),
        auc_gT=av.auc_group(cal.z_h1, cal.z_null, av.G_TABLE, rng_auc),
    ))
M = pd.DataFrame(rows)
print(M[["family", "attack", "strength", "n_h1", "auc", "auc_gT", "tpr1_gT"]].to_string(index=False))
out = HERE / "_repro_tab_main.tex"
av.write_main_table(M, util, out)
print(f"\nwrote {out}")
