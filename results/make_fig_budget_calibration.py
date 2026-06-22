#!/usr/bin/env python3
"""Render the merged budget+calibration paper figure (fig:tpr-vs-g).

LEFT = predicted-vs-measured calibration square; RIGHT = 2x2 TPR-vs-|G| grid
(one panel per family). Replaces the old standalone fig_tpr_vs_G.pdf +
fig_rate_calibration.pdf.

Dedicated make script (like make_fig_attack_combined.py) so the figure uses the
raw scorer the paper standardized on, with pi0.5/LIBERO read from the current
work-7+global scorer. This leaves analyze_verification's whitened default alone.
Writes paper/fig_tpr_calibration.pdf (+ a results/ copy).
"""
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402

RAW = HERE.parent / "attack_c_data" / "per_episode_scores_raw"
WORK7 = HERE.parent / "attack_c_data" / "per_episode_scores_work7_global_5050"
WORK7_FAMILY = "pi0.5/libero_10"
PAPER = HERE.parent / "paper"


def load_mixed_cache():
    """Use RAW for the common score set, then replace/add pi0.5/LIBERO from the
    work-7+global score directory used by the polished robustness figures."""
    cache = {}
    sources = {}

    av.SCORE_DIR = RAW
    for c in av.discover():
        key = (c["family"], c["attack"], c["strength"])
        cache[key] = av.calibrate(c["df"])
        sources[key] = "raw"

    av.SCORE_DIR = WORK7
    for c in av.discover():
        if c["family"] != WORK7_FAMILY:
            continue
        key = (c["family"], c["attack"], c["strength"])
        cache[key] = av.calibrate(c["df"])
        sources[key] = "work7_global_5050"

    return cache, sources


rng = np.random.default_rng(av.RNG_SEED)
cache, sources = load_mixed_cache()

rows = av.fig_budget_calibration(cache, rng, PAPER / "fig_tpr_calibration.pdf")
shutil.copy(PAPER / "fig_tpr_calibration.pdf", HERE / "fig_tpr_calibration.pdf")
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
npass = int(((df.predicted >= df.ci_lo) & (df.predicted <= df.ci_hi) | (df.gap.abs() < 0.05)).sum())
print(f"wrote fig_tpr_calibration.pdf to paper/ and results/  "
      f"(calibration {npass}/{len(df)} cells within band/0.05)")
for fam in sorted({k[0] for k in cache}):
    used = sorted({sources[k] for k in sources if k[0] == fam})
    print(f"  {fam}: {', '.join(used)}")
