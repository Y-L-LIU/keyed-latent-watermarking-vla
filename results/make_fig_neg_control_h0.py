#!/usr/bin/env python3
"""Render the negative-control H0 paper figure (fig:neg-control-h0, Figure 4),
and print the realized plain-rollout FPR at the 1% false-key threshold per
family (for the caption).

Dedicated make script (like make_fig_budget_calibration.py) so the figure uses the
raw scorer the paper standardized on, with pi0.5/LIBERO read from the current
work-7+global scorer. Writes paper/fig_neg_control_h0.pdf (+ a results/ copy).
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


cache, sources = load_mixed_cache()

av.fig_neg_control_h0(cache, np.random.default_rng(av.RNG_SEED), PAPER / "fig_neg_control_h0.pdf")
shutil.copy(PAPER / "fig_neg_control_h0.pdf", HERE / "fig_neg_control_h0.pdf")

# realized plain-FPR at the false-key 1% threshold, per family (for the caption)
rng = np.random.default_rng(av.RNG_SEED)
print(f"{'family':22s} {'G=1':>7} {'G=16':>7}   (realized plain FPR at 1% false-key threshold)")
for fam in sorted({k[0] for k in cache}):
    cal = cache.get((fam, "clean", None))
    if cal is None or len(cal.z_h0_plain) == 0:
        print(f"{fam:22s}   (no plain pool)"); continue
    vals = {}
    for G in (1, 16):
        _, vals[G] = av.neg_control_operating_fpr(cal, G, rng)
    used = sorted({sources[k] for k in sources if k[0] == fam})
    print(f"{fam:22s} {vals[1]*100:6.2f}% {vals[16]*100:6.2f}%   "
          f"n_plain={len(cal.z_h0_plain)} source={','.join(used)}")
