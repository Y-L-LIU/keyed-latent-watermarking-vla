#!/usr/bin/env bash
# Paper §6.3 Verification + Appendix A -> tab_main.tex, tab_utility.tex,
#   fig_tpr_vs_G.pdf, fig_rate_calibration.pdf, fig_neg_control_h0.pdf, fig_aggregation_mode.pdf,
#   fig_attack_combined.pdf (also §6.4), verification_metrics.csv
# Reads attack_c_data/per_episode_scores/*_partial_map_*.csv (+ utility_*.csv).
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/analyze_verification.py" "$@"
