#!/usr/bin/env bash
# Paper §6.5 Identification -> tab_identification.tex, fig_identification.pdf,
#   fig_identification_robustness.pdf, identification_metrics.csv
# Same per-episode scores as §6.3 + per_episode_scores_descendant/ for weight-level rows.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/analyze_identification.py" "$@"
