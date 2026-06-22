#!/usr/bin/env bash
# Paper §7.2 Unforgeability -> unforgeability_analysis.csv (feeds sec_unforgeability.tex)
# Brute-force forgery budget vs operating FPR; hill-climb attack baseline.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/make_unforgeability_analysis.py" "$@"
