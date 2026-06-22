#!/usr/bin/env bash
# Paper §7.1 Key Uniqueness -> key_collision_analysis.csv (feeds sec_key_uniqueness.tex)
# 1024 disjoint false keys vs true key on pi05/LIBERO-10 partial+MAP pool.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/make_key_collision_analysis.py" "$@"
