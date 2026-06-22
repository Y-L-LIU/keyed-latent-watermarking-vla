#!/usr/bin/env bash
# Paper §5.1 (Imperceptibility) -> latent_bandstop_sweep.csv (latent curve in fig_bandstop_sweep)
# GPU: re-runs partial+MAP recovery on band-stopped pi05 rollouts. Needs JAX/openpi env.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/make_latent_bandstop_sweep.py" "$@"
