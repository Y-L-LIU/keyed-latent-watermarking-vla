#!/usr/bin/env bash
# Paper §5.1 (Imperceptibility) -> fig_bandstop_sweep.pdf
# Sweeps a 1.4 Hz band-stop notch across the spectrum: kills the output-sine
# score only on the 1-2 Hz band. (Latent curve comes from make_latent_bandstop_sweep.py.)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/make_fig_bandstop_sweep.py" "$@"
