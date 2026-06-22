#!/usr/bin/env bash
# Paper §5.1 (Imperceptibility) -> fig_spectrum.pdf
# PSD of executed actions: output-sine watermark concentrates in 1-2 Hz; latent
# fingerprint is spectrally flat. Reads clean pi05 + lingbot LIBERO-10 rollouts.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec "$PY" "$VLA/results/make_fig_spectrum.py" "$@"
