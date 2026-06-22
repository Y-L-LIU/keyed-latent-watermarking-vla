#!/usr/bin/env bash
# §6.4 ROLLOUT (openpi): output-attack 'delay' on pi05/LIBERO.
#   args: [SUITE DELAY GPUS NTASKS NTRIALS]  default: libero_10 1 0,1,2,3 10 5
# (clip/ema/jitter for openpi LIBERO were run on the orig node; delay added here.)
# -> attack_c_data/rollouts/openpi_libero/<suite>_delay_<N>/
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_openpi_libero_delay.sh" "$@"
