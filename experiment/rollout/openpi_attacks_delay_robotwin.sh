#!/usr/bin/env bash
# §robustness ROLLOUT (openpi): pi0.5/RoboTwin base-clean + controller DELAY (1/2/3).
#   -> eval_out/openpi_robotwin_delay_s{1,2,3}/  (fills the pi0.5 delay n/a gap in fig_attack_combined)
# Underlying: $VLA/run_openpi_robotwin_delay_8gpu.sh  (env GPUS="0 1" to throttle)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_openpi_robotwin_delay_8gpu.sh" "$@"
