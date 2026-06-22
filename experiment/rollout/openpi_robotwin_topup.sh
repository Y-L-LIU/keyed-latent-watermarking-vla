#!/usr/bin/env bash
# §6.3/6.5 ROLLOUT (openpi): pi0.5/RoboTwin base-CLEAN top-up, 4-start MAP, 10 tasks.
#   -> eval_out/openpi_robotwin_topup/  (feeds pi05_robotwin10_partial_map_clean.csv)
# Underlying: $VLA/run_openpi_robotwin_topup_8gpu.sh (8-GPU pull queue)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_openpi_robotwin_topup_8gpu.sh" "$@"
