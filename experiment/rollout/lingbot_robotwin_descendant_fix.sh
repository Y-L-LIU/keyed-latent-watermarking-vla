#!/usr/bin/env bash
# §6.5 ROLLOUT (lingbot) FIX: re-run lingbot/RoboTwin DESCENDANT on the correct robotwin10
# 10-task set with BOTH variants going through MAP (no --skip-map). Fixes the asymmetric
# (plain-no-map_z) + single-task(beat_block_hammer) bug. -> eval_out/lingbot_rt_descendant_fix10/
# Underlying: $VLA/run_lingbot_robotwin_descendant_fix.sh  (env GPUS / TEST_NUM to tune)
# See experiment/LINGBOT_ROBOTWIN_PLAIN_MAP_FIX_PROMPT.md for the full bug writeup.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_lingbot_robotwin_descendant_fix.sh" "$@"
