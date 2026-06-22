#!/usr/bin/env bash
# BRIDGE: rollout NPZ -> per_episode_scores/*.csv (the input the §6.3/§6.5 analysis reads).
# lingbot exporter: lingbot-va/wan_va/wm/export_lingbot_per_episode_scores.py
# openpi exporter : openpi/.../scripts/attacks/export_per_episode_scores.py
# These need per-run args (--rollout-dir/--out/--attack/--preset). See README for examples.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
echo "This is a bridge step run per rollout dir; see ../README.md '§6.3 score export' for the exact commands." >&2
exit 0
