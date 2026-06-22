#!/usr/bin/env bash
# §6.4 ROLLOUT (lingbot): output-attack sweep (clip/ema/jitter/delay).
#   arg1 = libero | robotwin   (default libero)
# These campaign launchers carry hardcoded paths -> remapped on the fly.
set -euo pipefail; source "$(dirname "$0")/../env.sh"
case "${1:-libero}" in
  libero)   remap_and_run "$VLA/attack_c_data/campaign/scripts/sweep_lingbot_libero.sh" ;;
  robotwin) remap_and_run "$VLA/attack_c_data/campaign/scripts/sweep_lingbot_robotwin.sh" ;;
  *) echo "arg1 must be libero|robotwin"; exit 2 ;;
esac
