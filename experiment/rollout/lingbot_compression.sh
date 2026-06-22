#!/usr/bin/env bash
# §6.4 ROLLOUT (lingbot §12.5): prune + quant eval.
#   arg1 = libero | robotwin   (default libero)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
case "${1:-libero}" in
  libero)   shift || true; exec bash "$VLA/run_lingbot_eval_compression.sh" "$@" ;;
  robotwin) shift || true; exec bash "$VLA/run_lingbot_robotwin10_eval_compression.sh" "$@" ;;
  *) echo "arg1 must be libero|robotwin"; exit 2 ;;
esac
