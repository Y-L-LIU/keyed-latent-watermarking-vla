#!/usr/bin/env bash
# §6.4 ROLLOUT (openpi §12.5): prune30 + int8 quant on the descendant, base detector.
# Attacked ckpts are built by openpi/scripts/attacks/build_compressed_ckpt.py (see README).
#   arg1 = libero | robotwin   (default libero)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
case "${1:-libero}" in
  libero)   shift || true; exec bash "$VLA/run_openpi_eval_compression.sh" "$@" ;;
  robotwin) shift || true; exec bash "$VLA/run_openpi_robotwin10_eval_compression.sh" "$@" ;;
  *) echo "arg1 must be libero|robotwin"; exit 2 ;;
esac
