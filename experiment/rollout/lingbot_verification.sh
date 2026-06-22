#!/usr/bin/env bash
# §6.3 ROLLOUT (lingbot): clean detection rollouts, LIBERO-10 (wm + plain, 4-GPU split).
# Underlying: lingbot-va/wan_va/wm/eval_libero_watermark.py
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_lingbot_eval_parallel.sh" "$@"
