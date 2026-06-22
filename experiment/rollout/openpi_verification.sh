#!/usr/bin/env bash
# §6.3 ROLLOUT (openpi): clean detection rollouts, LIBERO. -> eval_out/<suite>/
# Underlying: openpi/scripts/eval_libero_action_inversion_postprocess_robustness.py
set -euo pipefail; source "$(dirname "$0")/../env.sh"
exec bash "$VLA/run_openpi_eval.sh" "$@"
