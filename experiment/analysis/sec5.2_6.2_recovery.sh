#!/usr/bin/env bash
# Paper §5.2 / §6.2 (MAP vs ODE recovery) -> feeds tab_recovery.tex (hand-tabulated)
# Computes per-chunk recovery error + separation AUC for the {full,partial}x{ODE,MAP}
# grid from pi05/LIBERO rollouts. tab_pad_ablation comes from the pad-value sweep.
#   arg1 = rollout dir (default: clean pi05 LIBERO-10)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
ROLL="${1:-$VLA/eval_out/base/libero_10/rollouts/none/task_rollout}"
exec "$PY" "$VLA/attack_c_data/campaign/scripts/compute_recovery_metrics.py" \
  --rollout-dir "$ROLL" --out-json "$VLA/results/recovery_metrics.json"
