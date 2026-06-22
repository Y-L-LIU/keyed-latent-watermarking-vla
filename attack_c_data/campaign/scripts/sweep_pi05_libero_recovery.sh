#!/bin/bash
# Phase-1 recovery study — pi05_libero BASE model, watermarked + plain rollouts on
# libero_goal + libero_spatial. Each rollout npz already saves chunk_recovered_noise
# (full+ODE) and chunk_map_restart_recovered_noise (partial+MAP); the other 2 cells
# (full+MAP, partial+ODE) we compute in a post-processing script.
#
# 8-GPU dispatch: 1 process per (suite, task), fan out across 5 GPUs (1/2/3/5/7)
# to avoid the historical EGL/SIGABRT issue on GPU 0/4/6 (mujoco-only — sapien is OK).
# 10 tasks × 2 suites = 20 jobs at 50 ep each. Each job ~30-60 min wall on 1 GPU.
set -uo pipefail
PY=/workspace/vla/openpi/.venv/bin/python
SCRIPT=/workspace/vla/openpi/scripts/eval_libero_action_inversion_postprocess_robustness.py
OPENPI=/workspace/vla/openpi
LIBERO=$OPENPI/third_party/libero
CFG=pi05_libero
CKPT=/workspace/vla/models/pi05_libero
OUT=/workspace/vla/eval_out/base
LOGDIR=/root/campaign_logs/phase1_pi05_libero
mkdir -p "$OUT" "$LOGDIR"

NUM_TRIALS=${NUM_TRIALS:-5}    # 50 ep / task -> 5 trials, repeated across 10 tasks
GPUS=(${GPUS:-1 2 3 5 7})       # mujoco-EGL-safe GPUs only
SUITES=(${SUITES:-libero_10})    # default to libero_10 to match lingbot side; override with `SUITES="libero_goal" ...`

# Build job list: (suite, task_offset)
JOBS=()
for suite in "${SUITES[@]}"; do
  for t in 0 1 2 3 4 5 6 7 8 9; do
    JOBS+=("$suite:$t")
  done
done
echo "[$(date +%T)] pi05 recovery sweep: ${#JOBS[@]} jobs (${#SUITES[@]} suites × 10 tasks), trials/task=$NUM_TRIALS, GPUs=${GPUS[*]}"

declare -A PIDS
launch() {  # $1=gpu $2=jobspec
  local gpu=$1 spec=$2
  IFS=':' read -r suite task <<<"$spec"
  local rolloutdir="$OUT/$suite/rollouts"
  local reportdir="$OUT/$suite/reports"
  mkdir -p "$rolloutdir" "$reportdir"
  local log="$LOGDIR/${suite}_task${task}.log"
  echo "[$(date +%T)] GPU$gpu <- $suite/task$task"
  cd "$OPENPI"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
    PYTHONPATH=$OPENPI:$LIBERO XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=disabled \
    $PY "$SCRIPT" \
      --controller-postprocess none --run-tag none \
      --config-name "$CFG" --checkpoint-dir "$CKPT" \
      --detector-config-name "$CFG" --detector-checkpoint-dir "$CKPT" \
      --task-suite-name "$suite" \
      --task-offset "$task" --num-tasks 1 --num-trials-per-task "$NUM_TRIALS" \
      --detector wmf --reference-mode gaussian --beta 1.0 \
      --null-decoy-count 32 --subspace-rank 3 \
      --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
      --score-step-scope full_chunk --num-inversion-steps 10 \
      --save-all-inversion-modes \
      --latent-map-iters 30 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 2 --obs-sigma 1e-4 \
      --save-rollout-dir "$rolloutdir" --save-report-dir "$reportdir" \
      > "$log" 2>&1 &
  PIDS[$gpu]=$!
}

idx=0
for g in "${GPUS[@]}"; do
  [ $idx -lt ${#JOBS[@]} ] && { launch "$g" "${JOBS[$idx]}"; idx=$((idx+1)); }
done
while [ $idx -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    pid=${PIDS[$g]:-}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      unset PIDS[$g]
      if [ $idx -lt ${#JOBS[@]} ]; then
        launch "$g" "${JOBS[$idx]}"; idx=$((idx+1))
      fi
    fi
  done
  sleep 20
done
echo "[$(date +%T)] DONE"
