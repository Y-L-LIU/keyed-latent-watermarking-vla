#!/usr/bin/env bash
# pi0.5/RoboTwin base-CLEAN + output-controller DELAY attack — fills the pi0.5 delay
# n/a gap in fig_attack_combined. Same base policy+detector and MAP config as the clean
# top-up, wrapped by eval_robotwin_watermark_map_robustness.py --controller-postprocess delay.
# Delay strengths 1/2/3 (integer step shifts), matching the lingbot delay_N convention.
# 8 workers (one per GPU) pull (task,strength) jobs; per-episode unstable-seed skip is
# inherited from the edited base main(). Per-strength output roots for separate export.
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
BASE=/workspace/vla/robotwin2_train/checkpoints/pi05_aloha_full_base/robotwin10_clean/4000
LOGD=/workspace/vla/eval_logs/delay
QDIR="$LOGD/queue"
mkdir -p "$LOGD"

EP_PER_JOB=${EP_PER_JOB:-10}
STRENGTHS=(${STRENGTHS:-1 2 3})
TASKS=(press_stapler adjust_bottle place_dual_shoes place_a2b_right open_laptop \
       click_bell place_phone_stand handover_block stack_blocks_two stack_bowls_two)

# ---- jobs = (task,strength); interleave by strength so early GPUs hit diverse tasks ----
JOBS=()
for s in "${STRENGTHS[@]}"; do
  for t in "${TASKS[@]}"; do
    JOBS+=("$t|$s")
  done
done
TOTAL=${#JOBS[@]}

rm -rf "$QDIR"; mkdir -p "$QDIR"; echo 0 > "$QDIR/counter"
claim() {
  exec 9>"$QDIR/lock"; flock 9
  local idx; idx=$(cat "$QDIR/counter"); echo $((idx+1)) > "$QDIR/counter"
  flock -u 9; echo "$idx"
}
stamp(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/dispatch.log"; }

worker() {
  local gpu=$1
  while true; do
    local idx; idx=$(claim)
    if [ "$idx" -ge "$TOTAL" ]; then break; fi
    local job="${JOBS[$idx]}" task n
    task="${job%|*}"; n="${job#*|}"
    local outroot="/workspace/vla/eval_out/openpi_robotwin_delay_s${n}"
    local log="$LOGD/gpu${gpu}__${task}__delay${n}.log"
    stamp "gpu=$gpu idx=$idx START task=$task delay=$n -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m scripts.eval_robotwin_watermark_map_robustness \
      --controller-postprocess delay --controller-delay-steps "$n" --run-tag "delay_${n}" \
      --task-name "$task" --task-config demo_clean \
      --config-name pi05_aloha_robotwin_base_local --checkpoint-dir "$BASE" \
      --detector-config-name pi05_aloha_robotwin_base_local --detector-checkpoint-dir "$BASE" \
      --robotwin-root /workspace/vla/RoboTwin \
      --num-episodes "$EP_PER_JOB" --seed 0 --secret-key 17 --beta 1.0 \
      --no-expert-check \
      --variants plain watermarked --reference-mode gaussian \
      --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
      --null-decoy-count 32 --subspace-rank 3 \
      --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 \
      --map-num-starts 4 --obs-sigma 1e-4 \
      --num-inference-steps 10 \
      --output-dir "$outroot/$task" \
      > "$log" 2>&1
    stamp "gpu=$gpu idx=$idx DONE  task=$task delay=$n exit=$?"
  done
  stamp "gpu=$gpu worker drained, exiting"
}

stamp "==== DELAY START: $TOTAL jobs (${#TASKS[@]} tasks x ${#STRENGTHS[@]} strengths), EP=$EP_PER_JOB, 8 GPUs ===="
for g in ${GPUS:-0 1 2 3 4 5 6 7}; do worker "$g" & done
wait
stamp "==== ALL DELAY JOBS DONE ===="
echo "OPENPI_RT_DELAY_COMPLETE"
