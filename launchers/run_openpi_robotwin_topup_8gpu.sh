#!/usr/bin/env bash
# pi0.5/RoboTwin base-CLEAN top-up — 4-start MAP, fills all 8 GPUs via a pull queue.
# Per TOPUP_PROMPT.md: fix the single-start-MAP bug + enlarge the weakest paper cell.
# 8 workers (one per GPU) pull (task,seed) jobs from a shared queue. --no-expert-check
# (curobo planner not installed here); the eval script now skips unstable seeds per-episode.
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
OUTROOT=/workspace/vla/eval_out/openpi_robotwin_topup
LOGD=/workspace/vla/eval_logs/topup
QDIR="$LOGD/queue"
mkdir -p "$OUTROOT" "$LOGD"

EP_PER_JOB=${EP_PER_JOB:-10}
# seeds shift the eval seed window by 100000 each -> disjoint, distinct episodes per shard
SEEDS=(${SEEDS:-0 1 2 3})

# The 10 robotwin10_clean tasks the base model was trained on.
TASKS=(press_stapler adjust_bottle place_dual_shoes place_a2b_right open_laptop \
       click_bell place_phone_stand handover_block stack_blocks_two stack_bowls_two)

# ---- build job list: (task,seed) pairs, interleaved so early GPUs hit diverse tasks ----
JOBS=()
for s in "${SEEDS[@]}"; do
  for t in "${TASKS[@]}"; do
    JOBS+=("$t|$s")
  done
done
TOTAL=${#JOBS[@]}

# ---- atomic pull queue (flock-protected counter) ----
rm -rf "$QDIR"; mkdir -p "$QDIR"
echo 0 > "$QDIR/counter"
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
    local job="${JOBS[$idx]}" task seed
    task="${job%|*}"; seed="${job#*|}"
    local out="$OUTROOT/${task}__s${seed}"
    local log="$LOGD/gpu${gpu}__${task}__s${seed}.log"
    stamp "gpu=$gpu idx=$idx START task=$task seed=$seed -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m scripts.eval_robotwin_watermark_map \
      --task-name "$task" --task-config demo_clean \
      --config-name pi05_aloha_robotwin_base_local --checkpoint-dir "$BASE" \
      --detector-config-name pi05_aloha_robotwin_base_local --detector-checkpoint-dir "$BASE" \
      --robotwin-root /workspace/vla/RoboTwin \
      --num-episodes "$EP_PER_JOB" --seed "$seed" --secret-key 17 --beta 1.0 \
      --no-expert-check \
      --variants plain watermarked --reference-mode gaussian \
      --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
      --null-decoy-count 32 --subspace-rank 3 \
      --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 \
      --map-num-starts 4 --obs-sigma 1e-4 \
      --num-inference-steps 10 \
      --output-dir "$out" \
      > "$log" 2>&1
    stamp "gpu=$gpu idx=$idx DONE  task=$task seed=$seed exit=$?"
  done
  stamp "gpu=$gpu worker drained, exiting"
}

stamp "==== TOPUP START: $TOTAL jobs (${#TASKS[@]} tasks x ${#SEEDS[@]} seeds), EP_PER_JOB=$EP_PER_JOB, 8 GPUs ===="
for g in 0 1 2 3 4 5 6 7; do worker "$g" & done
wait
stamp "==== ALL TOPUP JOBS DONE ===="
echo "OPENPI_RT_TOPUP_COMPLETE"
