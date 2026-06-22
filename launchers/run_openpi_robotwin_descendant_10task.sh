#!/usr/bin/env bash
# Fix pi0.5/RoboTwin DESCENDANT cell: re-run on the correct robotwin10 10-task set
# (the existing descendant_openpi_n100 is single-task beat_block_hammer — wrong + OOD).
# openpi eval runs 4-start MAP on BOTH variants already (symmetric), so the ONLY fix here is
# task coverage. Policy = LoRA descendant, detector = base. Pull queue over GPUs (default 0,1).
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
DESC=/workspace/vla/openpi-checkpoints/pi05_aloha_robotwin_lora_local/robotwin_descendant/1999
BASE=/workspace/vla/robotwin2_train/checkpoints/pi05_aloha_full_base/robotwin10_clean/4000
OUTROOT=/workspace/vla/eval_out/openpi_robotwin_descendant_10task
LOGD=/workspace/vla/eval_logs/descendant10
QDIR="$LOGD/queue"
mkdir -p "$OUTROOT" "$LOGD"

EP_PER_JOB=${EP_PER_JOB:-15}
MAXSTEPS=${MAXSTEPS:-250}
SEEDS=(${SEEDS:-0})
GPUS=(${GPUS:-0 1})
TASKS=(press_stapler adjust_bottle place_dual_shoes place_a2b_right open_laptop \
       click_bell place_phone_stand handover_block stack_blocks_two stack_bowls_two)

JOBS=()
for s in "${SEEDS[@]}"; do for t in "${TASKS[@]}"; do JOBS+=("$t|$s"); done; done
TOTAL=${#JOBS[@]}

rm -rf "$QDIR"; mkdir -p "$QDIR"; echo 0 > "$QDIR/counter"
claim(){ exec 9>"$QDIR/lock"; flock 9; local i; i=$(cat "$QDIR/counter"); echo $((i+1))>"$QDIR/counter"; flock -u 9; echo "$i"; }
stamp(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/dispatch.log"; }

worker(){
  local gpu=$1
  while true; do
    local idx; idx=$(claim); if [ "$idx" -ge "$TOTAL" ]; then break; fi
    local job="${JOBS[$idx]}" task seed; task="${job%|*}"; seed="${job#*|}"
    local out="$OUTROOT/${task}__s${seed}" log="$LOGD/gpu${gpu}__${task}__s${seed}.log"
    stamp "gpu=$gpu idx=$idx START task=$task seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m scripts.eval_robotwin_watermark_map \
      --task-name "$task" --task-config demo_clean \
      --config-name pi05_aloha_robotwin_lora_local --checkpoint-dir "$DESC" \
      --detector-config-name pi05_aloha_robotwin_base_local --detector-checkpoint-dir "$BASE" \
      --robotwin-root /workspace/vla/RoboTwin \
      --num-episodes "$EP_PER_JOB" --seed "$seed" --secret-key 17 --beta 1.0 \
      --no-expert-check \
      --variants plain watermarked --reference-mode gaussian \
      --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
      --null-decoy-count 32 --subspace-rank 3 \
      --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 \
      --map-num-starts 4 --obs-sigma 1e-4 --num-inference-steps 10 \
      --max-rollout-steps "$MAXSTEPS" \
      --output-dir "$out" > "$log" 2>&1
    stamp "gpu=$gpu idx=$idx DONE  task=$task seed=$seed exit=$?"
  done
  stamp "gpu=$gpu worker drained"
}

stamp "==== PI0.5 DESCENDANT 10-TASK START: $TOTAL jobs, EP=$EP_PER_JOB, GPUs=${GPUS[*]} ===="
for g in "${GPUS[@]}"; do worker "$g" & done
wait
stamp "==== ALL PI0.5 DESCENDANT 10-TASK DONE ===="
echo "OPENPI_RT_DESCENDANT10_COMPLETE"
