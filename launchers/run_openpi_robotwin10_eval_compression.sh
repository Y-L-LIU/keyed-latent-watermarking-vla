#!/usr/bin/env bash
# openpi pi0.5 RoboTwin COMPRESSION robustness — over the 10 tasks the model was
# actually trained on (robotwin10_clean), NOT just beat_block_hammer (which is NOT
# in the training set). Spreads 100+100 (wm/plain) across 10 tasks = 10 ep/task.
#   suspect = pi05_aloha_robotwin_lora_local descendant with prune30 OR int8 quant
#   detector = pi05_aloha_robotwin_base_local (unmodified base, no LoRA)
# 4 GPUs: prune on 0(tasks0-4)/1(tasks5-9), quant on 2(tasks0-4)/3(tasks5-9).
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
EP_PER_TASK=${1:-10}
# SCOPE=all (default): whole-model compressed ckpts. SCOPE=action: action-policy-only
# compressed ckpts (build_compressed_ckpt.py --scope action); outputs to _actiononly dirs.
SCOPE=${SCOPE:-all}
SUF=""; [ "$SCOPE" = action ] && SUF="_actiononly"
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
OUTROOT=/workspace/vla/eval_out_compression
DET_CFG=pi05_aloha_robotwin_base_local
DET_CKPT=/workspace/vla/robotwin2_train/checkpoints/pi05_aloha_full_base/robotwin10_clean/4000
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

# The 10 robotwin10_clean tasks (identified from the training data episode blocks).
TASKS=(press_stapler adjust_bottle place_dual_shoes place_a2b_right open_laptop \
       click_bell place_phone_stand handover_block stack_blocks_two stack_bowls_two)

run_group(){
  local atk=$1 gpu=$2 ckpt=$3 lo=$4 hi=$5   # tasks index [lo,hi)
  if [ ! -d "$ckpt/params" ]; then stamp "$atk: missing $ckpt/params"; return 1; fi
  local out="$OUTROOT/openpi_robotwin10_${atk}${SUF}"
  mkdir -p "$out"
  for ((i=lo; i<hi; i++)); do
    local task="${TASKS[$i]}"
    stamp "EVAL RT10/$atk gpu=$gpu task=$task ($((i+1))/10) ep=$EP_PER_TASK"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m scripts.eval_robotwin_watermark_map \
      --task-name "$task" --task-config demo_clean \
      --config-name pi05_aloha_robotwin_lora_local --checkpoint-dir "$ckpt" \
      --detector-config-name "$DET_CFG" --detector-checkpoint-dir "$DET_CKPT" \
      --robotwin-root /workspace/vla/RoboTwin \
      --num-episodes "$EP_PER_TASK" --seed 0 --secret-key 17 --beta 1.0 \
      --no-expert-check \
      --variants plain watermarked --reference-mode gaussian \
      --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
      --null-decoy-count 32 --subspace-rank 3 \
      --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 4 --obs-sigma 1e-4 \
      --num-inference-steps 10 \
      --output-dir "$out" \
      >> "$LOGD/eval_openpi_rt10_${atk}_gpu${gpu}.log" 2>&1
    stamp "EVAL RT10/$atk task=$task done exit=$?"
  done
}

PRUNE=/workspace/vla/openpi-checkpoints/pi05_aloha_robotwin_lora_local/robotwin_descendant_prune30${SUF}/1999
QUANT=/workspace/vla/openpi-checkpoints/pi05_aloha_robotwin_lora_local/robotwin_descendant_quant${SUF}/1999

run_group prune30 0 "$PRUNE" 0 5  &
run_group prune30 1 "$PRUNE" 5 10 &
run_group quant   2 "$QUANT" 0 5  &
run_group quant   3 "$QUANT" 5 10 &
wait
stamp "ALL openpi RT10 compression eval done"
echo "OPENPI_RT_COMP_EVAL_COMPLETE"
