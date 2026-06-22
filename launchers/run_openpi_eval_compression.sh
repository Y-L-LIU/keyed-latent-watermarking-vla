#!/usr/bin/env bash
# Run openpi LIBERO eval for the compressed descendants (threat-model §12.5):
# - rollouts come from the LoRA-fine-tuned SUSPECT with pruning OR int8 quantization
#   applied to all multi-dim weight tensors of the descendant ckpt
# - detection uses the ORIGINAL base detector (pi05_libero)
# Spawns one process per (attack), uses 4 GPUs each.
#
# Usage: run_openpi_eval_compression.sh [SUITE] [NUM_TASKS] [NUM_TRIALS]
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
SUITE=${1:-libero_goal}
NUM_TASKS=${2:-10}
NUM_TRIALS=${3:-10}
# SCOPE=all (default): whole-model compressed ckpts. SCOPE=action: action-policy-only
# compressed ckpts (built via build_compressed_ckpt.py --scope action). Action-only outputs
# land in a separate _actiononly dir so both §12.5 settings coexist for comparison.
SCOPE=${SCOPE:-all}
SUF=""; [ "$SCOPE" = action ] && SUF="_actiononly"
DET_CFG=pi05_libero
DET_CKPT=/workspace/vla/models/pi05_libero
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
OUTROOT=/workspace/vla/eval_out_compression
mkdir -p "$OUTROOT"
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

case "$SUITE" in
  libero_goal)    SUITE_CFG=pi05_libero_goal_lora_from_libero ;;
  libero_spatial) SUITE_CFG=pi05_libero_spatial_lora_from_libero ;;
  *) echo "Unknown suite: $SUITE"; exit 2 ;;
esac

run_eval(){
  local atk=$1 gpus=$2 ckpt_root=$3
  local ckpt="$ckpt_root/2499"
  if [ ! -d "$ckpt/params" ]; then stamp "$atk: missing $ckpt/params"; return 1; fi
  local out="$OUTROOT/${SUITE}_${atk}${SUF}"
  mkdir -p "$out"
  stamp "EVAL $SUITE/$atk${SUF} start: ckpt=$ckpt gpus=$gpus tasks=$NUM_TASKS trials=$NUM_TRIALS"
  CUDA_VISIBLE_DEVICES=$gpus $PY -m scripts.eval_libero_action_inversion_postprocess_robustness \
    --controller-postprocess none --run-tag none \
    --config-name "$SUITE_CFG" --checkpoint-dir "$ckpt" \
    --detector-config-name "$DET_CFG" --detector-checkpoint-dir "$DET_CKPT" \
    --task-suite-name "$SUITE" --num-tasks "$NUM_TASKS" --num-trials-per-task "$NUM_TRIALS" \
    --detector wmf --reference-mode gaussian --beta 1.0 \
    --null-decoy-count 32 --subspace-rank 3 \
    --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
    --score-step-scope full_chunk --num-inversion-steps 10 \
    --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 4 --obs-sigma 1e-4 \
    --save-rollout-dir "$out/rollouts" --save-report-dir "$out/reports" \
    --resume-from-rollouts \
    > "$LOGD/eval_openpi_${SUITE}_${atk}.log" 2>&1
  local rc=$?; stamp "EVAL $SUITE/$atk exit=$rc"; return $rc
}

# Resolve the attacked ckpt roots (SUF="_actiononly" when SCOPE=action)
PRUNE_ROOT=/workspace/vla/openpi-checkpoints/${SUITE_CFG}/descendant_lora_prune30${SUF}
QUANT_ROOT=/workspace/vla/openpi-checkpoints/${SUITE_CFG}/descendant_lora_quant${SUF}

run_eval prune30 0,1,2,3 "$PRUNE_ROOT" &
P1=$!
run_eval quant   4,5,6,7 "$QUANT_ROOT" &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
stamp "ALL EVAL DONE prune=$R1 quant=$R2"
echo "OPENPI_COMP_EVAL_COMPLETE prune=$R1 quant=$R2"
