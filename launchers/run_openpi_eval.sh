#!/usr/bin/env bash
# Watermark detection eval for the fine-tuned descendants (threat-model §12.5):
# rollouts come from the LoRA-fine-tuned SUSPECT model, detection uses the ORIGINAL
# base detector (pi05_libero). Runs goal (gpu0-3) + spatial (gpu4-7) in parallel.
#
# Usage: run_openpi_eval.sh [NUM_TASKS] [NUM_TRIALS]   (defaults scoped for overnight on OSMesa)
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa            # no nvidia EGL ICD on this box -> CPU rendering
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
EXP=descendant_lora
NUM_TASKS=${1:-4}
NUM_TRIALS=${2:-2}
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
OUT=/workspace/vla/eval_out
DET_CFG=pi05_libero
DET_CKPT=/workspace/vla/models/pi05_libero
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

# pick highest-numbered checkpoint step for a config
latest_ckpt(){ ls -d /workspace/vla/openpi-checkpoints/$1/$EXP/[0-9]* 2>/dev/null | sort -t/ -k7 -n | tail -1; }

run_eval(){
  local suite=$1 cfg=$2 gpus=$3
  local ckpt; ckpt=$(latest_ckpt "$cfg")
  if [ -z "$ckpt" ]; then stamp "EVAL $suite: NO checkpoint found for $cfg"; return 1; fi
  stamp "EVAL $suite start: ckpt=$ckpt gpus=$gpus tasks=$NUM_TASKS trials=$NUM_TRIALS"
  CUDA_VISIBLE_DEVICES=$gpus $PY -m scripts.eval_libero_action_inversion_postprocess_robustness \
    --controller-postprocess none --run-tag none \
    --config-name "$cfg" --checkpoint-dir "$ckpt" \
    --detector-config-name "$DET_CFG" --detector-checkpoint-dir "$DET_CKPT" \
    --task-suite-name "$suite" --num-tasks "$NUM_TASKS" --num-trials-per-task "$NUM_TRIALS" \
    --detector wmf --reference-mode gaussian --beta 1.0 \
    --null-decoy-count 32 --subspace-rank 3 \
    --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
    --score-step-scope full_chunk --num-inversion-steps 10 \
    --latent-map-iters 100 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 4 --obs-sigma 1e-4 \
    --save-rollout-dir "$OUT/$suite/rollouts" --save-report-dir "$OUT/$suite/reports" \
    > "$LOGD/eval_$suite.log" 2>&1
  local rc=$?; stamp "EVAL $suite exit=$rc"; return $rc
}

run_eval libero_goal    pi05_libero_goal_lora_from_libero    0,1,2,3 &
P1=$!
run_eval libero_spatial pi05_libero_spatial_lora_from_libero 4,5,6,7 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
stamp "ALL EVAL DONE goal_exit=$R1 spatial_exit=$R2"
echo "OPENPI_EVAL_COMPLETE goal=$R1 spatial=$R2"
