#!/usr/bin/env bash
# openpi pi0.5 LIBERO controller-DELAY robustness (Attack C, threat-model §12.1).
# This is the cell the original campaign SKIPPED (run_libero10_wm_4attacks.sh line 48:
# "delay skipped: separate mujoco SIGABRT issue") because it used MUJOCO_GL=egl.
# On THIS node EGL has no ICD, so we use MUJOCO_GL=osmesa (proven working for the same
# action-inversion eval in the compression runs) — which sidesteps the EGL SIGABRT.
#
# Single-model setup matching the Attack-C openpi LIBERO launcher: policy == detector ==
# pi05_libero (watermark lives in the base). The eval runs watermarked (beta=1) AND plain
# (beta=0) variants internally and writes a report with ROC-AUC. delay magnitude = N steps.
#
# Usage: run_openpi_libero_delay.sh [SUITE] [DELAY_STEPS] [GPUS] [NUM_TASKS] [NUM_TRIALS]
#   defaults: libero_10  1  0,1,2,3  10  5     (=> 10 tasks x 5 trials x {wm,plain} = 100 ep)
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11

SUITE=${1:-libero_10}
DELAY=${2:-1}
GPUS=${3:-0,1,2,3}
NUM_TASKS=${4:-10}
NUM_TRIALS=${5:-5}
TAG="delay_${DELAY}"

CKPT=/workspace/vla/models/pi05_libero
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
OUTROOT=/workspace/vla/attack_c_data/rollouts/openpi_libero
OUT="$OUTROOT/${SUITE}_${TAG}"
mkdir -p "$OUT"
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -d "$CKPT/params" ]; then stamp "FATAL: missing $CKPT/params"; exit 2; fi

stamp "EVAL openpi LIBERO $SUITE / $TAG start: ckpt=$CKPT gpus=$GPUS tasks=$NUM_TASKS trials=$NUM_TRIALS"
CUDA_VISIBLE_DEVICES=$GPUS $PY -m scripts.eval_libero_action_inversion_postprocess_robustness \
  --controller-postprocess delay --controller-delay-steps "$DELAY" --run-tag "$TAG" \
  --config-name pi05_libero --checkpoint-dir "$CKPT" \
  --detector-config-name pi05_libero --detector-checkpoint-dir "$CKPT" \
  --task-suite-name "$SUITE" --task-offset 0 --num-tasks "$NUM_TASKS" --num-trials-per-task "$NUM_TRIALS" \
  --detector wmf --reference-mode gaussian --beta 1.0 \
  --null-decoy-count 32 --subspace-rank 3 \
  --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
  --score-step-scope full_chunk --num-inversion-steps 10 \
  --fm-latent-map --latent-map-iters 50 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 1 \
  --obs-sigma 1e-4 \
  --save-rollout-dir "$OUT/rollouts" --save-report-dir "$OUT/reports" \
  --resume-from-rollouts \
  > "$LOGD/eval_openpi_libero_${SUITE}_${TAG}.log" 2>&1
rc=$?
stamp "EVAL openpi LIBERO $SUITE / $TAG exit=$rc"
echo "OPENPI_LIBERO_DELAY_COMPLETE suite=$SUITE tag=$TAG rc=$rc"
