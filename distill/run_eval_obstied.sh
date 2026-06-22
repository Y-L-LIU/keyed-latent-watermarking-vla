#!/usr/bin/env bash
# Obs-tied watermark eval (policy rollout -> base-detector MAP -> obs-tied key + 32 decoys).
# Reused for Phase-1 (teacher with injection) and the distillation arm (student, no injection).
#
# Env knobs:
#   GPU=0  TASKS=10  TRIALS=5  SECRET_KEY=42  Q=0.08  PROJ=0,1,2
#   POLICY_CFG=pi05_libero  POLICY_CKPT=/workspace/vla/models/pi05_libero
#   DET_CFG=pi05_libero     DET_CKPT=/workspace/vla/models/pi05_libero
#   SUITE=libero_goal  OUT=/workspace/vla/distill/phase1  TAG=teacher  KEYING=observation
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11

GPU=${GPU:-0}
TASKS=${TASKS:-10}
OFFSET=${OFFSET:-0}
TRIALS=${TRIALS:-5}
SECRET_KEY=${SECRET_KEY:-42}
Q=${Q:-0.08}
PROJ=${PROJ:-0,1,2}
KEYING=${KEYING:-observation}
SUITE=${SUITE:-libero_goal}
POLICY_CFG=${POLICY_CFG:-pi05_libero}
POLICY_CKPT=${POLICY_CKPT:-/workspace/vla/models/pi05_libero}
DET_CFG=${DET_CFG:-pi05_libero}
DET_CKPT=${DET_CKPT:-/workspace/vla/models/pi05_libero}
OUT=${OUT:-/workspace/vla/distill/phase1}
TAG=${TAG:-teacher}
MAP_ITERS=${MAP_ITERS:-50}
MAP_STARTS=${MAP_STARTS:-1}

out="$OUT/${SUITE}_${TAG}"
mkdir -p "$out"
echo "[eval] tag=$TAG policy=$POLICY_CKPT detector=$DET_CKPT keying=$KEYING k=$SECRET_KEY q=$Q proj=$PROJ tasks=$TASKS trials=$TRIALS gpu=$GPU -> $out"

CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.eval_libero_action_inversion \
  --config-name "$POLICY_CFG" --checkpoint-dir "$POLICY_CKPT" \
  --detector-config-name "$DET_CFG" --detector-checkpoint-dir "$DET_CKPT" \
  --task-suite-name "$SUITE" --num-tasks "$TASKS" --task-offset "$OFFSET" --num-trials-per-task "$TRIALS" \
  --detector wmf --reference-mode gaussian --beta 1.0 --secret-key "$SECRET_KEY" \
  --keying-mode "$KEYING" --obs-proj-dims "$PROJ" --obs-quantization "$Q" \
  --chunk-selection-strategy periodic --chunk-selection-period 1 --chunk-selection-count 1 \
  --null-decoy-count 32 --subspace-rank 3 \
  --score-step-scope full_chunk --num-inversion-steps 10 \
  --fm-latent-map --latent-map-iters "$MAP_ITERS" --latent-map-lr 0.1 --latent-prior-weight 1.0 \
  --map-num-starts "$MAP_STARTS" --obs-sigma 1e-4 \
  --save-rollout-dir "$out/rollouts" --save-report-dir "$out/reports"
echo "[eval] exit=$? tag=$TAG"
