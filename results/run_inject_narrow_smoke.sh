#!/bin/bash
# Inject-narrow smoke: rollout pi0.5/LIBERO with the key restricted to work dims (0..6),
# matched to the committed clean config (gaussian/beta=1.0/wmf/stateful_online/fm-latent-map).
# Usage: run_inject_narrow_smoke.sh <gpu> <watermark_dims> <out_tag> <num_tasks> <trials> [extra...]
set -e
GPU=$1; WMDIMS=$2; TAG=$3; NTASKS=${4:-1}; NTRIALS=${5:-2}; shift 5 || shift $#
EXTRA="$@"
ROOT=/workspace/vla
OUT=$ROOT/eval_out/inject_narrow_smoke/$TAG
LOG=$ROOT/eval_out/inject_narrow_smoke/${TAG}.log
mkdir -p "$OUT"
cd $ROOT/openpi
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=osmesa \
OPENPI_DATA_HOME="$ROOT/openpi-cache" \
JAX_COMPILATION_CACHE_DIR="$ROOT/openpi-cache/jax_cache" \
PYTHONPATH="$ROOT/openpi/src:$ROOT/openpi:$ROOT/openpi/third_party/libero" \
python3 scripts/eval_libero_action_inversion.py \
  --config-name pi05_libero --checkpoint-dir "$ROOT/models/pi05_libero" \
  --task-suite-name libero_10 --task-offset 0 --num-tasks "$NTASKS" --num-trials-per-task "$NTRIALS" \
  --secret-key 17 --beta 1.0 --sample-rate-hz 20.0 \
  --reference-mode gaussian --freq-min-hz 1.0 --freq-max-hz 2.0 --n-tones 4 \
  --detector wmf --subspace-rank 3 --null-decoy-count 32 \
  --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
  --score-step-scope full_chunk --num-inversion-steps 10 \
  --fm-latent-map --latent-map-iters 30 --latent-map-lr 0.1 --map-num-starts 2 \
  ${WMDIMS:+--watermark-dims $WMDIMS} \
  --save-rollout-dir "$OUT" \
  $EXTRA > "$LOG" 2>&1
echo "done $TAG -> $OUT (log: $LOG)"
