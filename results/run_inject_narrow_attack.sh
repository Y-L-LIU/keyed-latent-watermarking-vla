#!/bin/bash
# Inject-narrow UNDER ATTACK: pi0.5/LIBERO rollout with controller postprocess (ema-low etc.)
# via the robustness script, osmesa render, partial-MAP recovery, key restricted to work dims.
# Usage: run_inject_narrow_attack.sh <gpu> <watermark_dims|""> <out_tag> <num_tasks> <trials> <postproc> <strength>
#   postproc: smooth|clip|jitter ; strength: alpha/limit/std
set -e
GPU=$1; WMDIMS=$2; TAG=$3; NTASKS=${4:-2}; NTRIALS=${5:-4}; PP=${6:-smooth}; STR=${7:-0.2}
ROOT=/workspace/vla
OUT=$ROOT/eval_out/inject_narrow_smoke/$TAG
LOG=$ROOT/eval_out/inject_narrow_smoke/${TAG}.log
mkdir -p "$OUT"
case "$PP" in
  smooth) PPARG="--controller-smooth-alpha $STR";;
  clip)   PPARG="--controller-clip-limit $STR";;
  jitter) PPARG="--controller-jitter-std $STR";;
esac
cd $ROOT/openpi
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=osmesa \
OPENPI_DATA_HOME="$ROOT/openpi-cache" \
JAX_COMPILATION_CACHE_DIR="$ROOT/openpi-cache/jax_cache" \
PYTHONPATH="$ROOT/openpi/src:$ROOT/openpi:$ROOT/openpi/third_party/libero" \
python3 scripts/eval_libero_action_inversion_postprocess_robustness.py \
  --controller-postprocess "$PP" $PPARG --run-tag "$TAG" \
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
  > "$LOG" 2>&1
echo "done $TAG"
