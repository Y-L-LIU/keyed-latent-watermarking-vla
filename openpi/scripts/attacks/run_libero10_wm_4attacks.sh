#!/bin/bash
# libero_10 main wm task: normal (no attack) + 4 controller postprocess attacks.
# Attack params aligned with lingbot-va defaults: clip(1.0), smooth(0.5), jitter(0.01), delay(1).
# 5 conditions x 1 GPU each = 5 GPUs in parallel.
set -e
cd /workspace/vla/openpi

POLICY_CKPT=/workspace/vla/models/pi05_libero
EVAL_BASE=/workspace/scratch/anon/libero10_wm_postprocess_full
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${EVAL_BASE}" "${LOG_DIR}"

run_one() {
    local mode=$1
    local gpu=$2
    local extra=$3
    local tag=$4
    local out="${EVAL_BASE}/${tag}"
    local log="${LOG_DIR}/libero10_${tag}.log"
    mkdir -p "${out}"
    echo "[$(date)] ${tag} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID="${gpu}" \
    PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
    .venv/bin/python /workspace/vla/openpi/scripts/attacks/run_eval_with_libero_patch.py \
        /workspace/vla/openpi/scripts/eval_libero_action_inversion_postprocess_robustness.py -- \
        --controller-postprocess ${mode} ${extra} --run-tag "${tag}" \
        --config-name pi05_libero --checkpoint-dir "${POLICY_CKPT}" \
        --detector-config-name pi05_libero --detector-checkpoint-dir "${POLICY_CKPT}" \
        --task-suite-name libero_10 --task-offset 0 --num-tasks 10 --num-trials-per-task 5 \
        --detector wmf --reference-mode gaussian --beta 1.0 \
        --null-decoy-count 32 --subspace-rank 3 \
        --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
        --score-step-scope full_chunk --num-inversion-steps 10 \
        --fm-latent-map --latent-map-iters 50 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 1 \
        --obs-sigma 1e-4 \
        --save-rollout-dir "${out}" \
        > "${log}" 2>&1
}

PIDS=()
run_one none   0 ""                                   none           & PIDS+=($!)
run_one clip   1 "--controller-clip-limit 1.0"        clip_1.0       & PIDS+=($!)
run_one smooth 2 "--controller-smooth-alpha 0.5"      smooth_0.5     & PIDS+=($!)
run_one jitter 3 "--controller-jitter-std 0.01"       jitter_0.01    & PIDS+=($!)
# delay skipped: separate mujoco SIGABRT issue, not gated by step budget fix.

echo "[$(date)] libero10 wm+4attacks PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "[$(date)] libero10 wm+4attacks complete."
