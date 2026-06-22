#!/bin/bash
# Phase B param sweep on GPUs 4-7, plus `none` relaunch (GPU 0 crashes silently).
# Two waves of 4 conditions each, parallel within wave, sequential across waves.
# Uses libero_10's native 520-step budget (--max-rollout-steps NOT set).
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

echo "===== Wave 1 ====="
PIDS=()
run_one none   4 ""                                   none           & PIDS+=($!)
run_one clip   5 "--controller-clip-limit 0.5"        clip_0.5       & PIDS+=($!)
run_one clip   6 "--controller-clip-limit 2.0"        clip_2.0       & PIDS+=($!)
run_one smooth 7 "--controller-smooth-alpha 0.2"      smooth_0.2     & PIDS+=($!)
echo "[$(date)] Wave 1 PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "[$(date)] Wave 1 complete."

echo "===== Wave 2 ====="
PIDS=()
run_one smooth 4 "--controller-smooth-alpha 0.8"      smooth_0.8     & PIDS+=($!)
run_one jitter 5 "--controller-jitter-std 0.005"      jitter_0.005   & PIDS+=($!)
run_one jitter 6 "--controller-jitter-std 0.05"       jitter_0.05    & PIDS+=($!)
run_one jitter 7 "--controller-jitter-std 0.1"        jitter_0.1     & PIDS+=($!)
echo "[$(date)] Wave 2 PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "[$(date)] Wave 2 complete."

echo "[$(date)] Phase B param sweep all done."
