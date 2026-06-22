#!/bin/bash
# Phase B sweep dispatcher v2: stable-GPU pool only.
# Pool = {1, 2, 3, 5, 7}. GPUs 0/4/6 excluded (SIGABRT-flaky).
# Polls every 60s; when a pool GPU has mem.used < 5GB (idle), pops next queue
# entry and launches it there. 30s stagger between launches.
set -u
cd /workspace/vla/openpi

POLICY_CKPT=/workspace/vla/models/pi05_libero
EVAL_BASE=/workspace/scratch/anon/libero10_wm_postprocess_full
LOG_DIR=/workspace/vla/attack_c_data/logs

# 6 conditions to (re)run on the stable pool.
QUEUE=(
  "none|       |none"
  "clip|--controller-clip-limit 2.0|clip_2.0"
  "smooth|--controller-smooth-alpha 0.8|smooth_0.8"
  "jitter|--controller-jitter-std 0.005|jitter_0.005"
  "jitter|--controller-jitter-std 0.05|jitter_0.05"
  "jitter|--controller-jitter-std 0.1|jitter_0.1"
)

POOL="1 2 3 5 7"

launch() {
    local mode=$1
    local extra=$2
    local tag=$3
    local gpu=$4
    local out="${EVAL_BASE}/${tag}"
    local log="${LOG_DIR}/libero10_${tag}.log"
    mkdir -p "${out}"
    echo "[$(date)] LAUNCH ${tag} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID="${gpu}" \
    PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
    nohup .venv/bin/python /workspace/vla/openpi/scripts/attacks/run_eval_with_libero_patch.py \
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
        > "${log}" 2>&1 &
    sleep 30
}

free_gpu_in_pool() {
    for g in $POOL; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        if [ -n "$mem" ] && [ "$mem" -lt 5000 ]; then
            echo "$g"
            return
        fi
    done
}

while [ ${#QUEUE[@]} -gt 0 ]; do
    gpu=$(free_gpu_in_pool)
    if [ -z "${gpu}" ]; then
        sleep 60
        continue
    fi
    entry=${QUEUE[0]}
    QUEUE=("${QUEUE[@]:1}")
    IFS='|' read -r mode extra tag <<< "${entry}"
    mode=$(echo "$mode" | xargs)
    extra=$(echo "$extra" | xargs)
    tag=$(echo "$tag" | xargs)
    launch "${mode}" "${extra}" "${tag}" "${gpu}"
done

echo "[$(date)] dispatcher v2: all 6 queued items launched. Waiting for stragglers."
wait
echo "[$(date)] dispatcher v2: all done."
