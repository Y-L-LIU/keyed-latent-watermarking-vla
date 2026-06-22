#!/bin/bash
# Stage 3 dispatcher v2: re-run eval on 5 lambda checkpoints with FULL
# 520-step rollout budget (no --max-rollout-steps cap).
# Pool = {1, 2, 3, 5, 7}; GPUs 0/4/6 excluded (SIGABRT-flaky).
# 30s stagger to avoid simultaneous EGL init.
set -u
cd /workspace/vla/openpi

ATTACKED_BASE=/workspace/scratch/anon/attack_c/attacked/pi05_libero_low_mem_finetune
DETECTOR_CKPT=/workspace/vla/models/pi05_libero
EVAL_BASE=/workspace/scratch/anon/attack_c/eval_full
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${LOG_DIR}"

# 5 lambdas to re-eval
QUEUE=("0" "0.1" "1" "10" "100")
POOL="1 2 3 5 7"

launch_lam() {
    local lam=$1
    local gpu=$2
    local ckpt="${ATTACKED_BASE}/attack_c_lam${lam}_r8/1999"
    local out="${EVAL_BASE}/lam${lam}/libero_10"
    local log="${LOG_DIR}/eval_lam${lam}_full.log"
    mkdir -p "${out}"
    echo "[$(date)] LAUNCH lam=${lam} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID="${gpu}" \
    PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
    nohup .venv/bin/python /workspace/vla/openpi/scripts/attacks/run_eval_with_libero_patch.py \
        /workspace/vla/openpi/scripts/eval_libero_action_inversion_postprocess_robustness.py -- \
        --controller-postprocess none --run-tag "attack_c_lam${lam}" \
        --config-name pi05_libero --checkpoint-dir "${ckpt}" \
        --detector-config-name pi05_libero --detector-checkpoint-dir "${DETECTOR_CKPT}" \
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
    lam=${QUEUE[0]}
    QUEUE=("${QUEUE[@]:1}")
    launch_lam "${lam}" "${gpu}"
done

echo "[$(date)] stage3 dispatcher: all 5 launched. Waiting for stragglers."
wait
echo "[$(date)] stage3 dispatcher: all done."
