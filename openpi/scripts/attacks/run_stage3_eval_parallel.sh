#!/bin/bash
# Stage 3 — 5 parallel evals (one per λ), each on 1 GPU.
# Detector = owner's original pi05_libero. Rollout = attacked checkpoint.
# Threat model: attacker fine-tuned to invariance; we check whether owner can
# still recover the watermark by injecting beta=1.0 into the attacked model.
set -e
cd /workspace/vla/openpi

ATTACKED_BASE=/workspace/scratch/anon/attack_c/attacked/pi05_libero_low_mem_finetune
DETECTOR_CKPT=/workspace/vla/models/pi05_libero
EVAL_BASE=/workspace/scratch/anon/attack_c/eval
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${LOG_DIR}"

run_eval() {
    local lam=$1
    local gpu=$2
    local ckpt="${ATTACKED_BASE}/attack_c_lam${lam}_r8/1999"
    local out="${EVAL_BASE}/lam${lam}/libero_10"
    local log="${LOG_DIR}/eval_lam${lam}.log"
    mkdir -p "${out}"
    echo "[$(date)] eval lam=${lam} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID="${gpu}" \
    PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
    .venv/bin/python /workspace/vla/openpi/scripts/attacks/run_eval_with_libero_patch.py \
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
        > "${log}" 2>&1
}

PIDS=()
i=0
for lam in 0 0.1 1 10 100; do
    run_eval "${lam}" "${i}" &
    PIDS+=($!)
    i=$((i+1))
done

echo "[$(date)] Stage 3 PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "[$(date)] Stage 3 eval complete."
