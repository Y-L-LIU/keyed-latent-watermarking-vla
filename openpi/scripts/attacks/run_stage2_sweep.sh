#!/bin/bash
# Stage 2 — sequential JAX LoRA fine-tune sweep across lambda values.
# Each run uses all 8 H100s via FSDP (fsdp_devices=8).
set -e

cd /workspace/vla/openpi

SUBSPACE=/workspace/vla/attack_c_data/subspace/wm_subspace_r8_diff.npz
SAVE_BASE=/workspace/vla/attack_c_data/attacked
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${SAVE_BASE}" "${LOG_DIR}"

for lam in 0 0.1 1 10 100; do
    LOG_FILE="${LOG_DIR}/lam${lam}.log"
    echo "[$(date)] Starting lambda=${lam} -> ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MUJOCO_GL=egl \
    PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python /workspace/vla/openpi/scripts/attacks/finetune_attack_c_jax.py \
        --config-name pi05_libero_low_mem_finetune \
        --exp-name "attack_c_lam${lam}_r8" \
        --subspace-path "${SUBSPACE}" \
        --lambda-inv ${lam} \
        --inv-num-denoising-steps 4 \
        --inv-eps-sigma 0.5 \
        --batch-size 128 \
        --num-train-steps 2000 \
        --fsdp-devices 8 \
        --save-interval 500 \
        --log-interval 25 \
        --checkpoint-base-dir "${SAVE_BASE}" \
        --overwrite \
        > "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    echo "[$(date)] lambda=${lam} exited with code ${EXIT_CODE}"
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "Aborting sweep — lambda=${lam} failed."
        exit ${EXIT_CODE}
    fi
done

echo "[$(date)] Stage 2 sweep complete."
