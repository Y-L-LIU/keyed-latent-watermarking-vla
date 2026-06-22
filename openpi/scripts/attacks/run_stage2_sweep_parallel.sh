#!/bin/bash
# Stage 2 — 4 parallel JAX LoRA fine-tune jobs, 2 GPUs each (FSDP=2).
# Goal: cut wall time ~4x vs sequential 8-GPU runs.
# Memory: keep batch=32 so per-card activation matches the FSDP-8 batch=128 baseline.
set -e

cd /workspace/vla/openpi

SUBSPACE=/workspace/vla/attack_c_data/subspace/wm_subspace_r8_diff.npz
SAVE_BASE=/workspace/scratch/anon/attack_c/attacked
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${SAVE_BASE}" "${LOG_DIR}"

run_one() {
    local lam=$1
    local gpus=$2
    local logfile="${LOG_DIR}/lam${lam}.log"
    echo "[$(date)] starting lam=${lam} on GPUs ${gpus} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpus}" \
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
        --batch-size 32 \
        --num-train-steps 2000 \
        --fsdp-devices 2 \
        --save-interval 500 \
        --log-interval 25 \
        --checkpoint-base-dir "${SAVE_BASE}" \
        --overwrite \
        > "${logfile}" 2>&1
}

# First wave: 4 parallel jobs (4 × 2 GPUs = 8 GPUs).
run_one 0   "0,1" &
PID0=$!
run_one 0.1 "2,3" &
PID01=$!
run_one 1   "4,5" &
PID1=$!
run_one 10  "6,7" &
PID10=$!

echo "[$(date)] Wave 1 PIDs: lam0=${PID0} lam0.1=${PID01} lam1=${PID1} lam10=${PID10}"
wait ${PID0} ${PID01} ${PID1} ${PID10}
echo "[$(date)] Wave 1 complete."

# Second wave: only lam=100 left, use 8 GPUs.
echo "[$(date)] Starting wave 2: lam=100 on all 8 GPUs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MUJOCO_GL=egl \
PYTHONPATH=/workspace/vla/openpi/src:third_party/libero \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python /workspace/vla/openpi/scripts/attacks/finetune_attack_c_jax.py \
    --config-name pi05_libero_low_mem_finetune \
    --exp-name "attack_c_lam100_r8" \
    --subspace-path "${SUBSPACE}" \
    --lambda-inv 100 \
    --inv-num-denoising-steps 4 \
    --inv-eps-sigma 0.5 \
    --batch-size 128 \
    --num-train-steps 2000 \
    --fsdp-devices 8 \
    --save-interval 500 \
    --log-interval 25 \
    --checkpoint-base-dir "${SAVE_BASE}" \
    --overwrite \
    > "${LOG_DIR}/lam100.log" 2>&1

echo "[$(date)] Stage 2 sweep complete."
