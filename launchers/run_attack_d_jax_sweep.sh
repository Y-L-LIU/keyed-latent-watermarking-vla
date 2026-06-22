#!/usr/bin/env bash
# Attack-D (direct adversarial fine-tune, NO subspace) λ-sweep on pi0.5/LIBERO-10.
# 4 λ values, one per GPU (fsdp_devices=1), GPUs 0-3. λ=0 is the plain descendant
# (lam>0 gates the bilevel penalty). Base config = pi05_libero_10_lora_from_libero
# (LiberoHdf5, reads benchmark hdf5 directly; == paper §12.5 LIBERO descendant recipe).
set -uo pipefail
cd /workspace/vla/openpi
source /workspace/vla/launchers/node_env.sh
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src:$LIBERO_PP
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SAVE_BASE=/workspace/vla_out/attack_c/attacked
LOG_DIR=/workspace/vla/attack_c_data/logs_attackd
mkdir -p "$SAVE_BASE" "$LOG_DIR"

# --- tunables (finalized after smoke) ---
STEPS=${STEPS:-2000}
BATCH=${BATCH:-16}
ATK_BATCH=${ATK_BATCH:-8}
INNER=${INNER:-4}
DENOISE=${DENOISE:-4}
NKEYS=${NKEYS:-32}
# ----------------------------------------

run_one() {
  local lam=$1 gpu=$2
  local log="${LOG_DIR}/lam${lam}.log"
  echo "[$(date +%H:%M:%S)] launch lam=${lam} on GPU ${gpu} -> ${log}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
  nohup $PY scripts/attacks/finetune_attack_d_jax.py \
    --config-name pi05_libero_10_lora_from_libero \
    --exp-name "attack_d_lam${lam}" \
    --lambda-attack "${lam}" \
    --inner-iters "${INNER}" --inner-lr 0.1 --inner-prior-weight 1.0 --inner-obs-sigma 1e-2 \
    --inv-num-denoising-steps "${DENOISE}" \
    --num-false-keys "${NKEYS}" --reference-mode gaussian --sample-rate-hz 20.0 \
    --soft-max-temp 10.0 --attack-batch-size "${ATK_BATCH}" \
    --batch-size "${BATCH}" --num-train-steps "${STEPS}" --fsdp-devices 1 \
    --save-interval 1000 --log-interval 25 \
    --checkpoint-base-dir "${SAVE_BASE}" --overwrite \
    > "${log}" 2>&1 &
  echo $!
}

P0=$(run_one 0    0)
P01=$(run_one 0.1 1)
P1=$(run_one 1    2)
P10=$(run_one 10  3)
echo "[$(date +%H:%M:%S)] PIDs: lam0=$P0 lam0.1=$P01 lam1=$P1 lam10=$P10"
echo "$P0 $P01 $P1 $P10" > "${LOG_DIR}/sweep_pids.txt"
wait $P0 $P01 $P1 $P10
echo "[$(date +%H:%M:%S)] Attack-D sweep complete."
