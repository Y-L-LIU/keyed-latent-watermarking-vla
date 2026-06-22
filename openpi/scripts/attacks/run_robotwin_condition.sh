#!/bin/bash
# Per-condition robotwin runner: iterates all 10 trained tasks under one
# controller-postprocess attack, on a single GPU. Each task tries up to 3
# starting seeds if the first hits UnStableError mid-rollout.
# Args: $1=TAG  $2=MODE  $3="EXTRA"  $4=GPU  $5=NUM_EPISODES
set -u
TAG=$1
MODE=$2
EXTRA=$3
GPU=$4
NUM_EP=${5:-3}

WRAPPER=/workspace/vla/openpi/scripts/attacks/run_robotwin_eval_with_config.py
VENV_PY=/workspace/scratch/anon/robotwin2/venv_pi05/bin/python
ROBOTWIN_ROOT=/workspace/vla/RoboTwin
CKPT=/workspace/scratch/anon/robotwin2/checkpoints/pi05_aloha_full_base/robotwin10_clean/4000
OUT_BASE=${OUT_BASE:-/workspace/scratch/anon/robotwin2/wm_eval}
LOG_DIR=/workspace/vla/attack_c_data/logs

mkdir -p "${OUT_BASE}/${TAG}" "${LOG_DIR}"
LOG=${LOG_DIR}/robotwin_${TAG}.log

TASKS=(
  adjust_bottle click_bell handover_block open_laptop place_a2b_right
  place_phone_stand place_shoe press_stapler stack_blocks_two stack_bowls_two
)

echo "[$(date)] START tag=${TAG} mode=${MODE} extra='${EXTRA}' gpu=${GPU} num_ep=${NUM_EP}" | tee -a "${LOG}"
cd "${ROBOTWIN_ROOT}" || exit 1

for TASK in "${TASKS[@]}"; do
    for SEED in 0 1 2; do
        echo "[$(date)] tag=${TAG} task=${TASK} seed=${SEED} attempt" >> "${LOG}"
        CUDA_VISIBLE_DEVICES="${GPU}" timeout 1800 "${VENV_PY}" "${WRAPPER}" \
            --controller-postprocess "${MODE}" ${EXTRA} \
            --run-tag "${TAG}" \
            --robotwin-root "${ROBOTWIN_ROOT}" \
            --task-name "${TASK}" \
            --config-name pi05_aloha_full_base \
            --checkpoint-dir "${CKPT}" \
            --output-dir "${OUT_BASE}" \
            --num-episodes "${NUM_EP}" --no-expert-check --seed "${SEED}" \
            --detector wmf --reference-mode gaussian --beta 1.0 \
            --null-decoy-count 16 --subspace-rank 3 \
            --chunk-selection-strategy stateful_online --chunk-selection-period 1 --chunk-selection-count 5 \
            --score-step-scope full_chunk --num-inference-steps 10 \
            --latent-map-iters 50 --latent-map-lr 0.1 --latent-prior-weight 1.0 --map-num-starts 1 \
            --obs-sigma 1e-4 \
            >> "${LOG}" 2>&1
        rc=$?
        # Count how many npz files exist for this task. Success means we wrote
        # at least 2*NUM_EP files (one per variant per episode).
        NPZ_COUNT=$(ls "${OUT_BASE}/${TAG}/${TASK}/pi05_aloha_full_base/4000/"*.npz 2>/dev/null | wc -l)
        if [ "${NPZ_COUNT}" -ge $((2 * NUM_EP)) ]; then
            echo "[$(date)] tag=${TAG} task=${TASK} seed=${SEED} OK (rc=${rc}, npz=${NPZ_COUNT})" >> "${LOG}"
            break
        fi
        echo "[$(date)] tag=${TAG} task=${TASK} seed=${SEED} INCOMPLETE rc=${rc} npz=${NPZ_COUNT} -> retry" >> "${LOG}"
    done
done

echo "[$(date)] DONE tag=${TAG} gpu=${GPU}" | tee -a "${LOG}"
