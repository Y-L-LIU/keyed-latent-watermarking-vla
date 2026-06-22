#!/bin/bash
# Phase B sweep dispatcher: serializes worker launch with 30s stagger to avoid
# simultaneous EGL/CUDA context init SIGABRT race.
# Polls GPUs 4-7 every 60s; when a GPU is idle (mem < 5GB), pops the next
# queue entry and launches it there. Exits when queue is empty.
set -u
cd /workspace/vla/openpi

POLICY_CKPT=/workspace/vla/models/pi05_libero
EVAL_BASE=/workspace/scratch/anon/libero10_wm_postprocess_full
LOG_DIR=/workspace/vla/attack_c_data/logs

# QUEUE: each entry "mode|extra|tag"
QUEUE=(
  "none|       |none"
  "clip|--controller-clip-limit 2.0|clip_2.0"
  "smooth|--controller-smooth-alpha 0.8|smooth_0.8"
  "jitter|--controller-jitter-std 0.005|jitter_0.005"
  "jitter|--controller-jitter-std 0.05|jitter_0.05"
  "jitter|--controller-jitter-std 0.1|jitter_0.1"
)

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
    sleep 30  # stagger to avoid simultaneous EGL init race
}

free_gpu() {
    # echo first GPU in {4,5,6,7} with mem.used < 5000 MiB (idle)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' '{
          gsub(/ /,"",$1); gsub(/ /,"",$2);
          if (($1+0)>=4 && ($1+0)<=7 && ($2+0)<5000) { print $1; exit }
        }'
}

while [ ${#QUEUE[@]} -gt 0 ]; do
    gpu=$(free_gpu)
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

echo "[$(date)] dispatcher: all queued items launched. Waiting for stragglers."
wait
echo "[$(date)] dispatcher: all done."
