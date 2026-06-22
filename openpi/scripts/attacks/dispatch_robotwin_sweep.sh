#!/bin/bash
# RoboTwin watermark eval + attack sweep dispatcher.
# Pool: all GPUs that have mem.used < 5GB. Polls every 60s; when a GPU is
# free, pops the next condition from QUEUE and launches a per-condition worker
# (10-task loop) on it. 60s stagger between launches (sapien init is heavier
# than libero/mujoco and benefits from a slightly longer gap).
set -u

# 11 conditions — matches Phase B sweep grid for direct comparison.
QUEUE=(
  "none|       |none"
  "clip|--controller-clip-limit 0.5|clip_0.5"
  "clip|--controller-clip-limit 1.0|clip_1.0"
  "clip|--controller-clip-limit 2.0|clip_2.0"
  "smooth|--controller-smooth-alpha 0.2|smooth_0.2"
  "smooth|--controller-smooth-alpha 0.5|smooth_0.5"
  "smooth|--controller-smooth-alpha 0.8|smooth_0.8"
  "jitter|--controller-jitter-std 0.005|jitter_0.005"
  "jitter|--controller-jitter-std 0.01|jitter_0.01"
  "jitter|--controller-jitter-std 0.05|jitter_0.05"
  "jitter|--controller-jitter-std 0.1|jitter_0.1"
)

# Episodes per task per condition. 3 episodes × 10 tasks × 2 variants = 60 npz / cond.
NUM_EP=3

WORKER=/workspace/vla/openpi/scripts/attacks/run_robotwin_condition.sh
LOG_DIR=/workspace/vla/attack_c_data/logs
DISP_LOG=${LOG_DIR}/robotwin_dispatch.log
mkdir -p "${LOG_DIR}"

# Use whichever GPUs are idle (mem.used < 5GB). robotwin uses sapien (not
# mujoco/EGL), so the GPU 0/4/6 SIGABRT pattern seen in libero shouldn't apply.
free_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' '{
          gsub(/ /,"",$1); gsub(/ /,"",$2);
          if (($2+0) < 5000) { print $1; exit }
        }'
}

echo "[$(date)] dispatcher start, queue=${#QUEUE[@]}, num_ep=${NUM_EP}" >> "${DISP_LOG}"

# Track which GPUs are already assigned this session so we don't double-fire.
declare -A IN_USE

while [ ${#QUEUE[@]} -gt 0 ]; do
    GPU=$(free_gpu)
    if [ -z "${GPU}" ] || [ "${IN_USE[${GPU}]:-0}" = "1" ]; then
        sleep 60
        # Reap finished workers so their GPU becomes "not in use" again.
        for g in "${!IN_USE[@]}"; do
            if ! pgrep -f "run_robotwin_condition.sh.*${g} ${NUM_EP}$" >/dev/null 2>&1; then
                # No worker tagged with this gpu remains.
                IN_USE[$g]=0
            fi
        done
        continue
    fi
    ENTRY=${QUEUE[0]}
    QUEUE=("${QUEUE[@]:1}")
    IFS='|' read -r MODE EXTRA TAG <<< "${ENTRY}"
    MODE=$(echo "$MODE" | xargs)
    EXTRA=$(echo "$EXTRA" | xargs)
    TAG=$(echo "$TAG" | xargs)
    echo "[$(date)] LAUNCH ${TAG} on GPU ${GPU}" | tee -a "${DISP_LOG}"
    IN_USE[$GPU]=1
    nohup bash "${WORKER}" "${TAG}" "${MODE}" "${EXTRA}" "${GPU}" "${NUM_EP}" \
        > "${LOG_DIR}/robotwin_${TAG}_dispatch.log" 2>&1 &
    sleep 60
done

echo "[$(date)] all queued, waiting for stragglers" >> "${DISP_LOG}"
wait
echo "[$(date)] dispatcher all done" >> "${DISP_LOG}"
