#!/bin/bash
# RoboTwin watermark evaluation launcher — parallel across GPUs.
#
# Runs normal watermark rollout + robustness variants, each single-task job
# on one GPU. Jobs are queued and dispatched to free GPUs.
#
# Usage:
#   bash wan_va/wm/launch_robotwin_wm_eval.sh /path/to/RoboTwin

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROBOTWIN_ROOT="${1:?Usage: $0 /path/to/RoboTwin}"
LINGBOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_BASE="${LINGBOT_DIR}/outputs/wm_robotwin"
NUM_GPUS=8
TEST_NUM=5
MAP_STEPS=10

# 10 representative tasks
TASKS=(
    adjust_bottle stack_bowls_two place_a2b_right open_laptop press_stapler
    place_shoe handover_block click_bell place_phone_stand stack_blocks_two
)

# Robustness settings: "mode param_flag param_value"
ROBUST_SETTINGS=(
    "smooth --controller-smooth-alpha 0.3"
    "smooth --controller-smooth-alpha 0.5"
    "smooth --controller-smooth-alpha 0.7"
    "jitter --controller-jitter-std 0.005"
    "jitter --controller-jitter-std 0.01"
    "jitter --controller-jitter-std 0.02"
    "delay --controller-delay-steps 1"
    "delay --controller-delay-steps 2"
)

cd "$LINGBOT_DIR"

# Virtual environment activation
VENV_DIR="${LINGBOT_DIR}/../.venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    VENV_ACTIVATE="source ${VENV_DIR}/bin/activate && "
else
    VENV_ACTIVATE=""
fi

# Ensure EGL/Vulkan directories exist for SAPIEN
mkdir -p /usr/share/glvnd/egl_vendor.d 2>/dev/null || true
mkdir -p /etc/glvnd/egl_vendor.d 2>/dev/null || true

# --- Job queue ---
# Each job = (GPU_ID, command).
# We fill a FIFO of commands and dispatch to GPUs as they become free.

LOGDIR="${OUT_BASE}/logs"
mkdir -p "$LOGDIR"

# Build job list
JOBS=()

# 1. Normal watermark rollout (one job per task)
for task in "${TASKS[@]}"; do
    cmd="torchrun --nproc_per_node=1 --master_port=\$PORT \
        wan_va/wm/eval_robotwin_watermark.py \
        --robotwin-root ${ROBOTWIN_ROOT} \
        --task-names ${task} \
        --test-num ${TEST_NUM} \
        --map-steps ${MAP_STEPS} \
        --skip-expert-check \
        --out-dir ${OUT_BASE}/normal"
    JOBS+=("normal_${task}|${cmd}")
done

# 2. Plain rollout (beta=0, for baseline comparison)
for task in "${TASKS[@]}"; do
    cmd="torchrun --nproc_per_node=1 --master_port=\$PORT \
        wan_va/wm/eval_robotwin_watermark.py \
        --robotwin-root ${ROBOTWIN_ROOT} \
        --task-names ${task} \
        --test-num ${TEST_NUM} \
        --beta 0.0 \
        --skip-map \
        --skip-expert-check \
        --out-dir ${OUT_BASE}/plain"
    JOBS+=("plain_${task}|${cmd}")
done

# 3. Robustness runs
for setting in "${ROBUST_SETTINGS[@]}"; do
    read -r mode flag value <<< "$setting"
    for task in "${TASKS[@]}"; do
        cmd="torchrun --nproc_per_node=1 --master_port=\$PORT \
            wan_va/wm/eval_robotwin_watermark_robustness.py \
            --robotwin-root ${ROBOTWIN_ROOT} \
            --task-names ${task} \
            --test-num ${TEST_NUM} \
            --map-steps ${MAP_STEPS} \
            --controller-postprocess ${mode} ${flag} ${value} \
            --skip-expert-check \
            --out-dir ${OUT_BASE}/robust"
        tag="${mode}_${value}_${task}"
        JOBS+=("robust_${tag}|${cmd}")
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo "GPUs: ${NUM_GPUS}"
echo "Tasks: ${TASKS[*]}"
echo ""

# --- Dispatch loop ---
# Track PIDs per GPU slot
declare -A GPU_PIDS
declare -A GPU_JOB_NAMES
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=""
    GPU_JOB_NAMES[$g]=""
done

JOB_IDX=0

dispatch_to_gpu() {
    local gpu=$1
    if [ $JOB_IDX -ge ${#JOBS[@]} ]; then
        return 1
    fi

    local job_entry="${JOBS[$JOB_IDX]}"
    local job_name="${job_entry%%|*}"
    local job_cmd="${job_entry#*|}"
    JOB_IDX=$((JOB_IDX + 1))

    # Assign unique master_port
    local port=$((29500 + gpu * 10 + (JOB_IDX % 10)))
    job_cmd="${job_cmd//\$PORT/$port}"

    local logfile="${LOGDIR}/${job_name}.log"
    echo "[$(date +%H:%M:%S)] GPU $gpu <- $job_name (log: $logfile)"

    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=osmesa \
        bash -c "${VENV_ACTIVATE}${job_cmd}" > "$logfile" 2>&1 &

    GPU_PIDS[$gpu]=$!
    GPU_JOB_NAMES[$gpu]="$job_name"
    return 0
}

# Initial dispatch: fill all GPUs
for ((g=0; g<NUM_GPUS; g++)); do
    dispatch_to_gpu $g || true
done

# Wait loop: when a GPU finishes, dispatch next job
while true; do
    all_done=true
    for ((g=0; g<NUM_GPUS; g++)); do
        pid="${GPU_PIDS[$g]}"
        if [ -n "$pid" ]; then
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished
                status=0; wait "$pid" 2>/dev/null || status=$?
                name="${GPU_JOB_NAMES[$g]}"
                if [ $status -eq 0 ]; then
                    echo "[$(date +%H:%M:%S)] GPU $g DONE: $name"
                else
                    echo "[$(date +%H:%M:%S)] GPU $g FAIL($status): $name"
                fi
                GPU_PIDS[$g]=""
                GPU_JOB_NAMES[$g]=""

                # Try dispatch next
                dispatch_to_gpu $g || true
            fi
            if [ -n "${GPU_PIDS[$g]}" ]; then
                all_done=false
            fi
        fi
    done

    if $all_done && [ $JOB_IDX -ge ${#JOBS[@]} ]; then
        break
    fi

    sleep 10
done

echo ""
echo "[$(date +%H:%M:%S)] ALL JOBS COMPLETED"
echo "Results: ${OUT_BASE}"
