#!/usr/bin/env bash
# PATH B detection: roll out each student PLAIN in LIBERO (default: all 10 tasks), base
# mirrors in lockstep + MAP-recovers the seed per selected chunk -> per-episode NPZ. GPU-pooled
# over (student x task) units. Override TASKS_OVERRIDE for narrower diagnostics.
# Env: GPUS (comma list of GPUs to use), NEPS (episodes per task).
set -uo pipefail
LB=/workspace/vla/distill/lingbot
EV=/workspace/vla/lingbot-va
CK=/workspace/vla_out
PY=/usr/bin/python3.11
LOGD=/workspace/vla/ft_logs/detect_pathb
mkdir -p "$LOGD"
cd $EV
export PYTHONPATH=/workspace/vla/openpi/third_party/libero:/workspace/vla/lingbot-va:/workspace/vla/distill
export MUJOCO_GL=osmesa OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
stamp(){ echo "[$(date +%H:%M:%S)] DETECT $*"; }

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=',' read -ra GPULIST <<< "$GPUS"
NGPU=${#GPULIST[@]}
NEPS=${NEPS:-10}        # episodes per task
if [ -n "${TASKS_OVERRIDE:-}" ]; then
  TASKS=($TASKS_OVERRIDE)
else
  TASKS=(0 1 2 3 4 5 6 7 8 9)
fi

# students: "name n_keys"; clean10x10 uses n_keys 20 for the reference (decoy calibration uses the
# SAME bucketing; clean just shouldn't carry the seed -> low Z). We score each wm arm at its own N.
# Override via STUDENTS_OVERRIDE env (space-separated "name:nk" pairs),
# e.g. STUDENTS_OVERRIDE="clean10x10:20 n20_10x10:20".
if [ -n "${STUDENTS_OVERRIDE:-}" ]; then
  STUDENTS=(); for p in $STUDENTS_OVERRIDE; do STUDENTS+=("${p/:/ }"); done
else
  STUDENTS=("clean10x10 20" "n20_10x10 20" "n80_10x10 80")
fi

# build work units: student x task
UNITS=()
for spec in "${STUDENTS[@]}"; do
  set -- $spec; name=$1 nk=$2
  for t in "${TASKS[@]}"; do
    UNITS+=("$name $nk $t")
  done
done
stamp "${#UNITS[@]} units over $NGPU GPUs ($GPUS), NEPS=$NEPS/task"

declare -A GPU_PID
launch() {  # gpu_phys  "name nk task"  port
  local gpu=$1 port=$3; set -- $2; local name=$1 nk=$2 t=$3
  # model_root = student transformer/ + base vae/text_encoder/tokenizer (student saves only the
  # transformer; VA_Server needs a full model root). Built by the model_root symlink step.
  local ck=$CK/student_pathb_$name/model_root
  local sc="--student-ckpt $ck"; [ "$name" = "base" ] && sc=""
  local LOG="$LOGD/${name}_task${t}.log"
  stamp "GPU$gpu <- $name task$t nk=$nk port=$port"
  CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    $PY -m wan_va.wm.eval_pathb_detection $sc \
      --out outputs/pathb_det/$name --suite libero_10 --task-range $t $((t+1)) \
      --n-eps $NEPS --n-keys $nk --map-chunks 6 --map-iters 40 --map-steps 10 \
      > "$LOG" 2>&1 &
  GPU_PID[$gpu]=$!
}

ui=0; port=${PORT_BASE:-29630}   # override when running a 2nd dispatcher concurrently (avoid EADDRINUSE)
while [ $ui -lt ${#UNITS[@]} ]; do
  for gpu in "${GPULIST[@]}"; do
    pid=${GPU_PID[$gpu]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      if [ $ui -lt ${#UNITS[@]} ]; then
        launch "$gpu" "${UNITS[$ui]}" $port; ui=$((ui+1)); port=$((port+1))
      fi
    fi
  done
  sleep 20
done
for gpu in "${!GPU_PID[@]}"; do pid=${GPU_PID[$gpu]:-}; [ -n "$pid" ] && wait "$pid" 2>/dev/null; done

stamp "all detection units done; scoring"
for spec in "${STUDENTS[@]}"; do
  set -- $spec; name=$1
  echo "  $name: $(ls outputs/pathb_det/$name/*.npz 2>/dev/null|wc -l) episode NPZs"
done
stamp "DETECT_PATHB_COMPLETE"
