#!/usr/bin/env bash
# PATH B relabel corpus generation, GPU-pooled across all 8 H100s.
# 3 arms (clean beta=0 / n40 / n160 beta=1) x 10 task-ranges (20 eps each) = 30 work units.
# Each unit = relabel_pathb.py over one task's first 20 episodes, writing into a pre-built
# skeleton (symlinked latents/videos, only action column rewritten). ~20 eps x ~76s ~= 25min/unit.
set -uo pipefail
LB=/workspace/vla/distill/lingbot
LAT=/workspace/vla/lingbot_latents
PY=/usr/bin/python3.11
LOGD=/workspace/vla/ft_logs/relabel_pathb
mkdir -p "$LOGD"
export PYTHONPATH=/workspace/vla/openpi/third_party/libero:/workspace/vla/lingbot-va:/workspace/vla/distill
export MUJOCO_GL=osmesa
# cap per-process CPU threads: 8 GPUs x 10 threads = 80 < 96 cores (the CPU VAE encode under
# enable_offload + default all-core thread pools was thrashing -> load avg 270, ~1 chunk/min).
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export OFFLOAD=0   # VAE+text_encoder on GPU (one proc/GPU, ~46G) -> no CPU VAE contention
stamp(){ echo "[$(date +%H:%M:%S)] DISPATCH $*"; }

# arm spec: "name n_keys beta"
ARMS=("clean 40 0.0" "n40 40 1.0" "n160 160 1.0")
TASK_STARTS=(0 50 100 150 200 250 300 350 400 450)
NPER=50   # full 50 eps/task = all 500 episodes (meta lists all 500; latents symlinked -> data must match)
NGPU=8

# 1) build skeletons once (CPU only)
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1
  OUT=$LAT/relabel_pathb_$arm
  stamp "skeleton $arm -> $OUT"
  $PY $LB/relabel_pathb.py --out "$OUT" --n-keys 1 --beta 0 --skeleton-only \
      > "$LOGD/skeleton_$arm.log" 2>&1 || { stamp "skeleton $arm FAILED"; tail -5 "$LOGD/skeleton_$arm.log"; exit 1; }
done

# 2) build the work-unit list
UNITS=()
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1; nk=$2; beta=$3
  for st in "${TASK_STARTS[@]}"; do
    UNITS+=("$arm $nk $beta $st")
  done
done
stamp "total work units: ${#UNITS[@]} across $NGPU GPUs"

# 3) GPU-pooled dispatch
declare -A GPU_PID
launch_unit() {  # gpu  "arm nk beta st"  uniqport
  local gpu=$1 uport=$3
  set -- $2; local arm=$1 nk=$2 beta=$3 st=$4
  local en=$((st + NPER))
  local OUT=$LAT/relabel_pathb_$arm
  local LOG="$LOGD/${arm}_ep${st}_${en}.log"
  stamp "GPU$gpu <- $arm ep[$st,$en) nk=$nk beta=$beta port=$uport"
  CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$uport \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    $PY $LB/relabel_pathb.py --out "$OUT" --n-keys "$nk" --beta "$beta" \
      --ep-range "$st" "$en" --no-skeleton > "$LOG" 2>&1 &
  GPU_PID[$gpu]=$!
}

ui=0
while [ $ui -lt ${#UNITS[@]} ]; do
  for gpu in $(seq 0 $((NGPU-1))); do
    pid=${GPU_PID[$gpu]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      if [ $ui -lt ${#UNITS[@]} ]; then
        launch_unit "$gpu" "${UNITS[$ui]}" $((29550 + ui))   # unique port per unit (no reuse race)
        ui=$((ui+1))
      fi
    fi
  done
  sleep 20
done

# 4) wait for the last batch
stamp "all units dispatched; waiting for stragglers"
for gpu in "${!GPU_PID[@]}"; do
  pid=${GPU_PID[$gpu]:-}
  [ -n "$pid" ] && wait "$pid" 2>/dev/null
done

# 5) report counts
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1
  n=$(ls $LAT/relabel_pathb_$arm/data/chunk-000/*.parquet 2>/dev/null | wc -l)
  stamp "arm $arm: $n parquets written"
done
stamp "RELABEL_PATHB_DISPATCH_COMPLETE"
