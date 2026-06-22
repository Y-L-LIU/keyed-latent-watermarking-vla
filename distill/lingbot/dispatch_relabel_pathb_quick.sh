#!/usr/bin/env bash
# QUICK PATH B relabel: 2 watermarked arms (n20, n80) over the 10-task x 10-episode
# reindexed corpus. clean control = relabel_pathb_clean10x10 (original demos). 8 units
# = 2 arms x 4 position-shards, one per GPU, single batch. Outputs are named
# *_10x10 to preserve the older 2-task null-result artifacts.
set -uo pipefail
LB=/workspace/vla/distill/lingbot
LAT=/workspace/vla/lingbot_latents
PY=/usr/bin/python3.11
LOGD=/workspace/vla/ft_logs/relabel_pathb_quick
mkdir -p "$LOGD"
export PYTHONPATH=/workspace/vla/openpi/third_party/libero:/workspace/vla/lingbot-va:/workspace/vla/distill
export MUJOCO_GL=osmesa OFFLOAD=0 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
stamp(){ echo "[$(date +%H:%M:%S)] QUICK $*"; }

ARMS=("n20_10x10 20" "n80_10x10 80")
SHARDS=(0 25 50 75)   # pos-range starts; each 25 positions -> [0,25)(25,50)(50,75)(75,100)
SHLEN=25

# skeletons (10x10 sparse original episodes, re-numbered to contiguous positions)
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1
  stamp "skeleton $arm"
  $PY $LB/relabel_pathb.py --out $LAT/relabel_pathb_$arm --n-keys 1 --beta 0 \
      --skeleton-only --reindex > "$LOGD/skeleton_$arm.log" 2>&1 \
    || { stamp "skeleton $arm FAILED"; tail -5 "$LOGD/skeleton_$arm.log"; exit 1; }
done

# launch 8 units (n20 on gpu0-3, n80 on gpu4-7)
gpu=0; port=29600
PIDS=()
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1 nk=$2
  for st in "${SHARDS[@]}"; do
    en=$((st + SHLEN))
    LOG="$LOGD/${arm}_ep${st}_${en}.log"
    stamp "GPU$gpu <- $arm pos[$st,$en) nk=$nk port=$port"
    CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      $PY $LB/relabel_pathb.py --out $LAT/relabel_pathb_$arm --n-keys $nk --beta 1.0 \
        --reindex --pos-range $st $en --no-skeleton > "$LOG" 2>&1 &
    PIDS+=($!)
    gpu=$((gpu+1)); port=$((port+1))
  done
done

stamp "waiting for ${#PIDS[@]} units"
for p in "${PIDS[@]}"; do wait "$p"; done

for arm in clean10x10; do
  n=$(ls $LAT/relabel_pathb_$arm/data/chunk-000/*.parquet 2>/dev/null | wc -l)
  stamp "arm $arm: $n / 100 parquets"
done
for spec in "${ARMS[@]}"; do
  set -- $spec; arm=$1
  n=$(ls $LAT/relabel_pathb_$arm/data/chunk-000/*.parquet 2>/dev/null | wc -l)
  stamp "arm $arm: $n / 100 parquets"
done
stamp "QUICK_RELABEL_COMPLETE"
