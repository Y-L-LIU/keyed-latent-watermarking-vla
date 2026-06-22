#!/usr/bin/env bash
# PATH B student training: 3 LoRA students (clean / n40 / n160) by BC on the relabeled corpora.
# Runs 2 concurrently on disjoint GPU sets (0-3 and 4-7), then the 3rd. Students KEPT under
# /workspace/vla_out (re-detection is cheap; only changing injection needs retrain). Waits for the
# relabel corpus to finish first.
set -uo pipefail
LB=/workspace/vla/distill/lingbot
LAT=/workspace/vla/lingbot_latents
CK=/workspace/vla_out
stamp(){ echo "[$(date +%H:%M:%S)] TRAIN-DISPATCH $*"; }

# wait for corpus
while ! grep -q RELABEL_PATHB_DISPATCH_COMPLETE /workspace/vla/ft_logs/relabel_pathb_dispatch.log 2>/dev/null; do
  sleep 60
done
stamp "corpus complete; verifying parquet counts"
for arm in clean n40 n160; do
  n=$(ls $LAT/relabel_pathb_$arm/data/chunk-000/*.parquet 2>/dev/null | wc -l)
  stamp "  relabel_pathb_$arm: $n parquets"
  [ "$n" -lt 480 ] && { stamp "ABORT: $arm has too few parquets ($n, expected ~500)"; exit 1; }
done

run_one() {  # arm gpus mport lhport (blocks)
  local arm=$1 gpus=$2 mport=$3 lhport=$4
  ARM=pathb_$arm GPUS=$gpus MPORT=$mport LHPORT=$lhport \
    DATASET_PATH=$LAT/relabel_pathb_$arm SAVE_ROOT=$CK/student_pathb_$arm NUM_STEPS=1500 \
    bash $LB/launch_train.sh
}

# batch 1: clean (0-3) + n40 (4-7) concurrent
stamp "batch1: clean(gpu0-3) + n40(gpu4-7)"
run_one clean 0,1,2,3 29588 29610 &
P1=$!
run_one n40 4,5,6,7 29688 29710 &
P2=$!
wait $P1; stamp "clean train exit"
wait $P2; stamp "n40 train exit"

# batch 2: n160 (0-3)
stamp "batch2: n160(gpu0-3)"
run_one n160 0,1,2,3 29588 29610
stamp "n160 train exit"

for arm in clean n40 n160; do
  d=$CK/student_pathb_$arm/checkpoints/checkpoint_step_1500/transformer
  if [ -d "$d" ]; then stamp "  student $arm: ckpt OK ($d)"; else stamp "  student $arm: CKPT MISSING"; fi
done
stamp "TRAIN_PATHB_DISPATCH_COMPLETE"
