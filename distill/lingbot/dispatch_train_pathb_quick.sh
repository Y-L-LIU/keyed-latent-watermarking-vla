#!/usr/bin/env bash
# QUICK PATH B training: 3 LoRA students on the 10-task x 10-episode corpora —
# clean10x10 (original demos), n20_10x10, n80_10x10. 2 concurrent
# (gpu0-3 + 4-7), then the 3rd. Students KEPT under /workspace/vla_out.
# Waits for the quick 10x10 relabel to finish.
set -uo pipefail
LB=/workspace/vla/distill/lingbot
LAT=/workspace/vla/lingbot_latents
CK=/workspace/vla_out
BASE=/workspace/vla/models/lingbot-va-posttrain-libero-long
stamp(){ echo "[$(date +%H:%M:%S)] QTRAIN $*"; }

while ! grep -q QUICK_RELABEL_COMPLETE /workspace/vla/ft_logs/relabel_pathb_10x10_dispatch.log 2>/dev/null; do sleep 30; done
stamp "relabel done; verifying"
for arm in clean10x10 n20_10x10 n80_10x10; do
  n=$(ls $LAT/relabel_pathb_$arm/data/chunk-000/*.parquet 2>/dev/null | wc -l)
  stamp "  relabel_pathb_$arm: $n parquets"
  [ "$n" -lt 95 ] && { stamp "ABORT: $arm has $n (<95)"; exit 1; }
done

run_one() {  # arm gpus mport lhport
  local arm=$1 gpus=$2 mport=$3 lhport=$4
  ARM=pathb_$arm GPUS=$gpus MPORT=$mport LHPORT=$lhport \
    DATASET_PATH=$LAT/relabel_pathb_$arm SAVE_ROOT=$CK/student_pathb_$arm NUM_STEPS=1500 \
    bash $LB/launch_train.sh
}

stamp "batch1: clean10x10(gpu0-3) + n20_10x10(gpu4-7)"
run_one clean10x10 0,1,2,3 29588 29610 & P1=$!
run_one n20_10x10 4,5,6,7 29688 29710 & P2=$!
wait $P1; stamp "clean10x10 exit"
wait $P2; stamp "n20_10x10 exit"

stamp "batch2: n80_10x10(gpu0-3)"
run_one n80_10x10 0,1,2,3 29588 29610
stamp "n80_10x10 exit"

for arm in clean10x10 n20_10x10 n80_10x10; do
  d=$CK/student_pathb_$arm/checkpoints/checkpoint_step_1500/transformer
  [ -d "$d" ] && stamp "  student $arm: ckpt OK" || stamp "  student $arm: CKPT MISSING"
done

stamp "building model_roots"
for arm in clean10x10 n20_10x10 n80_10x10; do
  CKR=$CK/student_pathb_$arm/checkpoints/checkpoint_step_1500
  MR=$CK/student_pathb_$arm/model_root
  if [ ! -d "$CKR/transformer" ]; then
    stamp "  skip $arm: missing transformer"
    continue
  fi
  rm -rf "$MR"; mkdir -p "$MR"
  ln -sf "$CKR/transformer" "$MR/transformer"
  for sub in vae text_encoder tokenizer assets; do ln -sf "$BASE/$sub" "$MR/$sub"; done
  stamp "  model_root OK: $MR"
done
stamp "QUICK_TRAIN_COMPLETE"
