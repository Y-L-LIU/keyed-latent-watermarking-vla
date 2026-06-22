#!/usr/bin/env bash
set -uo pipefail
BASE=/workspace/vla/models/lingbot-va-posttrain-libero-long
CKR=/workspace/vla_out/student_pathb_n80_10x10/checkpoints/checkpoint_step_1500
MR=/workspace/vla_out/student_pathb_n80_10x10/model_root
stamp(){ echo "[$(date +%H:%M:%S)] N80CHAIN $*"; }
# wait for n80 ckpt
while [ ! -d "$CKR/transformer" ]; do sleep 60; done
sleep 20
stamp "n80 ckpt ready; building model_root"
rm -rf "$MR"; mkdir -p "$MR"
ln -sf "$CKR/transformer" "$MR/transformer"
for sub in vae text_encoder tokenizer assets; do ln -sf "$BASE/$sub" "$MR/$sub"; done
stamp "launching n80_10x10 detection on gpu0-3"
cd /workspace/vla/distill/lingbot
GPUS=0,1,2,3 NEPS=8 STUDENTS_OVERRIDE="n80_10x10:80" \
  bash dispatch_detect_pathb.sh > /workspace/vla/ft_logs/detect_n80_dispatch.log 2>&1
stamp "N80_DETECT_DONE"
