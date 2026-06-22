#!/usr/bin/env bash
# Launch one LoRA student train on the relabeled corpus. GPUs 4-7 (0-1 busy, leave 2-3 spare).
# Args via env: ARM (dc|hash). Derives DATASET_PATH / SAVE_ROOT.
set -uo pipefail
cd /workspace/vla/lingbot-va
export PYTHONPATH=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/lingbot_latents
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM=${ARM:?set ARM=dc|hash}
export DATASET_PATH=${DATASET_PATH:-/workspace/vla/lingbot_latents/relabel_${ARM}}
export SAVE_ROOT=${SAVE_ROOT:-/workspace/vla/lingbot_out/distill_ckpt/student_${ARM}}
export NUM_STEPS=${NUM_STEPS:-1500}
export BASE_CKPT=/workspace/vla/models/lingbot-va-posttrain-libero-long
PY=/usr/bin/python3.11
mkdir -p /workspace/vla/ft_logs
LOG=/workspace/vla/ft_logs/train_student_${ARM}.log
GPUS=${GPUS:-4,5,6,7}
NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MPORT=${MPORT:-29588}
LHPORT=${LHPORT:-29610}
echo "[$(date +%H:%M:%S)] student ${ARM} train start (gpu=${GPUS}) steps=${NUM_STEPS} -> ${SAVE_ROOT}" | tee $LOG
CUDA_VISIBLE_DEVICES=$GPUS TORCHFT_LIGHTHOUSE=http://localhost:$LHPORT $PY -m torch.distributed.run \
  --nproc_per_node=$NPROC --local-ranks-filter=0 --master_port $MPORT --tee 3 \
  /workspace/vla/distill/lingbot/train_student.py >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] student ${ARM} train exit=$?" | tee -a $LOG
echo "STUDENT_${ARM}_DONE" | tee -a $LOG
