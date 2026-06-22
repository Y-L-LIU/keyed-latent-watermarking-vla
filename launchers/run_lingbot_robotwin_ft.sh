#!/usr/bin/env bash
# LingBot-VA LoRA fine-tune on RoboTwin (beat_block_hammer) — descendant for the watermark scenario.
# Uses the cloned lerobot 0.3.3 + datasets 4.0 overlay via PYTHONPATH (no env churn). GPU 0-3.
set -uo pipefail
cd /workspace/vla/lingbot-va
export PYTHONPATH=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/lingbot_latents
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.11
echo "[$(date +%H:%M:%S)] LINGBOT robotwin_lora_train start (gpu0-3)"
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCHFT_LIGHTHOUSE=http://localhost:29620 $PY -m torch.distributed.run \
  --nproc_per_node=4 --local-ranks-filter=0 --master_port 29588 --tee 3 \
  -m wan_va.train --config-name robotwin_lora_train > /workspace/vla/ft_logs/train_lingbot_robotwin.log 2>&1
echo "[$(date +%H:%M:%S)] LINGBOT robotwin train exit=$?"; echo "LINGBOT_ROBOTWIN_FT_COMPLETE"
