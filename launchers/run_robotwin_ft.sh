#!/usr/bin/env bash
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=disabled
PY=/usr/bin/python3.11
EXP=robotwin_descendant
echo "[$(date +%H:%M:%S)] TRAIN robotwin lora start (8 GPU)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m scripts.train pi05_aloha_robotwin_lora_local \
  --exp-name=$EXP --overwrite > /workspace/vla/ft_logs/train_robotwin.log 2>&1
echo "[$(date +%H:%M:%S)] TRAIN robotwin exit=$?"
echo "ROBOTWIN_FT_COMPLETE"
