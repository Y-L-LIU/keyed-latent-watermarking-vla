#!/usr/bin/env bash
# LingBot-VA LoRA fine-tune on LIBERO-Long (libero_10) — descendant for the watermark scenario.
# Uses the cloned lerobot 0.3.3 (src on PYTHONPATH, ahead of /usr's openpi lerobot) — no new env.
set -uo pipefail
cd /workspace/vla/lingbot-va
# process-local overlay: datasets 4.0.0 (dataset uses the 'List' feature type) + lerobot 0.3.3 clone,
# both ahead of /usr so they shadow openpi's old lerobot/datasets without touching /usr.
export PYTHONPATH=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/lingbot_latents
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.11
echo "[$(date +%H:%M:%S)] LINGBOT libero_lora_train start (gpu4-7)"
CUDA_VISIBLE_DEVICES=4,5,6,7 TORCHFT_LIGHTHOUSE=http://localhost:29610 $PY -m torch.distributed.run \
  --nproc_per_node=4 --local-ranks-filter=0 --master_port 29577 --tee 3 \
  -m wan_va.train --config-name libero_lora_train > /workspace/vla/ft_logs/train_lingbot_libero.log 2>&1
echo "[$(date +%H:%M:%S)] LINGBOT train exit=$?"; echo "LINGBOT_FT_COMPLETE"
