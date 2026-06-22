#!/usr/bin/env bash
# §6.4 TRAIN (lingbot Wan): LoRA descendant fine-tune (feeds tab_descendant lingbot rows).
#   arg1 = libero | robotwin   (default libero)
#
# libero   -> run_lingbot_ft.sh : config libero_lora_train,  GPU 4-7 (torchrun 4-proc)
#               out  lingbot_out/libero_lora/checkpoints/checkpoint_step_*/
# robotwin -> run_lingbot_robotwin_ft.sh : config robotwin_lora_train (beat_block_hammer), GPU 0-3
#               out  lingbot_out/robotwin_lora/checkpoints/checkpoint_step_*/
# Both use the lerobot-0.3.3 + datasets-4.0 PYTHONPATH overlay (set in launcher); merge-on-save.
# logs: ft_logs/train_lingbot_*.log
set -euo pipefail; source "$(dirname "$0")/../env.sh"
case "${1:-libero}" in
  libero)   exec bash "$VLA/run_lingbot_ft.sh" ;;
  robotwin) exec bash "$VLA/run_lingbot_robotwin_ft.sh" ;;
  *) echo "arg1 must be libero|robotwin"; exit 2 ;;
esac
