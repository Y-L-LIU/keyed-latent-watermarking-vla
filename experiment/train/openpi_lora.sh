#!/usr/bin/env bash
# §6.4 TRAIN (openpi pi0.5): LoRA "descendant/suspect" fine-tune (feeds tab_descendant).
# Verifier later uses the UNMODIFIED base detector -> tests watermark survival of FT.
#   arg1 = libero | robotwin   (default libero)
#
# libero   -> run_openpi_ft.sh : 2 parallel 4-GPU jobs
#               configs  pi05_libero_goal_lora_from_libero  (GPU 0-3)
#                        pi05_libero_spatial_lora_from_libero (GPU 4-7)
#               exp      descendant_lora  (runs compute_norm_stats first)
#               out      openpi-checkpoints/<cfg>/descendant_lora/<step>/
# robotwin -> run_robotwin_ft.sh : 1 job, 8 GPU
#               config   pi05_aloha_robotwin_lora_local
#               exp      robotwin_descendant
#               needs    HF_LEROBOT_HOME=robotwin2_train/lerobot_home (set in launcher)
#               out      openpi-checkpoints/pi05_aloha_robotwin_lora_local/robotwin_descendant/<step>/
# logs: ft_logs/train_*.log
set -euo pipefail; source "$(dirname "$0")/../env.sh"
case "${1:-libero}" in
  libero)   exec bash "$VLA/run_openpi_ft.sh" ;;
  robotwin) exec bash "$VLA/run_robotwin_ft.sh" ;;
  *) echo "arg1 must be libero|robotwin"; exit 2 ;;
esac
