#!/usr/bin/env bash
# LIBERO-descendant campaign env (source me)
export OPENPI_PP=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src
export LINGBOT_PP=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va
export HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export PY=/usr/bin/python3.11
export LIBERO_PP=/workspace/vla/openpi/third_party/libero
# local overlay: datasets==3.6.0 (lerobot-0.3.3 pin) shadows the shared lingbot_pydeps 4.0.0
# which breaks lerobot's torch.stack(ds["timestamp"]) (4.0.0 returns lazy Column). Use LB_PP for all lingbot cmds.
export LB_PP=/workspace/vla/datasets_overlay:$LINGBOT_PP
