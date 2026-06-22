# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_lora_train_cfg import va_libero_lora_train_cfg
from .va_robotwin_lora_train_cfg import va_robotwin_lora_train_cfg
from .va_libero_goalspatial_lora_train_cfg import va_libero_goalspatial_lora_train_cfg
from .va_robotwin_setB_lora_train_cfg import va_robotwin_setB_lora_train_cfg
from .va_robotwin_setC_lora_train_cfg import va_robotwin_setC_lora_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_libero_descendant_cfg import va_libero_descendant_cfg
from .va_compression_descendants_cfg import (
    va_libero_descendant_prune_cfg,
    va_libero_descendant_quant_cfg,
    va_robotwin_descendant_cfg,
    va_robotwin_descendant_prune_cfg,
    va_robotwin_descendant_quant_cfg,
)

# Attack-D evaluation: a libero descendant-style eval config whose transformer is an
# adversarially fine-tuned (Attack-D) suspect model dir, taken from $WAN_ATTACKED_MODEL_DIR.
# Lets eval_libero_watermark.py score any saved attacked checkpoint without a new file.
import os as _os
import copy as _copy
va_libero_attacked_cfg = _copy.deepcopy(va_libero_descendant_cfg)
_attacked_dir = _os.environ.get("WAN_ATTACKED_MODEL_DIR", "")
if _attacked_dir:
    va_libero_attacked_cfg.wan22_pretrained_model_name_or_path = _attacked_dir
va_libero_attacked_cfg.save_root = _os.environ.get(
    "WAN_ATTACKED_SAVE_ROOT", "/workspace/vla/eval_out/lingbot_attackd/server_out")
# 80GB cards: keep VAE + text_encoder ON-GPU (no CPU offload) so per-chunk obs encoding is
# GPU-fast and parallel evals don't thrash the CPU. (offload was the throughput killer.)
va_libero_attacked_cfg.enable_offload = False

va_robotwin_attacked_cfg = _copy.deepcopy(va_robotwin_descendant_cfg)
_rt_attacked_dir = _os.environ.get("WAN_ATTACKED_MODEL_DIR", "")
if _rt_attacked_dir:
    va_robotwin_attacked_cfg.wan22_pretrained_model_name_or_path = _rt_attacked_dir
va_robotwin_attacked_cfg.save_root = _os.environ.get(
    "WAN_ATTACKED_SAVE_ROOT", "/workspace/vla_out/attack_c/eval_out/lingbot_rt_attackd/server_out")
va_robotwin_attacked_cfg.enable_offload = False

VA_CONFIGS = {
    'libero_attacked': va_libero_attacked_cfg,
    'robotwin_attacked': va_robotwin_attacked_cfg,
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_lora_train': va_libero_lora_train_cfg,
    'robotwin_lora_train': va_robotwin_lora_train_cfg,
    'libero_goalspatial_lora_train': va_libero_goalspatial_lora_train_cfg,
    'robotwin_setB_lora_train': va_robotwin_setB_lora_train_cfg,
    'robotwin_setC_lora_train': va_robotwin_setC_lora_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'libero_descendant': va_libero_descendant_cfg,
    'libero_descendant_prune': va_libero_descendant_prune_cfg,
    'libero_descendant_quant': va_libero_descendant_quant_cfg,
    'robotwin_descendant': va_robotwin_descendant_cfg,
    'robotwin_descendant_prune': va_robotwin_descendant_prune_cfg,
    'robotwin_descendant_quant': va_robotwin_descendant_quant_cfg,
}
