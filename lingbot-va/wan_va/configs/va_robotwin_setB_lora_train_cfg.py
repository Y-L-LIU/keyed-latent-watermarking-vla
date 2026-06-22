# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# LoRA fine-tune config: LingBot-VA RoboTwin descendant STAGE-1, trained on held-out task set B
# (10 tasks disjoint from the eval-10 and bbh), evaluated on the eval-10. New held-out-task design
# (2026-06-05). Base = the kept RoboTwin posttrain. Latents extracted from the sim-gen'd set-B
# lerobot dataset via wan_va/tools/extract_latents.py (base-model = posttrain-robotwin for the VAE).
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_setB_lora_train_cfg = EasyDict(__name__='Config: VA robotwin setB LoRA train (stage1)')
va_robotwin_setB_lora_train_cfg.update(va_robotwin_cfg)

va_robotwin_setB_lora_train_cfg.wan22_pretrained_model_name_or_path = "/workspace/vla/models/lingbot-va-posttrain-robotwin"
va_robotwin_setB_lora_train_cfg.save_root = '/workspace/vla/lingbot_out/robotwin_setB_lora'
va_robotwin_setB_lora_train_cfg.dataset_path = '/workspace/vla/lingbot_latents/robotwin_setB'
va_robotwin_setB_lora_train_cfg.empty_emb_path = os.path.join(va_robotwin_setB_lora_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_setB_lora_train_cfg.enable_wandb = False
va_robotwin_setB_lora_train_cfg.load_worker = 16
va_robotwin_setB_lora_train_cfg.save_interval = 1000
va_robotwin_setB_lora_train_cfg.gc_interval = 50
va_robotwin_setB_lora_train_cfg.cfg_prob = 0.1

# LoRA
va_robotwin_setB_lora_train_cfg.use_lora = True
va_robotwin_setB_lora_train_cfg.lora_rank = 16
va_robotwin_setB_lora_train_cfg.lora_alpha = 16
va_robotwin_setB_lora_train_cfg.lora_dropout = 0.0
va_robotwin_setB_lora_train_cfg.lora_include_ffn = False

# Training parameters (light descendant)
va_robotwin_setB_lora_train_cfg.learning_rate = 1e-4
va_robotwin_setB_lora_train_cfg.beta1 = 0.9
va_robotwin_setB_lora_train_cfg.beta2 = 0.95
va_robotwin_setB_lora_train_cfg.weight_decay = 0.1
va_robotwin_setB_lora_train_cfg.warmup_steps = 50
va_robotwin_setB_lora_train_cfg.batch_size = 1
va_robotwin_setB_lora_train_cfg.gradient_accumulation_steps = 1
va_robotwin_setB_lora_train_cfg.num_steps = 2000
