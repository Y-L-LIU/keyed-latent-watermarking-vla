# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# LoRA fine-tune config: LingBot-VA LIBERO descendant trained on libero_goal + libero_spatial,
# evaluated on libero_10. New held-out-task design (2026-06-05). Mirrors va_libero_lora_train_cfg
# except the dataset is the goal+spatial latents (extracted via wan_va/tools/extract_latents.py and
# merged into one latent dataset at lingbot_latents/libero_goalspatial).
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_goalspatial_lora_train_cfg = EasyDict(__name__='Config: VA libero goal+spatial LoRA train')
va_libero_goalspatial_lora_train_cfg.update(va_libero_cfg)

va_libero_goalspatial_lora_train_cfg.wan22_pretrained_model_name_or_path = "/workspace/vla/models/lingbot-va-posttrain-libero-long"
va_libero_goalspatial_lora_train_cfg.save_root = '/workspace/vla/lingbot_out/libero_goalspatial_lora'
va_libero_goalspatial_lora_train_cfg.dataset_path = '/workspace/vla/lingbot_latents/libero_goalspatial'
va_libero_goalspatial_lora_train_cfg.empty_emb_path = os.path.join(va_libero_goalspatial_lora_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_goalspatial_lora_train_cfg.enable_wandb = False
va_libero_goalspatial_lora_train_cfg.load_worker = 16
va_libero_goalspatial_lora_train_cfg.save_interval = 1000
va_libero_goalspatial_lora_train_cfg.gc_interval = 50
va_libero_goalspatial_lora_train_cfg.cfg_prob = 0.1

# LoRA
va_libero_goalspatial_lora_train_cfg.use_lora = True
va_libero_goalspatial_lora_train_cfg.lora_rank = 16
va_libero_goalspatial_lora_train_cfg.lora_alpha = 16
va_libero_goalspatial_lora_train_cfg.lora_dropout = 0.0
va_libero_goalspatial_lora_train_cfg.lora_include_ffn = False

# Training parameters (light descendant)
va_libero_goalspatial_lora_train_cfg.learning_rate = 1e-4
va_libero_goalspatial_lora_train_cfg.beta1 = 0.9
va_libero_goalspatial_lora_train_cfg.beta2 = 0.95
va_libero_goalspatial_lora_train_cfg.weight_decay = 1e-1
va_libero_goalspatial_lora_train_cfg.warmup_steps = 50
va_libero_goalspatial_lora_train_cfg.batch_size = 1
va_libero_goalspatial_lora_train_cfg.gradient_accumulation_steps = 1
va_libero_goalspatial_lora_train_cfg.num_steps = 2000
