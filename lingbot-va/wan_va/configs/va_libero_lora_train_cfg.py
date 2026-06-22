# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# LoRA fine-tune config for LingBot-VA on LIBERO (suspect/descendant model for the
# watermark fine-tuned-descendant scenario). Derived from va_libero_train_cfg with:
#   * use_lora=True (attention to_q/k/v/to_out adapters; base frozen, merged on save)
#   * local model path (this box maps /workspace -> /workspace; /data_sde was a foreign mount)
#   * wandb disabled for unattended runs
#   * light step budget
# NOTE: dataset_path must point at a LingBot latent dataset (latents/*.pth + meta/episodes.jsonl
#       + empty_emb.pt). The original LIBERO latent data lived on a cloud mount that is not
#       present on this box, and the repo ships no latent-extraction script, so this config is
#       launch-ready but BLOCKED until that dataset is provided/rebuilt.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_lora_train_cfg = EasyDict(__name__='Config: VA libero LoRA train')
va_libero_lora_train_cfg.update(va_libero_cfg)

# Local base checkpoint (was /workspace/models/models/... on the original training box).
va_libero_lora_train_cfg.wan22_pretrained_model_name_or_path = "/workspace/vla/models/lingbot-va-posttrain-libero-long"

va_libero_lora_train_cfg.save_root = '/workspace/vla/lingbot_out/libero_lora'
va_libero_lora_train_cfg.dataset_path = '/workspace/vla/lingbot_latents/libero_long'
va_libero_lora_train_cfg.empty_emb_path = os.path.join(va_libero_lora_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_lora_train_cfg.enable_wandb = False
va_libero_lora_train_cfg.load_worker = 16
va_libero_lora_train_cfg.save_interval = 1000  # each ckpt ~10GB; keep it lean
va_libero_lora_train_cfg.gc_interval = 50
va_libero_lora_train_cfg.cfg_prob = 0.1

# LoRA
va_libero_lora_train_cfg.use_lora = True
va_libero_lora_train_cfg.lora_rank = 16
va_libero_lora_train_cfg.lora_alpha = 16
va_libero_lora_train_cfg.lora_dropout = 0.0
va_libero_lora_train_cfg.lora_include_ffn = False

# Training parameters (light descendant)
va_libero_lora_train_cfg.learning_rate = 1e-4   # LoRA tolerates a higher LR than full FT
va_libero_lora_train_cfg.beta1 = 0.9
va_libero_lora_train_cfg.beta2 = 0.95
va_libero_lora_train_cfg.weight_decay = 1e-1
va_libero_lora_train_cfg.warmup_steps = 50
va_libero_lora_train_cfg.batch_size = 1
va_libero_lora_train_cfg.gradient_accumulation_steps = 1   # grad_accum=10 was ~13h; =1 gives ~1.25s/it
va_libero_lora_train_cfg.num_steps = 2000   # align with openpi descendants (~2000-2500 steps)
