# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Eval config for the LoRA fine-tuned LIBERO descendant: same as va_libero_cfg but the
# transformer points at the merged descendant checkpoint (assembled dir = base vae/
# text_encoder/tokenizer/assets + descendant transformer/).
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg

va_libero_descendant_cfg = EasyDict(__name__='Config: VA libero descendant eval')
va_libero_descendant_cfg.update(va_libero_cfg)
va_libero_descendant_cfg.wan22_pretrained_model_name_or_path = "/workspace/vla/models/lingbot-descendant-libero"
va_libero_descendant_cfg.save_root = "/workspace/vla/eval_out/lingbot_libero10_descendant/server_out"
