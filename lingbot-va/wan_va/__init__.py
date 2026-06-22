# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# The cuDNN SDPA backend returns NaN gradients on torch 2.9 + cuDNN 9.10, which
# breaks the gradient-based watermark MAP inversion (forward inference is fine).
# Disable it so scaled_dot_product_attention uses the flash/mem-efficient/math
# backends, which have correct backward passes.
try:
    import torch as _torch
    _torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass

from . import configs, distributed, modules