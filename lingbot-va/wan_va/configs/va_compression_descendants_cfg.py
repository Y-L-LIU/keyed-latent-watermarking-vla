# Threat-model §12.5 (pruning / quantization) — eval configs for the compressed
# LIBERO and RoboTwin descendants. Each one is a clone of the corresponding base
# eval cfg with `wan22_pretrained_model_name_or_path` redirected at a descendant
# model dir whose `transformer/` was rewritten by
# wan_va/attacks/build_compressed_transformer.py.
from easydict import EasyDict

from .va_libero_cfg import va_libero_cfg
from .va_robotwin_cfg import va_robotwin_cfg


def _clone(src, name, model_dir, save_root):
    cfg = EasyDict(__name__=name)
    cfg.update(src)
    cfg.wan22_pretrained_model_name_or_path = model_dir
    cfg.save_root = save_root
    return cfg


va_libero_descendant_prune_cfg = _clone(
    va_libero_cfg,
    "Config: VA libero descendant prune30 eval",
    "/workspace/vla/models/lingbot-descendant-libero-prune30",
    "/workspace/vla/eval_out_compression/lingbot_libero10_prune30/server_out",
)
va_libero_descendant_quant_cfg = _clone(
    va_libero_cfg,
    "Config: VA libero descendant int8 eval",
    "/workspace/vla/models/lingbot-descendant-libero-quant",
    "/workspace/vla/eval_out_compression/lingbot_libero10_quant/server_out",
)
va_robotwin_descendant_prune_cfg = _clone(
    va_robotwin_cfg,
    "Config: VA robotwin descendant prune30 eval",
    "/workspace/vla/models/lingbot-descendant-robotwin-prune30",
    "/workspace/vla/eval_out_compression/lingbot_robotwin_prune30/server_out",
)
va_robotwin_descendant_quant_cfg = _clone(
    va_robotwin_cfg,
    "Config: VA robotwin descendant int8 eval",
    "/workspace/vla/models/lingbot-descendant-robotwin-quant",
    "/workspace/vla/eval_out_compression/lingbot_robotwin_quant/server_out",
)

# RoboTwin needs a robotwin-base override matching where the LoRA-merged transformer
# was assembled (HANDOFF.md). We also expose a vanilla 'robotwin_descendant' that the
# eval can compare against — same as compression configs but with the unmodified
# descendant transformer.
va_robotwin_descendant_cfg = _clone(
    va_robotwin_cfg,
    "Config: VA robotwin descendant eval (no attack)",
    "/workspace/vla/models/lingbot-descendant-robotwin",
    "/workspace/vla/eval_out_compression/lingbot_robotwin_descendant/server_out",
)
