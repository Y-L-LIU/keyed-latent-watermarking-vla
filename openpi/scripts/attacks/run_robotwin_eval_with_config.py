"""Wrapper: inject pi05_aloha_full_base into the shared openpi config dict, then run the robotwin watermark eval (with controller-postprocess attack hooks).

The shared openpi (which is what eval_robotwin_watermark_map.py imports via sys.path.insert) lacks pi05_aloha_full_base; this wrapper monkey-patches it in at runtime so we don't have to edit the shared tree.

All argv after the first arg are forwarded verbatim to the underlying eval script.
"""
from __future__ import annotations

import pathlib
import runpy
import sys

SHARED = pathlib.Path("/workspace/vla/openpi")
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(SHARED / "src"))

from openpi import transforms as _transforms  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.training import config as cfg  # noqa: E402
from openpi.training import weight_loaders  # noqa: E402
from openpi.training.config import (  # noqa: E402
    DataConfig,
    LeRobotAlohaDataConfig,
    TrainConfig,
)

NAME = "pi05_aloha_full_base"

if NAME not in cfg._CONFIGS_DICT:
    cfg._CONFIGS_DICT[NAME] = TrainConfig(
        name=NAME,
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/robotwin10_clean",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/workspace/scratch/anon/models/pi05_base/params"
        ),
        num_train_steps=20_000,
        batch_size=16,
        fsdp_devices=8,
    )

TARGET = str(SHARED / "scripts" / "eval_robotwin_watermark_map_robustness.py")
sys.argv = [TARGET] + sys.argv[1:]
runpy.run_path(TARGET, run_name="__main__")
