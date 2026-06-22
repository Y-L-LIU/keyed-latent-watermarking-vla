import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import robotwin_pi05_prepare


def test_robotwin_pi05_config_uses_data_sdh_defaults():
    config = _config.get_config("pi05_aloha_robotwin_full")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert data_config.repo_id == "robotwin/pi05_aloha"
    assert data_config.prompt_from_task is True
    assert str(config.assets_dirs).startswith("/data_sdh/")
    assert str(config.checkpoint_base_dir).startswith("/data_sdh/")
    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"


def test_prepare_plan_builds_data_sdh_paths_without_starting_training(tmp_path: pathlib.Path):
    plan = robotwin_pi05_prepare.build_plan(
        data_root=tmp_path,
        task_name="beat_block_hammer",
        task_config="demo_clean",
        expert_data_num=50,
        repo_id="robotwin/pi05_aloha",
        model_name="demo_clean",
        gpu_use="0,1",
    )

    assert plan.raw_dir == tmp_path / "robotwin" / "data" / "beat_block_hammer" / "demo_clean"
    assert plan.processed_dir == tmp_path / "openpi" / "processed_data" / "beat_block_hammer-demo_clean-50"
    assert plan.training_dir == tmp_path / "openpi" / "training_data" / "demo_clean"
    assert plan.lerobot_dir == tmp_path / "openpi-cache" / "huggingface" / "lerobot" / "robotwin" / "pi05_aloha"
    assert plan.train_command is not None
    assert "scripts/train.py" in " ".join(plan.train_command)
    assert plan.should_start_training is False
