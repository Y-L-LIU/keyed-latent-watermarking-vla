"""Prepare RoboTwin Pi0.5 fine-tuning paths and commands.

This script does not start training. It creates the directory skeleton and prints
the conversion, norm-stat, and training commands to run when data is present.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import pathlib
import shlex


DEFAULT_DATA_ROOT = pathlib.Path("/data_sdh/anon")
DEFAULT_CONFIG_NAME = "pi05_aloha_robotwin_full"
DEFAULT_REPO_ID = "robotwin/pi05_aloha"


@dataclasses.dataclass(frozen=True)
class RobotwinPi05Plan:
    data_root: pathlib.Path
    raw_dir: pathlib.Path
    processed_dir: pathlib.Path
    training_dir: pathlib.Path
    xdg_cache_home: pathlib.Path
    lerobot_dir: pathlib.Path
    assets_dir: pathlib.Path
    checkpoint_dir: pathlib.Path
    process_command: tuple[str, ...]
    stage_command: tuple[str, ...]
    generate_lerobot_command: tuple[str, ...]
    norm_stats_command: tuple[str, ...]
    train_command: tuple[str, ...]
    should_start_training: bool = False


def _repo_path(repo_id: str) -> pathlib.Path:
    return pathlib.Path(*repo_id.split("/"))


def _shell_env_command(env: dict[str, pathlib.Path | str], command: Sequence[str]) -> tuple[str, ...]:
    exports = " ".join(f"{name}={shlex.quote(str(value))}" for name, value in env.items())
    body = " ".join(shlex.quote(part) for part in command)
    return ("bash", "-lc", f"{exports} {body}")


def build_plan(
    *,
    data_root: pathlib.Path = DEFAULT_DATA_ROOT,
    task_name: str,
    task_config: str,
    expert_data_num: int,
    repo_id: str = DEFAULT_REPO_ID,
    model_name: str,
    gpu_use: str,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> RobotwinPi05Plan:
    data_root = data_root.expanduser()
    raw_dir = data_root / "robotwin" / "data" / task_name / task_config
    processed_dir = data_root / "openpi" / "processed_data" / f"{task_name}-{task_config}-{expert_data_num}"
    training_dir = data_root / "openpi" / "training_data" / model_name
    xdg_cache_home = data_root / "openpi-cache"
    assets_dir = data_root / "openpi-assets" / config_name
    checkpoint_dir = data_root / "openpi-checkpoints" / config_name / model_name
    lerobot_dir = xdg_cache_home / "huggingface" / "lerobot" / _repo_path(repo_id)

    process_command = (
        "uv",
        "run",
        "scripts/process_robotwin_data.py",
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(processed_dir),
        "--episodes",
        str(expert_data_num),
    )
    stage_command = (
        "rsync",
        "-a",
        f"{processed_dir}/",
        f"{training_dir / processed_dir.name}/",
    )
    generate_lerobot_command = _shell_env_command(
        {"XDG_CACHE_HOME": xdg_cache_home},
        (
            "uv",
            "run",
            "examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py",
            "--raw-dir",
            str(training_dir),
            "--repo-id",
            repo_id,
        ),
    )
    norm_stats_command = _shell_env_command(
        {"XDG_CACHE_HOME": xdg_cache_home, "OPENPI_DATA_HOME": data_root / "openpi-data"},
        ("uv", "run", "scripts/compute_norm_stats.py", "--config-name", config_name),
    )
    train_command = _shell_env_command(
        {
            "CUDA_VISIBLE_DEVICES": gpu_use,
            "XDG_CACHE_HOME": xdg_cache_home,
            "OPENPI_DATA_HOME": data_root / "openpi-data",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        },
        ("uv", "run", "scripts/train.py", config_name, f"--exp-name={model_name}", "--overwrite"),
    )

    return RobotwinPi05Plan(
        data_root=data_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        training_dir=training_dir,
        xdg_cache_home=xdg_cache_home,
        lerobot_dir=lerobot_dir,
        assets_dir=assets_dir,
        checkpoint_dir=checkpoint_dir,
        process_command=process_command,
        stage_command=stage_command,
        generate_lerobot_command=generate_lerobot_command,
        norm_stats_command=norm_stats_command,
        train_command=train_command,
    )


def create_directory_skeleton(plan: RobotwinPi05Plan) -> None:
    for path in [
        plan.data_root / "robotwin" / "data",
        plan.processed_dir.parent,
        plan.training_dir.parent,
        plan.xdg_cache_home,
        plan.assets_dir.parent,
        plan.checkpoint_dir.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--expert-data-num", type=int, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gpu-use", default="0")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--no-create-dirs", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(
        data_root=args.data_root,
        task_name=args.task_name,
        task_config=args.task_config,
        expert_data_num=args.expert_data_num,
        repo_id=args.repo_id,
        model_name=args.model_name,
        gpu_use=args.gpu_use,
        config_name=args.config_name,
    )
    if not args.no_create_dirs:
        create_directory_skeleton(plan)

    print(f"raw_dir={plan.raw_dir}")
    print(f"processed_dir={plan.processed_dir}")
    print(f"training_dir={plan.training_dir}")
    print(f"lerobot_dir={plan.lerobot_dir}")
    print(f"assets_dir={plan.assets_dir}")
    print(f"checkpoint_dir={plan.checkpoint_dir}")
    print()
    print("# 1. Convert RoboTwin raw data to OpenPI HDF5")
    print(_format_command(plan.process_command))
    print("# 2. Stage processed episodes for LeRobot conversion")
    print(_format_command(plan.stage_command))
    print("# 3. Generate LeRobotDataset under /data_sdh")
    print(_format_command(plan.generate_lerobot_command))
    print("# 4. Compute norm stats")
    print(_format_command(plan.norm_stats_command))
    print("# 5. Training command, do not run until explicitly confirmed")
    print(_format_command(plan.train_command))


if __name__ == "__main__":
    main()
