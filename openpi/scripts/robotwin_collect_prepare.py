"""Prepare RoboTwin data collection under /data_sdh without starting collection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import pathlib
import shlex


DEFAULT_DATA_ROOT = pathlib.Path("/data_sdh/anon")
DEFAULT_ROBOTWIN_URL = "https://github.com/RoboTwin-Platform/RoboTwin.git"
DEFAULT_BRANCH = "main"


@dataclasses.dataclass(frozen=True)
class RobotwinCollectPlan:
    data_root: pathlib.Path
    repo_dir: pathlib.Path
    data_dir: pathlib.Path
    project_data_path: pathlib.Path
    raw_output_dir: pathlib.Path
    clone_command: tuple[str, ...]
    symlink_command: tuple[str, ...]
    collect_command: tuple[str, ...]
    should_start_collection: bool = False


def _shell_command(command: str) -> tuple[str, ...]:
    return ("bash", "-lc", command)


def build_plan(
    *,
    data_root: pathlib.Path = DEFAULT_DATA_ROOT,
    task_name: str,
    task_config: str,
    gpu_id: str,
    repo_url: str = DEFAULT_ROBOTWIN_URL,
    branch: str = DEFAULT_BRANCH,
) -> RobotwinCollectPlan:
    data_root = data_root.expanduser()
    repo_dir = data_root / "robotwin" / "RoboTwin"
    data_dir = data_root / "robotwin" / "data"
    project_data_path = repo_dir / "data"
    raw_output_dir = data_dir / task_name / task_config

    clone_command = (
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        branch,
        repo_url,
        str(repo_dir),
    )
    symlink_command = _shell_command(
        "mkdir -p "
        f"{shlex.quote(str(data_dir))} && "
        f"rm -rf {shlex.quote(str(project_data_path))} && "
        f"ln -s {shlex.quote(str(data_dir))} {shlex.quote(str(project_data_path))}"
    )
    collect_command = _shell_command(
        f"cd {shlex.quote(str(repo_dir))} && "
        f"bash collect_data.sh {shlex.quote(task_name)} {shlex.quote(task_config)} {shlex.quote(gpu_id)}"
    )

    return RobotwinCollectPlan(
        data_root=data_root,
        repo_dir=repo_dir,
        data_dir=data_dir,
        project_data_path=project_data_path,
        raw_output_dir=raw_output_dir,
        clone_command=clone_command,
        symlink_command=symlink_command,
        collect_command=collect_command,
    )


def create_directory_skeleton(plan: RobotwinCollectPlan) -> None:
    plan.data_dir.mkdir(parents=True, exist_ok=True)
    plan.repo_dir.parent.mkdir(parents=True, exist_ok=True)


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--repo-url", default=DEFAULT_ROBOTWIN_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-create-dirs", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(
        data_root=args.data_root,
        task_name=args.task_name,
        task_config=args.task_config,
        gpu_id=args.gpu_id,
        repo_url=args.repo_url,
        branch=args.branch,
    )
    if not args.no_create_dirs:
        create_directory_skeleton(plan)

    print(f"repo_dir={plan.repo_dir}")
    print(f"data_dir={plan.data_dir}")
    print(f"project_data_path={plan.project_data_path}")
    print(f"raw_output_dir={plan.raw_output_dir}")
    print()
    print("# 1. Clone RoboTwin if the repo is not already present")
    print(_format_command(plan.clone_command))
    print("# 2. Link RoboTwin's project-local data directory to /data_sdh")
    print(_format_command(plan.symlink_command))
    print("# 3. Collection command, do not run until explicitly confirmed")
    print(_format_command(plan.collect_command))


if __name__ == "__main__":
    main()
