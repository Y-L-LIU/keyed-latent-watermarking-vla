"""Convert RoboTwin/OpenPI ALOHA-style HDF5 episodes to LeRobot format."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib
import random
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm


MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _create_dataset(repo_id: str, *, mode: str = "image") -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if pathlib.Path(output_path).exists():
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=mode == "video",
        image_writer_processes=10,
        image_writer_threads=5,
    )


def _find_hdf5_files(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in raw_dir.rglob("*.hdf5") if path.name.startswith("episode_"))


def _load_images(ep: h5py.File) -> dict[str, np.ndarray]:
    images = {}
    for camera in CAMERAS:
        values = ep[f"/observations/images/{camera}"]
        if values.ndim == 4:
            images[camera] = values[:]
            continue
        decoded = []
        for payload in values:
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode {camera} frame")
            decoded.append(image)
        images[camera] = np.asarray(decoded)
    return images


def _load_episode(path: pathlib.Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str]:
    with h5py.File(path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        images = _load_images(ep)

    instructions_path = path.parent / "instructions.json"
    instructions = json.loads(instructions_path.read_text())["instructions"]
    instruction = instructions if isinstance(instructions, str) else random.choice(instructions)
    return images, state, action, instruction


def convert(raw_dir: pathlib.Path, repo_id: str, *, mode: str = "image", push_to_hub: bool = False) -> pathlib.Path:
    hdf5_files = _find_hdf5_files(raw_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {raw_dir}")

    dataset = _create_dataset(repo_id, mode=mode)
    for path in tqdm.tqdm(hdf5_files, desc="Writing LeRobot episodes"):
        images, state, action, instruction = _load_episode(path)
        for frame_index in range(state.shape[0]):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
                "task": instruction,
            }
            for camera in CAMERAS:
                frame[f"observation.images.{camera}"] = images[camera][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

    return pathlib.Path(HF_LEROBOT_HOME / repo_id)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args(argv)
    output_path = convert(args.raw_dir, args.repo_id, mode=args.mode, push_to_hub=args.push_to_hub)
    print(f"Wrote LeRobot dataset to: {output_path}")


if __name__ == "__main__":
    main()
