"""Convert RoboTwin collected data into ALOHA-style HDF5 episodes for OpenPI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib

import cv2
import h5py
import numpy as np


DESC_TYPE = "seen"
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _load_robotwin_episode(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin episode does not exist: {path}")

    with h5py.File(path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        images = {
            camera_name: root[f"/observation/{camera_name}/rgb"][()]
            for camera_name in root["/observation"]
            if camera_name in CAMERA_MAP
        }

    missing = set(CAMERA_MAP) - set(images)
    if missing:
        raise KeyError(f"Missing RoboTwin cameras in {path}: {sorted(missing)}")

    return left_gripper, left_arm, right_gripper, right_arm, images


def _decode_image(image_bits: bytes | np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bits, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RoboTwin image bytes")
    return cv2.resize(image, (640, 480))


def _encode_images(images: Sequence[np.ndarray]) -> tuple[list[bytes], int]:
    encoded: list[bytes] = []
    max_len = 0
    for image in images:
        ok, encoded_image = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode OpenPI image")
        payload = encoded_image.tobytes()
        encoded.append(payload)
        max_len = max(max_len, len(payload))
    return [payload.ljust(max_len, b"\0") for payload in encoded], max_len


def _load_instructions(raw_dir: pathlib.Path, episode_index: int, desc_type: str) -> list[str]:
    path = raw_dir / "instructions" / f"episode{episode_index}.json"
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin instructions do not exist: {path}")
    payload = json.loads(path.read_text())
    instructions = payload[desc_type]
    if isinstance(instructions, str):
        return [instructions]
    return list(instructions)


def convert_episode(raw_dir: pathlib.Path, output_dir: pathlib.Path, episode_index: int, desc_type: str = DESC_TYPE) -> None:
    instructions = _load_instructions(raw_dir, episode_index, desc_type)
    episode_dir = output_dir / f"episode_{episode_index}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "instructions.json").write_text(json.dumps({"instructions": instructions}, indent=2))

    left_gripper, left_arm, right_gripper, right_arm, image_dict = _load_robotwin_episode(
        raw_dir / "data" / f"episode{episode_index}.hdf5"
    )

    states = []
    actions = []
    images_by_output = {output_name: [] for output_name in CAMERA_MAP.values()}
    left_arm_dim = []
    right_arm_dim = []

    for frame_index in range(left_gripper.shape[0]):
        state = np.array(
            [
                *left_arm[frame_index].tolist(),
                left_gripper[frame_index],
                *right_arm[frame_index].tolist(),
                right_gripper[frame_index],
            ],
            dtype=np.float32,
        )

        if frame_index != left_gripper.shape[0] - 1:
            states.append(state)
            for raw_camera, output_camera in CAMERA_MAP.items():
                images_by_output[output_camera].append(_decode_image(image_dict[raw_camera][frame_index]))

        if frame_index != 0:
            actions.append(state)
            left_arm_dim.append(left_arm[frame_index].shape[0])
            right_arm_dim.append(right_arm[frame_index].shape[0])

    with h5py.File(episode_dir / f"episode_{episode_index}.hdf5", "w") as output:
        output.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
        observations = output.create_group("observations")
        observations.create_dataset("qpos", data=np.asarray(states, dtype=np.float32))
        observations.create_dataset("left_arm_dim", data=np.asarray(left_arm_dim))
        observations.create_dataset("right_arm_dim", data=np.asarray(right_arm_dim))
        image_group = observations.create_group("images")
        for camera_name, images in images_by_output.items():
            encoded, max_len = _encode_images(images)
            image_group.create_dataset(camera_name, data=encoded, dtype=f"S{max_len}")


def convert_dataset(raw_dir: pathlib.Path, output_dir: pathlib.Path, episodes: int, desc_type: str = DESC_TYPE) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_index in range(episodes):
        convert_episode(raw_dir, output_dir, episode_index, desc_type=desc_type)
        print(f"processed episode {episode_index}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--desc-type", default=DESC_TYPE)
    args = parser.parse_args(argv)
    convert_dataset(args.raw_dir, args.output_dir, args.episodes, desc_type=args.desc_type)


if __name__ == "__main__":
    main()
