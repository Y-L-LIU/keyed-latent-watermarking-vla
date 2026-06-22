"""Convert LIBERO HDF5 (yifengzhu-hf/LIBERO-datasets) to LeRobot format.

Reads HDF5 demos from a directory like
``/workspace/vla/openpi/third_party/libero/libero/datasets/libero_10/``
and writes a LeRobotDataset under ``$HF_LEROBOT_HOME / repo_id``.

State layout matches openpi LiberoInputs expectations:
  state = concat(joint_states[:7], gripper_states[:1]) → (8,)
  action = actions[:, :7] (already 7-d)

Images are kept at native 128x128; openpi's preprocess pipeline resizes to 224.

Usage:
    python scripts/attacks/hdf5_to_lerobot_libero.py \
        --hdf5-dir /workspace/vla/openpi/third_party/libero/libero/datasets/libero_10 \
        --repo-id local/libero_10
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


_SCENE_PREFIX_RE = re.compile(r"^(?:KITCHEN_SCENE\d+|LIVING_ROOM_SCENE\d+|STUDY_SCENE\d+)_")


def _filename_to_task(fname: str) -> str:
    stem = fname.replace("_demo.hdf5", "")
    stem = _SCENE_PREFIX_RE.sub("", stem)
    return stem.replace("_", " ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir", required=True)
    p.add_argument("--repo-id", default="local/libero_10")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--image-h", type=int, default=128)
    p.add_argument("--image-w", type=int, default=128)
    p.add_argument("--max-demos-per-file", type=int, default=None,
                   help="If set, only take the first N demos per HDF5 (for quick smoke).")
    args = p.parse_args()

    hdf5_dir = pathlib.Path(args.hdf5_dir)
    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        print(f"Removing existing {out_path}")
        shutil.rmtree(out_path)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (args.image_h, args.image_w, 3),
                       "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (args.image_h, args.image_w, 3),
                             "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    hdf5s = sorted(hdf5_dir.glob("*.hdf5"))
    if not hdf5s:
        raise SystemExit(f"No .hdf5 files in {hdf5_dir}")

    total_demos = 0
    total_frames = 0
    for hdf5_path in hdf5s:
        task_name = _filename_to_task(hdf5_path.name)
        print(f"== {hdf5_path.name}  task='{task_name}' ==")
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if args.max_demos_per_file is not None:
                demo_keys = demo_keys[: args.max_demos_per_file]
            for demo_key in demo_keys:
                d = f["data"][demo_key]
                actions = np.asarray(d["actions"], dtype=np.float32)
                agentview = np.asarray(d["obs/agentview_rgb"], dtype=np.uint8)
                wrist = np.asarray(d["obs/eye_in_hand_rgb"], dtype=np.uint8)
                joint = np.asarray(d["obs/joint_states"], dtype=np.float32)
                gripper = np.asarray(d["obs/gripper_states"], dtype=np.float32)
                T = actions.shape[0]
                # Image origin in robomimic HDF5 is flipped (bottom-up); LIBERO eval
                # uses cv2.flip(image, 0). Flip both cameras to match inference.
                agentview = agentview[:, ::-1, :, :]
                wrist = wrist[:, ::-1, :, :]
                # state = 7 joints + first gripper channel
                state = np.concatenate([joint[:, :7], gripper[:, :1]], axis=1).astype(np.float32)
                action7 = actions[:, :7].astype(np.float32)
                for t in range(T):
                    ds.add_frame({
                        "image": agentview[t],
                        "wrist_image": wrist[t],
                        "state": state[t],
                        "actions": action7[t],
                        "task": task_name,
                    })
                ds.save_episode()
                total_demos += 1
                total_frames += T
        print(f"  cumulative demos={total_demos} frames={total_frames}")

    print(f"\nDONE. {total_demos} demos / {total_frames} frames -> {out_path}")


if __name__ == "__main__":
    main()
