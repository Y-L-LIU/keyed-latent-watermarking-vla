"""Convert openpi LIBERO HDF5 (a suite dir of <task>_demo.hdf5) -> LeRobot v2.1 dataset matching
the libero_long format the LingBot latent extractor expects.

Target schema (from lingbot_latents/libero_long/meta/info.json):
  observation.state                  float32 [8]  = concat(ee_pos[3], ee_ori[3], gripper_states[2])
  action                             float32 [7]  = hdf5 /data/demo_i/actions
  observation.images.agentview_rgb   image (128,128,3)
  observation.images.eye_in_hand_rgb image (128,128,3)
  fps 60, robot_type Franka
The 8-dim state composition mirrors openpi/examples/libero/main.py:133 (eef_pos + axis-angle + gripper);
the LIBERO hdf5 already stores ee_ori as 3-dim axis-angle, so we concat directly.
Instruction = /data attrs problem_info -> language_instruction (one per task file).

Usage:
  HF_LEROBOT_HOME=/workspace/vla/lingbot_latents/_src \
  python3 libero_hdf5_to_lerobot.py --suite-dir openpi/third_party/libero/libero/datasets/libero_goal \
     --repo-id libero_goal
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import h5py
import numpy as np
import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

CAMS = {"agentview_rgb": "agentview_rgb", "eye_in_hand_rgb": "eye_in_hand_rgb"}


def _create(repo_id: str) -> LeRobotDataset:
    out = HF_LEROBOT_HOME / repo_id
    if pathlib.Path(out).exists():
        shutil.rmtree(out)
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [["s%d" % i for i in range(8)]]},
        "action": {"dtype": "float32", "shape": (7,), "names": [["a%d" % i for i in range(7)]]},
    }
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=60, robot_type="Franka", features=features,
        use_videos=False, image_writer_processes=10, image_writer_threads=5,
    )


def _instruction(h: h5py.File) -> str:
    try:
        pi = json.loads(h["data"].attrs.get("problem_info", "{}"))
        ins = pi.get("language_instruction")
        if ins:
            return str(ins)
    except Exception:
        pass
    return ""


def convert(suite_dir: pathlib.Path, repo_id: str, limit_demos: int | None) -> pathlib.Path:
    files = sorted(suite_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no .hdf5 under {suite_dir}")
    ds = _create(repo_id)
    total = 0
    for fp in files:
        with h5py.File(fp, "r") as h:
            instr = _instruction(h) or fp.stem.replace("_demo", "").replace("_", " ")
            demos = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
            if limit_demos:
                demos = demos[:limit_demos]
            for dname in tqdm.tqdm(demos, desc=fp.stem[:40]):
                d = h["data"][dname]
                obs = d["obs"]
                state = np.concatenate(
                    [np.asarray(obs["ee_pos"], np.float32),
                     np.asarray(obs["ee_ori"], np.float32),
                     np.asarray(obs["gripper_states"], np.float32)], axis=1)  # (T,8)
                action = np.asarray(d["actions"], np.float32)                  # (T,7)
                agv = np.asarray(obs["agentview_rgb"])                         # (T,128,128,3) uint8
                eih = np.asarray(obs["eye_in_hand_rgb"])
                T = min(state.shape[0], action.shape[0], agv.shape[0], eih.shape[0])
                for t in range(T):
                    ds.add_frame({
                        "observation.state": state[t],
                        "action": action[t],
                        "observation.images.agentview_rgb": agv[t],
                        "observation.images.eye_in_hand_rgb": eih[t],
                        "task": instr,
                    })
                ds.save_episode()
                total += 1
    print(f"wrote {total} episodes to {HF_LEROBOT_HOME / repo_id}")
    return pathlib.Path(HF_LEROBOT_HOME / repo_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", type=pathlib.Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--limit-demos", type=int, default=None)
    a = ap.parse_args()
    convert(a.suite_dir, a.repo_id, a.limit_demos)


if __name__ == "__main__":
    main()
