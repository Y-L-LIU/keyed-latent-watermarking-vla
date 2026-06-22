"""Convert RoboTwin2 collect_data HDF5 -> inline-PNG LeRobot v2.1 (aloha), merging tasks per set.

RoboTwin2 `collect_data` writes a NATIVE schema that the stock
openpi/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py does NOT read:
  /joint_action/vector      (T,14) float  -- 14-dim JOINT (left 7 + right 7)
  /observation/<cam>/rgb     (T,)  S<n>    -- jpeg-encoded frames per camera
  /endpose/...                              -- 16-dim eef (unused here)

This converter reproduces the robotwin10_clean format the pi05_aloha base trains on:
  - state[t]  = joint_action/vector[t]
  - action[t] = joint_action/vector[t+1]   (next-state target; last action = last state)
  - images    = decode jpeg -> inline PNG (LeRobotDataset.create mode=image)
Cameras are auto-detected and mapped head->cam_high, left->cam_left_wrist, right->cam_right_wrist.

Usage (run after sim-gen, before training):
  HF_LEROBOT_HOME=/workspace/vla/robotwin2_train/lerobot_home \
  python3 robotwin2_to_lerobot.py \
    --collect-root /workspace/vla/robotwin_collect --config demo_clean_gen \
    --repo-id local/robotwin10_setB \
    --tasks blocks_ranking_rgb,blocks_ranking_size,...   # 10 set-B tasks
Then: python3 -m scripts.compute_norm_stats --config-name pi05_aloha_robotwin_lora_setB
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import cv2
import h5py
import numpy as np
import tqdm

# lerobot 0.1.0 (system) provides the old create() API that writes v2.1 inline-image format,
# byte-compatible with robotwin10_clean.
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _create_dataset(repo_id: str) -> LeRobotDataset:
    out = HF_LEROBOT_HOME / repo_id
    if pathlib.Path(out).exists():
        shutil.rmtree(out)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=50, robot_type="aloha", features=features,
        use_videos=False, image_writer_processes=10, image_writer_threads=5,
    )


def _map_cameras(obs_group: h5py.Group) -> dict[str, str]:
    """Map RoboTwin camera-group names -> aloha cam_high/cam_left_wrist/cam_right_wrist."""
    cams = [k for k in obs_group.keys() if "rgb" in obs_group[k]]
    mapping: dict[str, str] = {}
    for name in cams:
        low = name.lower()
        if "head" in low or "top" in low or "high" in low:
            mapping["cam_high"] = name
        elif "left" in low:
            mapping["cam_left_wrist"] = name
        elif "right" in low:
            mapping["cam_right_wrist"] = name
    if set(mapping) != set(CAMERAS):  # fallback: positional
        ordered = sorted(cams)
        if len(ordered) >= 3:
            mapping = {CAMERAS[i]: ordered[i] for i in range(3)}
    missing = set(CAMERAS) - set(mapping)
    if missing:
        raise ValueError(f"could not map cameras {missing}; hdf5 has {cams}")
    return mapping


def _decode_rgb(ds: h5py.Dataset) -> np.ndarray:
    """Decode a (T,) jpeg-bytes dataset -> (T,H,W,3) uint8."""
    frames = []
    for payload in ds[:]:
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode rgb frame")
        frames.append(img)
    return np.asarray(frames)


def _instruction_for(task: str, ep_idx: int, task_dir: pathlib.Path) -> str:
    """Find the task instruction. RoboTwin generates per-episode instruction json; fall back to
    a humanized task name. Auto-detects common locations; override with --instruction if needed."""
    for cand in [
        task_dir / "instructions" / f"episode{ep_idx}.json",
        task_dir / "instructions.json",
        task_dir.parent / "instructions" / f"episode{ep_idx}.json",
    ]:
        if cand.exists():
            try:
                obj = json.loads(cand.read_text())
                ins = obj.get("instructions", obj.get("instruction"))
                if isinstance(ins, list) and ins:
                    return str(ins[0])
                if isinstance(ins, str) and ins:
                    return ins
            except Exception:
                pass
    return task.replace("_", " ")


def _episodes(task_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(task_dir.glob("data/episode*.hdf5"),
                  key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or -1))


def convert(collect_root: pathlib.Path, config: str, tasks: list[str], repo_id: str,
            limit: int | None, instruction_override: str | None) -> pathlib.Path:
    dataset = _create_dataset(repo_id)
    total = 0
    for task in tasks:
        task_dir = collect_root / task / config
        eps = _episodes(task_dir)
        if limit:
            eps = eps[:limit]
        if not eps:
            print(f"[warn] no hdf5 for task {task} under {task_dir}/data")
            continue
        for ep_path in tqdm.tqdm(eps, desc=f"{task}"):
            ep_idx = int("".join(c for c in ep_path.stem if c.isdigit()) or 0)
            with h5py.File(ep_path, "r") as ep:
                joint = np.asarray(ep["/joint_action/vector"][:], dtype=np.float32)  # (T,14)
                obs = ep["/observation"]
                cam_map = _map_cameras(obs)
                imgs = {c: _decode_rgb(obs[cam_map[c]]["rgb"]) for c in CAMERAS}
            T = joint.shape[0]
            action = np.empty_like(joint)
            action[:-1] = joint[1:]
            action[-1] = joint[-1]
            instr = instruction_override or _instruction_for(task, ep_idx, task_dir)
            for t in range(T):
                frame = {"observation.state": joint[t], "action": action[t], "task": instr}
                for c in CAMERAS:
                    frame[f"observation.images.{c}"] = imgs[c][t]
                dataset.add_frame(frame)
            dataset.save_episode()
            total += 1
    print(f"wrote {total} episodes to {HF_LEROBOT_HOME / repo_id}")
    return pathlib.Path(HF_LEROBOT_HOME / repo_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-root", type=pathlib.Path, default=pathlib.Path("/workspace/vla/robotwin_collect"))
    ap.add_argument("--config", default="demo_clean_gen")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--limit", type=int, default=None, help="cap episodes/task (validation)")
    ap.add_argument("--instruction", default=None, help="override instruction for all episodes")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    convert(args.collect_root, args.config, tasks, args.repo_id, args.limit, args.instruction)


if __name__ == "__main__":
    main()
