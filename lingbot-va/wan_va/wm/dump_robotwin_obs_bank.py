"""Dump initial RoboTwin observations for LingBot Attack-D training.

The Attack-D loop only needs realistic first observations to prime LingBot's video
KV-cache. Rendering them once keeps the fine-tune loop free of simulator imports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .eval_robotwin_watermark import (
    DEFAULT_10_TASKS,
    _class_decorator,
    _format_obs,
    _load_task_config,
    _setup_robotwin,
)


def _instruction_from_json(robotwin_root: str, task_name: str) -> str:
    path = Path(robotwin_root) / "description" / "task_instruction" / f"{task_name}.json"
    with open(path) as f:
        data = json.load(f)
    return data.get("full_description", task_name.replace("_", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin-root", type=str, required=True)
    ap.add_argument("--out", type=str,
                    default="/workspace/vla_out/attack_c/obs_banks/lingbot_robotwin_init_obs_bank.npz")
    ap.add_argument("--task-names", type=str, nargs="+", default=None)
    ap.add_argument("--bank-tasks", type=int, default=8)
    ap.add_argument("--task-config", type=str, default="demo_clean")
    ap.add_argument("--start-seed", type=int, default=10000)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    _setup_robotwin(args.robotwin_root)
    from evaluation.robotwin.test_render import Sapien_TEST
    Sapien_TEST()

    task_names = (args.task_names or DEFAULT_10_TASKS)[: args.bank_tasks]
    task_config_args = _load_task_config(args.robotwin_root, args.task_config)

    prompts, obs_bank, used_tasks, seeds = [], [], [], []
    for i, task_name in enumerate(task_names):
        task_env = _class_decorator(task_name)
        task_args = dict(task_config_args)
        task_args["task_name"] = task_name
        task_args["task_config"] = args.task_config
        task_args["eval_mode"] = True
        task_args["render_freq"] = 0
        task_args["eval_video_log"] = False
        seed = args.start_seed + i
        try:
            task_env.suc = 0
            task_env.test_num = 0
            task_env.setup_demo(now_ep_num=i, seed=seed, is_test=True, **task_args)
            instruction = _instruction_from_json(args.robotwin_root, task_name)
            task_env.set_instruction(instruction=instruction)
            prompt = task_env.get_instruction()
            obs_bank.append(_format_obs(task_env.get_obs(), prompt))
            prompts.append(prompt)
            used_tasks.append(task_name)
            seeds.append(seed)
            print(f"[dump-obs-bank] {i}: {task_name} seed={seed} | {prompt}", flush=True)
        finally:
            try:
                task_env.close_env()
            except Exception:
                pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        task_names=np.asarray(used_tasks, dtype=object),
        prompts=np.asarray(prompts, dtype=object),
        obs=np.asarray(obs_bank, dtype=object),
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    print(f"[dump-obs-bank] wrote {len(obs_bank)} entries -> {out}", flush=True)


if __name__ == "__main__":
    main()
