#!/usr/bin/env python3.11
"""Removability of the surviving (DC per-task) mark.

The only mark that survives distillation is a constant per-task action offset.
That is exactly the per-task MEAN of the action deviation, which an adversary
estimates from a few rollouts and subtracts. We show detection Z collapses after
per-task mean subtraction -> the surviving mark is trivially removable, completing
the trade-off: distillable <=> removable.
"""
from __future__ import annotations
import glob, hashlib, numpy as np, h5py
from collections import defaultdict
from openpi.training.config import get_config
from openpi.training.data_loader import _read_libero_image, _read_libero_state
import openpi.policies.policy_config as pc
import dc_keying

CKPT = "/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_outputtaskdc/distill_outputtaskdc_k42/1499"
KEY, BETA, NDEC = 42, 0.1, 32

base = pc.create_trained_policy(get_config("pi05_libero"), "/workspace/vla/models/pi05_libero")
stu = pc.create_trained_policy(get_config("pi05_libero_goal_lora_distill_outputtaskdc"), CKPT)
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_goal"]()

by_task = defaultdict(list)  # prompt -> list of (residual HxD)
for f in sorted(glob.glob("/workspace/vla/openpi/third_party/libero/libero/datasets/libero_goal/*.hdf5")):
    tname = f.split("/")[-1]
    prompt = next((suite.get_task(t).language for t in range(suite.n_tasks)
                   if suite.get_task_demonstration(t).split("/")[-1] == tname), "do task")
    with h5py.File(f, "r") as h:
        for dk in sorted(h["data"].keys())[:2]:
            obs = h["data"][dk]["obs"]; T = h["data"][dk]["actions"].shape[0]
            for s in range(0, T, 10):
                st = _read_libero_state(obs, s)
                o = {"observation/image": _read_libero_image(obs, s, ("agentview_image", "agentview_rgb")),
                     "observation/wrist_image": _read_libero_image(obs, s, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")),
                     "observation/state": st, "prompt": prompt}
                ab = np.asarray(base.infer(o)["actions"], dtype=np.float64)
                au = np.asarray(stu.infer(o)["actions"], dtype=np.float64)
                H = min(len(ab), len(au))
                by_task[prompt].append(au[:H] - ab[:H])

D = next(iter(by_task.values()))[0].shape[1]
keys = [KEY] + [KEY + 1 + j for j in range(NDEC)]


def detect(residuals_by_task):
    sums = {k: 0.0 for k in keys}
    for prompt, resids in residuals_by_task.items():
        ts = dc_keying.prompt_seed(prompt)
        for resid in resids:
            H = resid.shape[0]
            for k in keys:
                c = dc_keying.dc_offset(k, ts, D); rn = np.tile(c[None, :], (H, 1))
                rn = rn / (np.linalg.norm(rn) + 1e-8)
                sums[k] += float(np.sum(rn * resid))
    dec = np.array([sums[k] for k in keys[1:]])
    return (sums[KEY] - dec.mean()) / (dec.std() + 1e-8)

Z_raw = detect(by_task)
# adversary removal: subtract per-task mean residual (the constant offset)
removed = {p: [r - np.mean(np.stack(rs), axis=0) for r in rs] for p, rs in by_task.items()}
Z_removed = detect(removed)
print(f"=== DC mark removability (per-task mean subtraction) ===")
print(f"detection Z  before removal = {Z_raw:+.2f}")
print(f"detection Z  after  removal = {Z_removed:+.2f}")
print(f"=> the surviving mark is {'REMOVED (distillable<=>removable)' if abs(Z_removed) < 1.0 else 'still present'}")
