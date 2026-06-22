#!/usr/bin/env python3.11
"""Faithful latent-DC relabel: inject in the SEED (paper's site) with a low-entropy,
non-zero-mean per-task reference.

The paper injects z^fp = sqrt(1-b^2) z + b r. Here r is a CONSTANT-in-time, non-zero-mean
vector keyed only on the task (low entropy) -- the two obstacles (high-entropy keying,
zero-mean) removed, but the injection site is exactly the paper's seed. At beta=1 the
teacher becomes a deterministic per-task policy F(r_task; o), which a BC student can learn.
We inject by passing the keyed seed as the `noise=` arg to the CLEAN base policy (no
watermark.py change). Observations stay byte-identical; only actions change.

Expected: SURVIVES distillation (faithful to the paper's injection point) -- but z^fp is no
longer N(0,I) (mean = b*r != 0), so imperceptibility is forfeited.
"""
from __future__ import annotations
import argparse, pathlib, shutil, time, sys
import h5py, numpy as np
sys.path.insert(0, "/workspace/vla/distill")
import dc_keying
from openpi.training.config import get_config
from openpi.training.data_loader import _read_libero_image, _read_libero_state
import openpi.policies.policy_config as policy_config
from libero.libero import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--task-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-demos", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config("pi05_libero")
    H = int(cfg.model.action_horizon); Draw = int(cfg.model.action_dim)
    policy = policy_config.create_trained_policy(cfg, "/workspace/vla/models/pi05_libero")
    print(f"[latent_dc] base loaded H={H} Draw={Draw} beta={args.beta}", flush=True)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    root = pathlib.Path("/workspace/vla/openpi/third_party/libero/libero/datasets")
    out_root = pathlib.Path(args.out_dir) / "libero_goal"; out_root.mkdir(parents=True, exist_ok=True)
    task_ids = list(range(suite.n_tasks))
    if args.task_range:
        lo, hi = args.task_range; task_ids = [t for t in task_ids if lo <= t < hi]

    for tid in task_ids:
        rel = suite.get_task_demonstration(tid); src = root / rel
        dst = out_root / pathlib.Path(rel).name; prompt = suite.get_task(tid).language
        if dst.exists():
            print(f"[latent_dc] {dst.name} exists, skip", flush=True); continue
        # DC seed for this task: constant in time, non-zero-mean, keyed on (key, task)
        c = dc_keying.dc_offset(args.secret_key, dc_keying.prompt_seed(prompt), Draw)  # (Draw,)
        r_seed = np.tile(c[None, :], (H, 1)).astype(np.float32)                        # (H, Draw)
        tmp = dst.with_suffix(".tmp.hdf5"); shutil.copyfile(src, tmp); t0 = time.time()
        with h5py.File(tmp, "r+") as f:
            dks = sorted(f["data"].keys())
            if args.max_demos:
                dks = dks[: args.max_demos]
            for dk in dks:
                obs = f["data"][dk]["obs"]; acts = f["data"][dk]["actions"]
                T = int(acts.shape[0]); ad = int(acts.shape[1]); out = np.zeros_like(acts[...])
                s = 0
                while s < T:
                    o = {"observation/image": _read_libero_image(obs, s, ("agentview_image", "agentview_rgb")),
                         "observation/wrist_image": _read_libero_image(obs, s, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")),
                         "observation/state": _read_libero_state(obs, s), "prompt": prompt}
                    # z^fp = sqrt(1-b^2) z + b r_seed ; at beta=1 -> r_seed (deterministic per-task)
                    z = np.random.standard_normal((H, Draw)).astype(np.float32)
                    zfp = (np.sqrt(max(0.0, 1 - args.beta**2)) * z + args.beta * r_seed).astype(np.float32)
                    chunk = np.asarray(policy.infer(o, noise=zfp)["actions"], dtype=np.float32)
                    n = min(H, T - s, chunk.shape[0]); out[s:s+n] = chunk[:n, :ad]
                    s += H
                acts[...] = out
            f.flush()
        tmp.rename(dst)
        print(f"[latent_dc] task {tid} {dst.name}: {len(dks)} demos in {time.time()-t0:.1f}s", flush=True)
    print("[latent_dc] DONE", flush=True)


if __name__ == "__main__":
    main()
