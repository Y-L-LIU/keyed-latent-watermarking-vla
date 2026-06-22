#!/usr/bin/env python3.11
"""Entropy-sweep latent-DC relabel: a CONSTANT (non-zero-mean) per-chunk offset whose
KEY INDEX is the observation bucket modulo N_KEYS -- everything identical to
relabel_latent_dc.py except the seed source, so N_KEYS is a pure entropy knob.

  relabel_latent_dc.py      : c = dc_offset(key, prompt_seed(prompt))   -> ~10 distinct (per-task)  LOW entropy
  this script (--n-keys N)  : c = dc_offset(key, obs_bucket % N)        -> N distinct (per obs region)

Persistence is held fixed: c is tiled constant over the H-horizon (non-zero-mean),
exactly like the surviving latent-DC arm. Only the number of distinct constant
offsets the student must memorize changes. So a drop in survival as N rises
isolates ENTROPY (key learnability) from persistence -- the axis the two-point
table (high-entropy/zero-mean vs low-entropy/DC) could not separate.

z^fp = sqrt(1-b^2) z + b r_seed, injected as noise= to the CLEAN base (paper's seed
site). Observations stay byte-identical; only actions change.
"""
from __future__ import annotations
import argparse, pathlib, shutil, time, sys
import h5py, numpy as np
sys.path.insert(0, "/workspace/vla/distill")
import dc_keying
from openpi.training.config import get_config
from openpi.training.data_loader import _read_libero_image, _read_libero_state
import openpi.policies.policy_config as policy_config
from openpi.policies import watermark as wm
from libero.libero import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-keys", type=int, required=True, help="cardinality of the DC key lookup (entropy knob)")
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--task-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-demos", type=int, default=None)
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))

    cfg = get_config("pi05_libero")
    H = int(cfg.model.action_horizon); Draw = int(cfg.model.action_dim)
    policy = policy_config.create_trained_policy(cfg, "/workspace/vla/models/pi05_libero")
    print(f"[ldc-obskey] base loaded H={H} Draw={Draw} beta={args.beta} N_KEYS={args.n_keys} q={args.q} proj={proj}", flush=True)

    def key_index(state):
        # high-cardinality observation bucket folded into N_KEYS classes
        return int(wm.compute_obs_seed(state, quantization=args.q, proj_dims=proj) % args.n_keys)

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
            print(f"[ldc-obskey] {dst.name} exists, skip", flush=True); continue
        tmp = dst.with_suffix(".tmp.hdf5"); shutil.copyfile(src, tmp); t0 = time.time(); seen = set()
        with h5py.File(tmp, "r+") as f:
            dks = sorted(f["data"].keys())
            if args.max_demos:
                dks = dks[: args.max_demos]
            for dk in dks:
                obs = f["data"][dk]["obs"]; acts = f["data"][dk]["actions"]
                T = int(acts.shape[0]); ad = int(acts.shape[1]); out = np.zeros_like(acts[...])
                s = 0
                while s < T:
                    st = _read_libero_state(obs, s)
                    idx = key_index(st); seen.add(idx)
                    # DC seed for this obs region: constant in time, non-zero-mean, keyed on (key, obs-bucket-mod-N)
                    c = dc_keying.dc_offset(args.secret_key, idx, Draw)             # (Draw,)
                    r_seed = np.tile(c[None, :], (H, 1)).astype(np.float32)         # (H, Draw)
                    o = {"observation/image": _read_libero_image(obs, s, ("agentview_image", "agentview_rgb")),
                         "observation/wrist_image": _read_libero_image(obs, s, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")),
                         "observation/state": st, "prompt": prompt}
                    z = np.random.standard_normal((H, Draw)).astype(np.float32)
                    zfp = (np.sqrt(max(0.0, 1 - args.beta**2)) * z + args.beta * r_seed).astype(np.float32)
                    chunk = np.asarray(policy.infer(o, noise=zfp)["actions"], dtype=np.float32)
                    n = min(H, T - s, chunk.shape[0]); out[s:s+n] = chunk[:n, :ad]
                    s += H
                acts[...] = out
            f.flush()
        tmp.rename(dst)
        print(f"[ldc-obskey] task {tid} {dst.name}: {len(dks)} demos, {len(seen)} distinct keys hit, {time.time()-t0:.1f}s", flush=True)
    print("[ldc-obskey] DONE", flush=True)


if __name__ == "__main__":
    main()
