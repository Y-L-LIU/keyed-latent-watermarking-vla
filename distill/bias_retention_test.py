#!/usr/bin/env python3.11
"""Decisive distillation test on matched observations (no rollout needed).

Runs the base and a distilled student on the SAME demo observations and measures
how much of the intended keyed bias the student reproduces:

  residual(o)      = student_action(o) - base_action(o)
  intended_bias(o) = beta_out * r_out(k*, seed(o))     [seed per --keying]
  retention        = <residual, bias> / <bias, bias>   (1=fully learned, 0=not)

Also reports the decoy-calibrated Z (true key vs 32 decoys) on the residual.
Compares keying=output (hash of ee_pos, unlearnable) vs output_task (instruction,
learnable) to pin distillation survival to keying learnability.
"""
from __future__ import annotations
import argparse, glob, hashlib
import numpy as np
from openpi.training.config import get_config
from openpi.training.data_loader import _read_libero_image, _read_libero_state
import openpi.policies.policy_config as pc
from openpi.policies import watermark as wm
import h5py


import dc_keying


def prompt_seed(p): return int.from_bytes(hashlib.blake2b(str(p).encode(), digest_size=8).digest(), "little")


def reference(keying, k, state, prompt, H, D, q, proj):
    """Unit (beta=1) keyed reference for key k on this chunk, shape (H,D)."""
    if keying == "output_task_dc":
        c = dc_keying.dc_offset(k, dc_keying.prompt_seed(prompt), D)
        return np.tile(c[None, :], (H, 1))
    seed = prompt_seed(prompt) if keying == "output_task" else wm.compute_obs_seed(state, quantization=q, proj_dims=proj)
    cfg = wm.InternalNoiseWatermarkConfig(secret_key=k, control_freq=20.0, beta=1.0,
                                          reference_mode="gaussian", watermark_dims=tuple(range(D)))
    return np.asarray(wm.generate_keyed_reference(length=H, action_dim=D, sample_rate_hz=20.0,
                      config=cfg, context=wm.WatermarkContext(obs_seed=seed)), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-config", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--keying", choices=["output", "output_task", "output_task_dc"], required=True)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--beta-out", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--n-decoys", type=int, default=32)
    ap.add_argument("--demos-per-task", type=int, default=2)
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))
    base = pc.create_trained_policy(get_config("pi05_libero"), "/workspace/vla/models/pi05_libero")
    stu = pc.create_trained_policy(get_config(args.student_config), args.student_ckpt)

    R, B = [], []
    sums = {k: 0.0 for k in [args.secret_key] + [args.secret_key + 1 + j for j in range(args.n_decoys)]}
    nchunks = 0
    for f in sorted(glob.glob("/workspace/vla/openpi/third_party/libero/libero/datasets/libero_goal/*.hdf5")):
        with h5py.File(f, "r") as h:
            # task language for prompt-keying must match what the student trained on; use the LIBERO benchmark prompt
            from libero.libero import benchmark
            suite = benchmark.get_benchmark_dict()["libero_goal"]()
            # map file -> task language by matching demonstration filename
            tname = f.split("/")[-1]
            prompt = next((suite.get_task(t).language for t in range(suite.n_tasks)
                           if suite.get_task_demonstration(t).split("/")[-1] == tname), "do task")
            for dk in sorted(h["data"].keys())[: args.demos_per_task]:
                obs = h["data"][dk]["obs"]; T = h["data"][dk]["actions"].shape[0]
                for s in range(0, T, 10):
                    st = _read_libero_state(obs, s)
                    o = {"observation/image": _read_libero_image(obs, s, ("agentview_image", "agentview_rgb")),
                         "observation/wrist_image": _read_libero_image(obs, s, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")),
                         "observation/state": st, "prompt": prompt}
                    ab = np.asarray(base.infer(o)["actions"], dtype=np.float64)
                    au = np.asarray(stu.infer(o)["actions"], dtype=np.float64)
                    H = min(len(ab), len(au)); D = ab.shape[1]
                    resid = au[:H] - ab[:H]
                    rtrue = reference(args.keying, args.secret_key, st, prompt, H, D, args.q, proj)
                    R.append(resid.ravel()); B.append((args.beta_out * rtrue).ravel())
                    nchunks += 1
                    for k in sums:
                        rk = reference(args.keying, k, st, prompt, H, D, args.q, proj)
                        rn = rk / (np.linalg.norm(rk) + 1e-8)
                        sums[k] += float(np.sum(rn * resid))
    R = np.concatenate(R); B = np.concatenate(B)
    retention = float(np.dot(R, B) / (np.dot(B, B) + 1e-9))
    corr = float(np.corrcoef(R, B)[0, 1])
    decoys = np.array([sums[k] for k in list(sums)[1:]])
    Z = (sums[args.secret_key] - decoys.mean()) / (decoys.std() + 1e-8)
    print(f"=== bias-retention: keying={args.keying} ckpt={args.student_ckpt.split('/')[-2]} ===")
    print(f"chunks={nchunks}  residual_rms={np.sqrt(np.mean(R**2)):.4f}  bias_rms={np.sqrt(np.mean(B**2)):.4f}")
    print(f"retention coefficient = {retention:+.3f}   corr(residual,bias) = {corr:+.3f}")
    print(f"decoy-calibrated Z (true vs {args.n_decoys} decoys, all chunks) = {Z:+.2f}")
    print(f"VERDICT: {'SURVIVES (learnable key)' if retention > 0.3 else 'does NOT survive (key not learned)'}")


if __name__ == "__main__":
    main()
