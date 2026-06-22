#!/usr/bin/env python3.11
"""Base-residual detector for the OUTPUT (perceptible) distillation arm.

The owner holds the base policy. For a suspect student's plain rollout, the owner
recomputes the base action on each recorded observation and forms the residual
  residual(o) = student_executed_action(o) - base_action(o)
which removes the common-mode task action, leaving the keyed output bias
beta_out * r_out(k*, bucket(o)) plus small task drift. Correlating the residual
with r_out and calibrating against 32 decoys isolates the key. This is the
natural detector for an output-space watermark (cf. the latent arm's MAP detector).
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from openpi.training.config import get_config
import openpi.policies.policy_config as policy_config
from openpi.policies import watermark as wm


def base_action(policy, img, wrist, state, prompt):
    obs = {
        "observation/image": np.asarray(img, dtype=np.uint8),
        "observation/wrist_image": np.asarray(wrist, dtype=np.uint8),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": str(prompt),
    }
    return np.asarray(policy.infer(obs)["actions"], dtype=np.float64)  # (H,7)


def episode_Z(npz, policy, *, secret_key, q, proj, n_decoys):
    d = np.load(npz)
    sel = d["chunk_selected"]; acts = d["chunk_observed_actions"]
    imgs = d["chunk_observation_image"]; wrists = d["chunk_observation_wrist_image"]
    states = d["chunk_observation_state"]; prompts = d["chunk_prompt"]
    keys = [int(secret_key)] + [int(secret_key) + 1 + j for j in range(n_decoys)]
    sums = {k: 0.0 for k in keys}; used = 0
    for c in range(len(sel)):
        if not bool(sel[c]):
            continue
        y = np.asarray(acts[c], dtype=np.float64)              # (H,7)
        ab = base_action(policy, imgs[c], wrists[c], states[c], prompts[c])
        H = min(y.shape[0], ab.shape[0]); D = y.shape[1]
        resid = y[:H] - ab[:H]                                  # (H,7) base residual
        obs_seed = wm.compute_obs_seed(states[c], quantization=q, proj_dims=proj)
        used += 1
        for k in keys:
            cfg = wm.InternalNoiseWatermarkConfig(secret_key=k, control_freq=20.0, beta=1.0,
                                                  reference_mode="gaussian", watermark_dims=tuple(range(D)))
            r = np.asarray(wm.generate_keyed_reference(length=H, action_dim=D, sample_rate_hz=20.0,
                            config=cfg, context=wm.WatermarkContext(obs_seed=obs_seed)), dtype=np.float64)
            rn = r / (np.linalg.norm(r) + 1e-8)
            sums[k] += float(np.sum(rn * resid))
    if used == 0:
        return None
    sd = np.array([sums[k] for k in keys[1:]])
    return (sums[keys[0]] - sd.mean()) / (sd.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-rollouts", required=True)
    ap.add_argument("--clean-rollouts", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", default="/workspace/vla/models/pi05_libero")
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--n-decoys", type=int, default=32)
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))
    policy = policy_config.create_trained_policy(get_config(args.config_name), args.checkpoint_dir)

    def run(d):
        zs = []
        for p in sorted(glob.glob(str(pathlib.Path(d) / "*_plain.npz"))):
            z = episode_Z(p, policy, secret_key=args.secret_key, q=args.q, proj=proj, n_decoys=args.n_decoys)
            if z is not None:
                zs.append(z)
        return np.array(zs)

    print("=== OUTPUT-ARM base-residual detector ===")
    zo = run(args.output_rollouts); zc = run(args.clean_rollouts)
    print(f"output student PLAIN: n={len(zo)} Z mean={zo.mean():+.2f} std={zo.std():.2f} frac>2={np.mean(zo>2):.2f}")
    print(f"clean  student PLAIN: n={len(zc)} Z mean={zc.mean():+.2f} std={zc.std():.2f} frac>2={np.mean(zc>2):.2f}")
    from itertools import product
    auc = sum(a > b for a, b in product(zo, zc)) / (len(zo) * len(zc))
    print(f"cross-student AUC (output-plain vs clean-plain) = {auc:.3f}")
    rng = np.random.default_rng(0)
    for g in [1, 4, 8, 16, 32]:
        pos = np.array([zo[rng.integers(0, len(zo), g)].sum() for _ in range(3000)])
        null = rng.standard_normal((3000, g)).sum(1)
        print(f"  |G|={g:2d}: TPR@1%={np.mean(pos > np.quantile(null, 0.99)):.3f}")


if __name__ == "__main__":
    main()
