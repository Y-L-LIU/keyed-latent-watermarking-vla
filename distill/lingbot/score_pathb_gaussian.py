#!/usr/bin/env python3.11
"""Score LingBot Path-B zero-mean gaussian seed distillation.

This is the deployed-style counterpart to score_pathb.py. It reads MAP-recovered
student noise from eval_pathb_detection.py and matches it against the zero-mean
gaussian reference keyed by the chunk-start observation bucket.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/workspace/vla/lingbot-va")
from wan_va.wm.watermark import (  # noqa: E402
    InternalNoiseWatermarkConfig,
    WatermarkContext,
    compute_obs_seed,
    generate_keyed_reference,
)

ACTIVE = list(range(7))
D = len(ACTIVE)
F, H = 4, 4
LENGTH = F * H


def _ref(key: int, bucket: int) -> np.ndarray:
    cfg = InternalNoiseWatermarkConfig(
        secret_key=int(key),
        control_freq=float(LENGTH),
        beta=1.0,
        reference_mode="gaussian",
        keying_mode="obs",
    )
    ctx = WatermarkContext(obs_seed=int(bucket))
    r = generate_keyed_reference(
        length=LENGTH,
        action_dim=D,
        sample_rate_hz=float(LENGTH),
        config=cfg,
        context=ctx,
    )
    r = np.asarray(r, dtype=np.float64)
    return r / (np.linalg.norm(r) + 1e-8)


def episode_z(npz, key, ndec, n_keys, q, proj):
    d = np.load(npz)
    zhat = d["chunk_recovered_noise"]
    states = d["chunk_observation_state"]
    if zhat.ndim < 2 or len(zhat) == 0:
        return None
    keys = [key] + [key + 1 + j for j in range(ndec)]
    sums = {k: 0.0 for k in keys}
    used = 0
    for c in range(len(zhat)):
        z = np.nan_to_num(np.asarray(zhat[c], dtype=np.float64))
        if z.ndim == 4:
            z = z[..., 0]
        z2 = z[ACTIVE].reshape(D, LENGTH).T
        st = np.asarray(states[c], dtype=np.float64)
        bucket = int(compute_obs_seed(st, quantization=q, proj_dims=proj) % n_keys)
        used += 1
        for k in keys:
            sums[k] += float(np.sum(_ref(k, bucket) * z2))
    if used == 0:
        return None
    dec = np.array([sums[k] for k in keys[1:]], dtype=np.float64)
    return (sums[key] - dec.mean()) / (dec.std() + 1e-8), bool(d["success"])


def run(d, key, ndec, n_keys, q, proj):
    zs, sr = [], []
    for p in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
        r = episode_z(p, key, ndec, n_keys, q, proj)
        if r:
            zs.append(r[0])
            sr.append(r[1])
    return np.array(zs), np.array(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-rollouts", required=True)
    ap.add_argument("--clean-rollouts", default=None)
    ap.add_argument("--n-keys", type=int, required=True)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--n-decoys", type=int, default=32)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    a = ap.parse_args()
    proj = tuple(int(x) for x in a.proj_dims.split(","))
    print(f"=== PATH B detection (zero-mean gaussian, obs-bucket mod N_KEYS={a.n_keys}) ===")
    zw, sw = run(a.wm_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
    print(f"watermarked student: n={len(zw)} Z mean={zw.mean():+.2f} std={zw.std():.2f} | "
          f"rollout SR={sw.mean():.3f}")
    if a.clean_rollouts:
        zc, sc = run(a.clean_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
        print(f"clean       student: n={len(zc)} Z mean={zc.mean():+.2f} std={zc.std():.2f} | "
              f"rollout SR={sc.mean():.3f}")
        auc = sum(x > y for x in zw for y in zc) / max(len(zw) * len(zc), 1)
        print(f"cross-student AUC (wm vs clean) = {auc:.3f}")
    print(f"VERDICT: {'SURVIVES' if len(zw) and zw.mean() > 0.5 else 'does NOT survive'}")


if __name__ == "__main__":
    main()
