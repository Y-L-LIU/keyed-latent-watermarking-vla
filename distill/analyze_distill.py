#!/usr/bin/env python3.11
"""Distillation-survival verdict from obs-tied rollout NPZs.

For a DISTILLED student we care about the PLAIN rollouts (clean sampler): if the
keyed warp was baked into the student's weights, inverting its plain actions
through the BASE recovers a latent that still correlates with r(k*,o). We score
this with the paper's verifier replicated offline:

  per chunk:   recovered noise z_hat  (chunk_recovered_noise, from base-detector MAP)
  reference:   r(k, o) regenerated from chunk_observation_state via the obs-tied key
  per-episode: s_e(k) = sum_chunks <r_k, z_hat>            (raw matched filter)
  calibrate:   Z_e(k*) = (s_e(k*) - mean_j s_e(k_j)) / std_j s_e(k_j)   over 32 decoys
  aggregate:   T_G(k) = sum_{e in G} Z_e(k);  TPR@1% vs the decoy null of T_G

Compares the obs-tied student against the clean-teacher control student (and,
optionally, the teacher). A surviving watermark => obs-tied plain Z_e(k*) >> 0,
clean-control plain Z_e(k*) ~ 0, and TPR@1% rising with |G|.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

from openpi.policies import watermark as wm


def episode_scores(npz_path, *, secret_key, q, proj_dims, n_decoys, variant_dims, signal_key="chunk_recovered_noise"):
    """Return s_e for the true key and each decoy key for one episode NPZ.

    signal_key selects the channel the obs-tied reference is matched against:
      chunk_recovered_noise  -> LATENT arm (base-detector MAP recovered seed, D=32)
      chunk_observed_actions -> OUTPUT arm (executed action, D=7); decoy calibration
                                cancels the common-mode task action, isolating the bias.
    """
    d = np.load(npz_path)
    selected = d["chunk_selected"]
    zhat = d[signal_key]                        # (n_chunks, H, D)
    states = d["chunk_observation_state"]       # (n_chunks, state_dim)
    n_chunks, H, D = zhat.shape
    keys = [int(secret_key)] + [int(secret_key) + 1 + j for j in range(n_decoys)]
    sums = {k: 0.0 for k in keys}
    used = 0
    for c in range(n_chunks):
        if not bool(selected[c]):
            continue
        obs_seed = wm.compute_obs_seed(states[c], quantization=q, proj_dims=proj_dims)
        z = np.nan_to_num(np.asarray(zhat[c], dtype=np.float64))   # (H, D)
        used += 1
        for k in keys:
            cfg = wm.InternalNoiseWatermarkConfig(
                secret_key=k, control_freq=20.0, beta=1.0, reference_mode="gaussian",
                watermark_dims=tuple(range(D)),
            )
            ctx = wm.WatermarkContext(obs_seed=obs_seed)
            r = wm.generate_keyed_reference(length=H, action_dim=D, sample_rate_hz=20.0, config=cfg, context=ctx)
            r = np.asarray(r, dtype=np.float64)
            # normalize per chunk so episode sum is a clean matched-filter accumulation
            rn = r / (np.linalg.norm(r) + 1e-8)
            sums[k] += float(np.sum(rn * z))
    if used == 0:
        return None
    s_true = sums[keys[0]]
    s_decoy = np.array([sums[k] for k in keys[1:]], dtype=np.float64)
    mu, sd = s_decoy.mean(), s_decoy.std()
    z_true = (s_true - mu) / (sd + 1e-8)
    return dict(s_true=s_true, s_decoy=s_decoy, z_true=z_true)


def collect(rollout_dir, variant, *, signal_key="chunk_recovered_noise", **kw):
    paths = sorted(glob.glob(str(pathlib.Path(rollout_dir) / f"*_{variant}.npz")))
    out = []
    for p in paths:
        r = episode_scores(p, signal_key=signal_key, **kw)
        if r is not None:
            out.append(r)
    return out


def bootstrap_tpr(pos_z, null_z, group_sizes, n_boot=2000, fpr=0.01, seed=0):
    """TPR@fpr for aggregated T_G, with the decoy-null built from each episode's decoy spread."""
    rng = np.random.default_rng(seed)
    pos_z = np.asarray(pos_z);
    res = {}
    for g in group_sizes:
        if len(pos_z) == 0:
            res[g] = float("nan"); continue
        # positive: sum of g resampled Z_e(k*); null: sum of g resampled Z under H0 (std normal)
        pos = np.array([pos_z[rng.integers(0, len(pos_z), g)].sum() for _ in range(n_boot)])
        null = rng.standard_normal((n_boot, g)).sum(axis=1)  # decoy-calibrated Z ~ N(0,1) under H0
        thr = np.quantile(null, 1 - fpr)
        res[g] = float(np.mean(pos > thr))
    return res


def summarize(name, eps):
    if not eps:
        print(f"  {name:22s}: (no episodes)"); return None
    z = np.array([e["z_true"] for e in eps])
    print(f"  {name:22s}: n={len(z):3d}  Z_e(k*) mean={z.mean():+.2f} std={z.std():.2f}  "
          f"median={np.median(z):+.2f}  frac>2={np.mean(z>2):.2f}")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstied-rollouts", required=True)
    ap.add_argument("--clean-rollouts", default=None)
    ap.add_argument("--teacher-rollouts", default=None)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--n-decoys", type=int, default=32)
    ap.add_argument("--signal-key", default="chunk_recovered_noise",
                    choices=["chunk_recovered_noise", "chunk_observed_actions"],
                    help="latent arm: recovered_noise; output arm: observed_actions")
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))
    kw = dict(secret_key=args.secret_key, q=args.q, proj_dims=proj, n_decoys=args.n_decoys,
              variant_dims=None, signal_key=args.signal_key)

    print("=== DISTILLATION-SURVIVAL VERDICT (obs-tied, base-detector MAP) ===")
    print(f"key={args.secret_key} q={args.q} proj={proj}\n")
    print("Per-episode calibrated Z_e(k*) (true key vs 32 decoys):")
    obs_plain = summarize("obstied student PLAIN", collect(args.obstied_rollouts, "plain", **kw))
    if args.clean_rollouts:
        clean_plain = summarize("clean   student PLAIN", collect(args.clean_rollouts, "plain", **kw))
    else:
        clean_plain = None
    if args.teacher_rollouts:
        summarize("teacher WATERMARKED", collect(args.teacher_rollouts, "watermarked", **kw))
        summarize("teacher PLAIN", collect(args.teacher_rollouts, "plain", **kw))

    if obs_plain is not None:
        print("\nAggregated detection TPR@1% (decoy-calibrated null):")
        tpr = bootstrap_tpr(obs_plain, clean_plain, group_sizes=[1, 4, 8, 16, 32])
        for g, t in tpr.items():
            print(f"  |G|={g:2d}: TPR@1%={t:.3f}")
        if clean_plain is not None:
            # cross-student AUC at true key (obstied-plain positive vs clean-plain null)
            from itertools import product
            wins = sum(a > b for a, b in product(obs_plain, clean_plain))
            auc = wins / (len(obs_plain) * len(clean_plain))
            print(f"\nCross-student AUC (obstied-plain vs clean-plain, true key) = {auc:.3f}")


if __name__ == "__main__":
    main()
