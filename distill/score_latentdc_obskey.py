#!/usr/bin/env python3.11
"""Entropy-sweep latent-DC detector (obs-bucket-mod-N DC, paper's seed site).

Mirror of score_latentdc.py, but the per-chunk DC reference is keyed on the chunk's
recorded observation bucket folded into N_KEYS classes -- matching relabel_latent_dc_obskey.py.

  ref(k, state) = dc_offset(k, compute_obs_seed(state, q, proj) % N_KEYS, D), tiled over H

The student rolls out plain; the eval MAP-recovers the seed through the BASE
(chunk_recovered_noise). We correlate that recovered seed with the DC reference,
calibrated against `--n-decoys` wrong keys (same observation, wrong key). Cross-student
AUC vs the clean-control student is the survival metric, identical pipeline to the
AUC=0.89 low-entropy (per-task) latent-DC point.
"""
from __future__ import annotations
import argparse, glob, pathlib, sys
import numpy as np
sys.path.insert(0, "/workspace/vla/distill")
import dc_keying
from openpi.policies import watermark as wm


def ref(k, state, H, D, n_keys, q, proj):
    idx = int(wm.compute_obs_seed(state, quantization=q, proj_dims=proj) % n_keys)
    c = dc_keying.dc_offset(k, idx, D)
    r = np.tile(c[None, :], (H, 1))
    return r / (np.linalg.norm(r) + 1e-8)


def episode(npz, key, ndec, n_keys, q, proj):
    d = np.load(npz)
    sel = d["chunk_selected"]; zhat = d["chunk_recovered_noise"]; states = d["chunk_observation_state"]
    keys = [key] + [key + 1 + j for j in range(ndec)]
    sums = {k: 0.0 for k in keys}; rET = 0.0; bb = 0.0; used = 0
    for c in range(len(sel)):
        if not bool(sel[c]):
            continue
        z = np.nan_to_num(np.asarray(zhat[c], dtype=np.float64)); H, D = z.shape
        st = np.asarray(states[c], dtype=np.float64)
        used += 1
        for k in keys:
            sums[k] += float(np.sum(ref(k, st, H, D, n_keys, q, proj) * z))
        rt = ref(key, st, H, D, n_keys, q, proj)
        rET += float(np.sum(rt * z)); bb += float(np.sum(rt * rt))
    if used == 0:
        return None
    dec = np.array([sums[k] for k in keys[1:]])
    Z = (sums[key] - dec.mean()) / (dec.std() + 1e-8)
    return Z, rET / (bb + 1e-9)


def run(d, key, ndec, n_keys, q, proj):
    zs, rs = [], []
    for p in sorted(glob.glob(str(pathlib.Path(d) / "*_plain.npz"))):
        r = episode(p, key, ndec, n_keys, q, proj)
        if r:
            zs.append(r[0]); rs.append(r[1])
    return np.array(zs), np.array(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latentdc-rollouts", required=True)
    ap.add_argument("--clean-rollouts", default=None)
    ap.add_argument("--n-keys", type=int, required=True)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--n-decoys", type=int, default=32)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    a = ap.parse_args()
    proj = tuple(int(x) for x in a.proj_dims.split(","))
    print(f"=== LATENT-DC entropy detector (obs-bucket mod N_KEYS={a.n_keys}, seed site) ===")
    zl, rl = run(a.latentdc_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
    print(f"latentdc-N{a.n_keys} student PLAIN: n={len(zl)} Z_e mean={zl.mean():+.2f} std={zl.std():.2f} | seed-retention mean={rl.mean():+.3f}")
    if a.clean_rollouts:
        zc, rc = run(a.clean_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
        print(f"clean         student PLAIN: n={len(zc)} Z_e mean={zc.mean():+.2f} std={zc.std():.2f} | seed-retention mean={rc.mean():+.3f}")
        from itertools import product
        auc = sum(x > y for x, y in product(zl, zc)) / max(len(zl) * len(zc), 1)
        print(f"cross-student AUC (latentdc-N{a.n_keys} vs clean) = {auc:.3f}")
    rng = np.random.default_rng(0)
    for g in [1, 4, 8, 16, 32]:
        if len(zl):
            pos = np.array([zl[rng.integers(0, len(zl), g)].sum() for _ in range(3000)])
            null = rng.standard_normal((3000, g)).sum(1)
            print(f"  |G|={g:2d}: TPR@1%={np.mean(pos > np.quantile(null, 0.99)):.3f}")
    print(f"VERDICT: {'SURVIVES' if len(zl) and zl.mean()>0.5 else 'does NOT survive (key too high-entropy to memorize)'}")


if __name__ == "__main__":
    main()
