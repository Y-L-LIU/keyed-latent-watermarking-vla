#!/usr/bin/env python3.11
"""Latent-DC detector: does the seed-injected DC mark survive distillation?

The student rolls out plain; the eval MAP-recovers the seed through the BASE
(chunk_recovered_noise). We correlate that recovered seed with the per-task DC seed
reference r_DC(k, task) = dc_offset(k, prompt_seed, Draw) tiled over H, calibrated
against 32 decoys. This is the paper's own latent detector, faithful to the seed
injection site. Compares the latent-DC student to the clean-control student.
"""
from __future__ import annotations
import argparse, glob, pathlib, sys
import numpy as np
sys.path.insert(0, "/workspace/vla/distill")
import dc_keying


def ref(k, prompt, H, D):
    c = dc_keying.dc_offset(k, dc_keying.prompt_seed(prompt), D)
    r = np.tile(c[None, :], (H, 1))
    return r / (np.linalg.norm(r) + 1e-8)


def episode(npz, key, ndec):
    d = np.load(npz)
    sel = d["chunk_selected"]; zhat = d["chunk_recovered_noise"]; prompts = d["chunk_prompt"]
    keys = [key] + [key + 1 + j for j in range(ndec)]
    sums = {k: 0.0 for k in keys}; rET = 0.0; bb = 0.0; used = 0
    for c in range(len(sel)):
        if not bool(sel[c]):
            continue
        z = np.nan_to_num(np.asarray(zhat[c], dtype=np.float64)); H, D = z.shape
        used += 1
        for k in keys:
            sums[k] += float(np.sum(ref(k, str(prompts[c]), H, D) * z))
        # retention vs unit-norm true ref in seed space
        rt = ref(key, str(prompts[c]), H, D)
        rET += float(np.sum(rt * z)); bb += float(np.sum(rt * rt))
    if used == 0:
        return None
    dec = np.array([sums[k] for k in keys[1:]])
    Z = (sums[key] - dec.mean()) / (dec.std() + 1e-8)
    return Z, rET / (bb + 1e-9)


def run(d, key, ndec):
    zs, rs = [], []
    for p in sorted(glob.glob(str(pathlib.Path(d) / "*_plain.npz"))):
        r = episode(p, key, ndec)
        if r:
            zs.append(r[0]); rs.append(r[1])
    return np.array(zs), np.array(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latentdc-rollouts", required=True)
    ap.add_argument("--clean-rollouts", default=None)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--n-decoys", type=int, default=32)
    a = ap.parse_args()
    print("=== LATENT-DC detector (faithful: seed injection site) ===")
    zl, rl = run(a.latentdc_rollouts, a.secret_key, a.n_decoys)
    print(f"latent-DC student PLAIN: n={len(zl)} Z_e mean={zl.mean():+.2f} std={zl.std():.2f} | seed-retention mean={rl.mean():+.3f}")
    if a.clean_rollouts:
        zc, rc = run(a.clean_rollouts, a.secret_key, a.n_decoys)
        print(f"clean    student PLAIN: n={len(zc)} Z_e mean={zc.mean():+.2f} std={zc.std():.2f} | seed-retention mean={rc.mean():+.3f}")
        from itertools import product
        auc = sum(x > y for x, y in product(zl, zc)) / max(len(zl) * len(zc), 1)
        print(f"cross-student AUC (latentdc-plain vs clean-plain) = {auc:.3f}")
    rng = np.random.default_rng(0)
    for g in [1, 4, 8, 16, 32]:
        if len(zl):
            pos = np.array([zl[rng.integers(0, len(zl), g)].sum() for _ in range(3000)])
            null = rng.standard_normal((3000, g)).sum(1)
            print(f"  |G|={g:2d}: TPR@1%={np.mean(pos > np.quantile(null, 0.99)):.3f}")
    print(f"VERDICT: {'SURVIVES' if len(zl) and zl.mean()>0.5 else 'does NOT survive (seed-insensitivity attenuates it)'}")


if __name__ == "__main__":
    main()
