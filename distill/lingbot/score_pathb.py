"""PATH B detection scorer for LingBot-VA -- the score_latentdc_obskey.py analog.

Reads per-episode NPZs from eval_pathb_detection.py (chunk_recovered_noise, chunk_observation_state)
and computes, per episode, the raw matched-filter Z of the true DC key vs same-observation/wrong-key
decoys, accumulated over the MAP'd chunks (no mean-subtraction -- the cosine in scoring.py would
zero out a DC reference). Cross-student AUC (watermarked student Z vs clean student Z) is the
distillation-survival metric, identical to pi0.5.

  ref(k, state) = dc_offset(k, compute_obs_seed(state, q, proj) % N_KEYS, D) tiled over H, L2-normed
  Z_episode     = (sum_chunks ref(key)·z  -  decoys.mean) / decoys.std
  retention     = sum_chunks ref(key)·z  /  sum_chunks ref(key)·ref(key)

Usage:
  python3.11 score_pathb.py --wm-rollouts outputs/pathb_det/n40 \
      --clean-rollouts outputs/pathb_det/clean --n-keys 40
"""
from __future__ import annotations
import argparse, glob, pathlib, sys
import numpy as np
sys.path.insert(0, "/workspace/vla/distill")
sys.path.insert(0, "/workspace/vla/lingbot-va")
import dc_keying
from wan_va.wm.watermark import compute_obs_seed

ACTIVE = list(range(7))   # used_action_channel_ids (libero)
D = len(ACTIVE)
F, H = 4, 4
LENGTH = F * H            # 16


def episode_Z(npz, key, ndec, n_keys, q, proj):
    # self-generated NPZs (eval_pathb_detection.py); only numpy arrays/scalars -> no pickle needed
    d = np.load(npz)
    zhat = d["chunk_recovered_noise"]
    states = d["chunk_observation_state"]
    if zhat.ndim < 2 or len(zhat) == 0:
        return None
    keys = [key] + [key + 1 + j for j in range(ndec)]
    sums = {k: 0.0 for k in keys}
    rET = 0.0; bb = 0.0; used = 0
    for c in range(len(zhat)):
        z = np.nan_to_num(np.asarray(zhat[c], dtype=np.float64))   # [30, F, H, 1]
        if z.ndim == 4:
            z = z[..., 0]
        z2 = z[ACTIVE].reshape(D, LENGTH).T                         # [16, 7]
        st = np.asarray(states[c], dtype=np.float64)
        idx = int(compute_obs_seed(st, quantization=q, proj_dims=proj) % n_keys)
        used += 1
        def ref(k):
            c_ = dc_keying.dc_offset(k, idx, D)
            r = np.tile(c_[None, :], (LENGTH, 1))
            return r / (np.linalg.norm(r) + 1e-8)
        for k in keys:
            sums[k] += float(np.sum(ref(k) * z2))
        rt = ref(key)
        rET += float(np.sum(rt * z2)); bb += float(np.sum(rt * rt))
    if used == 0:
        return None
    dec = np.array([sums[k] for k in keys[1:]])
    Z = (sums[key] - dec.mean()) / (dec.std() + 1e-8)
    return Z, rET / (bb + 1e-9), bool(d["success"])


def run(d, key, ndec, n_keys, q, proj):
    zs, rs, sr = [], [], []
    for p in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
        r = episode_Z(p, key, ndec, n_keys, q, proj)
        if r:
            zs.append(r[0]); rs.append(r[1]); sr.append(r[2])
    return np.array(zs), np.array(rs), np.array(sr)


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
    print(f"=== PATH B detection (DC obs-bucket mod N_KEYS={a.n_keys}, raw matched filter) ===")
    zl, rl, sl = run(a.wm_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
    print(f"watermarked student: n={len(zl)} Z mean={zl.mean():+.2f} std={zl.std():.2f} | "
          f"seed-retention mean={rl.mean():+.3f} | rollout SR={sl.mean():.3f}")
    if a.clean_rollouts:
        zc, rc, sc = run(a.clean_rollouts, a.secret_key, a.n_decoys, a.n_keys, a.q, proj)
        print(f"clean       student: n={len(zc)} Z mean={zc.mean():+.2f} std={zc.std():.2f} | "
              f"seed-retention mean={rc.mean():+.3f} | rollout SR={sc.mean():.3f}")
        from itertools import product
        auc = sum(x > y for x, y in product(zl, zc)) / max(len(zl) * len(zc), 1)
        print(f"cross-student AUC (wm vs clean) = {auc:.3f}")
    rng = np.random.default_rng(0)
    for g in [1, 4, 8, 16]:
        if len(zl):
            pos = np.array([zl[rng.integers(0, len(zl), g)].sum() for _ in range(3000)])
            null = rng.standard_normal((3000, g)).sum(1)
            print(f"  |G|={g:2d}: TPR@1%={np.mean(pos > np.quantile(null, 0.99)):.3f}")
    print(f"VERDICT: {'SURVIVES' if len(zl) and zl.mean() > 0.5 else 'does NOT survive'}")


if __name__ == "__main__":
    main()
