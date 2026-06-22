"""OFFLINE skeptical verification of the chunkdc LingBot watermark construction.

Checks A (bug), B (ceiling+null Z), C (THE FUNDAMENTAL TEST: E[bias|conditioning]),
for chunkdc40 / chunkdc160, contrasted against hashmod40. No GPU, no training.

Run:
  export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/distill:$PYTHONPATH
  /usr/bin/python3.11 verify_chunkdc.py
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/vla/lingbot-va")
sys.path.insert(0, "/workspace/vla/distill")

import dc_keying
from wan_va.wm.watermark import (
    compute_obs_seed, WatermarkContext, generate_keyed_reference,
    InternalNoiseWatermarkConfig,
)

DATA   = "/workspace/vla/lingbot_latents/libero_long"
CLEAN  = f"{DATA}/data/chunk-000"
RELDIR = {
    "chunkdc40":  "/workspace/vla/lingbot_latents/relabel_chunkdc40",
    "chunkdc160": "/workspace/vla/lingbot_latents/relabel_chunkdc160",
    "hashmod40":  "/workspace/vla/lingbot_latents/relabel_hashmod40",
}
NKEYS = {"chunkdc40": 40, "chunkdc160": 160, "hashmod40": 40}

KEY = 42
BETA_OUT = 0.1
ACTION_DIM = 7
HORIZON = 16
QUANT = 0.08
PROJ = (0, 1, 2)
N_DECOY = 16


# ---- primitives (verbatim from relabel/retention_test) ----
def hash_ref_for_seed_inject(obs_seed):
    """relabel.hash_ref_for_seed (NO beta): generate_keyed_reference(length=2)[0]."""
    cfg = InternalNoiseWatermarkConfig(
        secret_key=KEY, control_freq=float(HORIZON), beta=BETA_OUT,
        reference_mode="gaussian", keying_mode="obs",
        obs_proj_dims=PROJ, obs_quantization=QUANT)
    ctx = WatermarkContext(obs_seed=int(obs_seed))
    ref = generate_keyed_reference(length=2, action_dim=ACTION_DIM,
                                   sample_rate_hz=float(HORIZON), config=cfg, context=ctx)
    return ref[0]  # (7,)


def get_sample_eps():
    all_eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                     for p in glob.glob(f"{CLEAN}/episode_*.parquet"))
    rng = np.random.default_rng(0)
    return sorted(rng.choice(all_eps, size=min(40, len(all_eps)), replace=False).tolist())


def load_tasks():
    tasks = {}
    with open(f"{DATA}/meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]
    return tasks


def load_ep(arm, ep):
    cdf = pd.read_parquet(f"{CLEAN}/episode_{ep:06d}.parquet")
    rdf = pd.read_parquet(f"{RELDIR[arm]}/data/chunk-000/episode_{ep:06d}.parquet")
    a_clean = np.stack(cdf["action"].values).astype(np.float64)
    a_rel   = np.stack(rdf["action"].values).astype(np.float64)
    state   = np.stack(cdf["observation.state"].values).astype(np.float64)
    task_idx = int(cdf["task_index"].iloc[0])
    return a_clean, a_rel, state, task_idx


# ================= A. BUG CHECK =================
def recompute_bias(arm, a_clean, state, prompt):
    N = NKEYS[arm]
    bias = np.zeros_like(a_clean)
    if arm.startswith("chunkdc"):
        for w in range(0, a_clean.shape[0], HORIZON):
            bucket = compute_obs_seed(state[w], quantization=QUANT, proj_dims=PROJ) % N
            c = BETA_OUT * dc_keying.dc_offset(KEY, bucket, ACTION_DIM)
            bias[w:w + HORIZON] = c[None, :]
    else:  # hashmod
        for t in range(a_clean.shape[0]):
            seed = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ) % N
            bias[t] = BETA_OUT * hash_ref_for_seed_inject(seed)
    return bias


def check_A(sample_eps, tasks):
    print("=" * 74)
    print("A. BUG CHECK: recomputed injected bias vs (relabel - clean)")
    print("=" * 74)
    for arm in ["chunkdc40", "chunkdc160", "hashmod40"]:
        N = NKEYS[arm]
        maxerrs = []
        perstep_l2s = []
        const_in_chunk_maxdev = []   # max within-chunk deviation (should be 0 for chunkdc)
        buckets_used = set()
        chunkstart_keyed_ok = True
        for ep in sample_eps:
            a_clean, a_rel, state, task_idx = load_ep(arm, ep)
            prompt = tasks[task_idx]
            actual = a_rel - a_clean
            recomp = recompute_bias(arm, a_clean, state, prompt)
            maxerrs.append(float(np.abs(actual - recomp).max()))
            perstep_l2s.append(float(np.linalg.norm(actual, axis=1).mean()))
            T = actual.shape[0]
            for w in range(0, T, HORIZON):
                seg = actual[w:w + HORIZON]
                if seg.shape[0] == 0:
                    continue
                if arm.startswith("chunkdc"):
                    dev = float(np.abs(seg - seg[0:1]).max())
                    const_in_chunk_maxdev.append(dev)
                    bucket = compute_obs_seed(state[w], quantization=QUANT, proj_dims=PROJ) % N
                    buckets_used.add(int(bucket))
                    # verify keyed on chunk-START bucket: seg[0] == beta*dc_offset(KEY,bucket)
                    expect = BETA_OUT * dc_keying.dc_offset(KEY, int(bucket), ACTION_DIM)
                    # parquet stores float32 -> compare at float32 tolerance
                    if float(np.abs(seg[0] - expect).max()) > 1e-6:
                        chunkstart_keyed_ok = False
                else:
                    for t in range(seg.shape[0]):
                        b = compute_obs_seed(state[w + t], quantization=QUANT, proj_dims=PROJ) % N
                        buckets_used.add(int(b))
        maxerr = max(maxerrs)
        line = (f"  {arm:11s}: max|actual-recomp|={maxerr:.2e} "
                f"-> {'OK' if maxerr < 1e-6 else 'MISMATCH!'}")
        print(line)
        print(f"    {'':9s}  per-step bias L2 (mean) = {np.mean(perstep_l2s):.4f}  "
              f"(raw dc_offset L2~sqrt7=2.65 -> beta-scaled ~{BETA_OUT*np.sqrt(7):.3f}; "
              f"NOT raw ~{np.sqrt(7):.2f})")
        if arm.startswith("chunkdc"):
            print(f"    {'':9s}  CONST within H-chunk: max within-chunk deviation = "
                  f"{max(const_in_chunk_maxdev):.2e}  -> "
                  f"{'OK (constant, float32)' if max(const_in_chunk_maxdev)<1e-6 else 'NOT CONSTANT!'}")
            print(f"    {'':9s}  keyed on chunk-START bucket mod N: "
                  f"{'OK' if chunkstart_keyed_ok else 'WRONG KEYING!'}")
        print(f"    {'':9s}  distinct buckets used over sample = {len(buckets_used)} "
              f"(N={N})")
    print()


# ================= B. CEILING + NULL Z =================
def chunkdc_ref_firstchunk(eef0, N, key=KEY):
    """retention_test chunkdc reference (PERSTEP=0): first-frame bucket, tiled over H."""
    s = compute_obs_seed(eef0, quantization=QUANT, proj_dims=PROJ) % N
    v = BETA_OUT * dc_keying.dc_offset(key, s, ACTION_DIM)
    return np.tile(v[None, :], (HORIZON, 1))


def hashmod_ref_firstchunk(eef0, N, key=KEY):
    s = compute_obs_seed(eef0, quantization=QUANT, proj_dims=PROJ) % N
    cfg = InternalNoiseWatermarkConfig(
        secret_key=key, control_freq=float(HORIZON), beta=BETA_OUT,
        reference_mode="gaussian", keying_mode="obs",
        obs_proj_dims=PROJ, obs_quantization=QUANT)
    ctx = WatermarkContext(obs_seed=int(s))
    ref = generate_keyed_reference(length=2, action_dim=ACTION_DIM,
                                   sample_rate_hz=float(HORIZON), config=cfg, context=ctx)
    v = BETA_OUT * ref[0]
    return np.tile(v[None, :], (HORIZON, 1))


def ref_firstchunk(arm, eef0, key=KEY):
    N = NKEYS[arm]
    if arm.startswith("chunkdc"):
        return chunkdc_ref_firstchunk(eef0, N, key=key)
    return hashmod_ref_firstchunk(eef0, N, key=key)


def ceiling_for_arm(arm, sample_eps, tasks, key=KEY):
    """sum<b[:H], ref>/sum<ref,ref> with true injected bias b (perfect student)."""
    num = 0.0; den = 0.0
    for ep in sample_eps:
        a_clean, a_rel, state, task_idx = load_ep(arm, ep)
        eef0 = state[0]
        b = (a_rel - a_clean)[:HORIZON]
        ref = ref_firstchunk(arm, eef0, key=key)
        num += float((b * ref).sum()); den += float((ref * ref).sum())
    return num / den


def check_B(sample_eps, tasks):
    print("=" * 74)
    print("B. DETECTOR CEILING (PERSTEP=0 per-chunk ref) + decoy-key NULL Z")
    print("   ceiling = sum<bias[:H],ref>/sum<ref,ref> (true bias); should be ~1.0")
    print("=" * 74)
    print(f"  {'arm':11s} {'ceiling':>9s} {'decoy_mean':>11s} {'decoy_std':>10s} {'Z':>8s}")
    for arm in ["chunkdc40", "chunkdc160", "hashmod40"]:
        c = ceiling_for_arm(arm, sample_eps, tasks, key=KEY)
        decoys = []
        for j in range(N_DECOY):
            dk = KEY + 1000 + j
            decoys.append(ceiling_for_arm(arm, sample_eps, tasks, key=dk))
        decoys = np.array(decoys)
        z = (c - decoys.mean()) / (decoys.std() + 1e-9)
        print(f"  {arm:11s} {c:+9.4f} {decoys.mean():+11.4f} "
              f"{decoys.std():10.4f} {z:+8.2f}")
    print()


# ================= C. THE FUNDAMENTAL TEST: E[bias | conditioning] =================
def cond_mean_analysis(arm, sample_eps, tasks, all_phase, fold_modN):
    """Build (window, conditioning) pairs; group window-mean-bias by conditioning bucket;
    return L2 of the per-bucket conditional mean and the overall per-step bias L2.

    all_phase=False -> windows start at 0,H,2H,... (H-aligned)
    all_phase=True  -> windows start at every s=0..T-H (BC all-phase slicing)
    fold_modN: if True, conditioning bucket = obs_seed(state[s]) % N (matches injection key)
               if False, conditioning bucket = raw obs_seed(state[s]) (un-folded)
    """
    N = NKEYS[arm]
    bucket_sums = {}   # bucket -> [sum_window_bias (7,), count]
    perstep_sq_sum = 0.0
    perstep_n = 0
    for ep in sample_eps:
        a_clean, a_rel, state, task_idx = load_ep(arm, ep)
        b = a_rel - a_clean  # (T,7)
        T = b.shape[0]
        # accumulate overall per-step L2 (over all steps)
        perstep_sq_sum += float((np.linalg.norm(b, axis=1) ** 2).sum())
        perstep_n += T
        starts = range(0, T - HORIZON + 1) if all_phase else range(0, T - HORIZON + 1, HORIZON)
        for s in starts:
            win = b[s:s + HORIZON]               # (H,7)
            wbias = win.mean(axis=0)             # (7,) window-mean bias = BC target component
            raw = compute_obs_seed(state[s], quantization=QUANT, proj_dims=PROJ)
            cond = (raw % N) if fold_modN else raw
            if cond not in bucket_sums:
                bucket_sums[cond] = [np.zeros(ACTION_DIM), 0]
            bucket_sums[cond][0] += wbias
            bucket_sums[cond][1] += 1
    # per-bucket conditional mean, then the BC-learnable signal:
    # weight each bucket's |mean|^2 by its window count, take sqrt of the count-weighted
    # mean -> RMS magnitude of E[bias|cond] across windows (this is what BC can reproduce).
    tot_w = 0; learn_sq = 0.0
    n_buckets = len(bucket_sums)
    for cond, (ssum, cnt) in bucket_sums.items():
        cmean = ssum / cnt                       # E[window_bias | cond]
        learn_sq += (np.linalg.norm(cmean) ** 2) * cnt
        tot_w += cnt
    learnable_l2 = float(np.sqrt(learn_sq / tot_w))      # RMS of conditional-mean over windows
    perstep_l2 = float(np.sqrt(perstep_sq_sum / perstep_n))
    return learnable_l2, perstep_l2, n_buckets, tot_w


def check_C(sample_eps, tasks):
    print("=" * 74)
    print("C. THE FUNDAMENTAL TEST: E[bias | conditioning]  (BC-learnable signal)")
    print("   learnable_L2 = RMS over windows of E[window-mean-bias | obs-bucket(start)]")
    print("   ratio = learnable_L2 / overall per-step-bias L2")
    print("   conditioning bucket = obs_seed(state[window_start]) % N (folded)")
    print("=" * 74)
    for fold in [True, False]:
        tag = "FOLDED mod N" if fold else "UN-folded raw obs_seed"
        print(f"\n  --- conditioning = {tag} ---")
        print(f"  {'arm':11s} {'phase':10s} {'learn_L2':>9s} {'perstep_L2':>11s} "
              f"{'ratio':>7s} {'#buckets':>9s} {'#windows':>9s}")
        for arm in ["chunkdc40", "chunkdc160", "hashmod40"]:
            for all_phase in [False, True]:
                ll, pl, nb, nw = cond_mean_analysis(arm, sample_eps, tasks,
                                                    all_phase=all_phase, fold_modN=fold)
                ph = "ALL-PHASE" if all_phase else "H-ALIGNED"
                print(f"  {arm:11s} {ph:10s} {ll:9.4f} {pl:11.4f} "
                      f"{ll/pl:7.3f} {nb:9d} {nw:9d}")
    print()


def main():
    sample_eps = get_sample_eps()
    tasks = load_tasks()
    print(f"[verify] sampled {len(sample_eps)} eps (rng(0).choice): "
          f"{sample_eps[:6]}...{sample_eps[-3:]}\n")
    check_A(sample_eps, tasks)
    check_B(sample_eps, tasks)
    check_C(sample_eps, tasks)
    print("[verify] DONE")


if __name__ == "__main__":
    main()
