"""Offline diagnosis of LingBot distillation-retention non-monotonicity.

Computes the metric-intrinsic CEILING (replace student residual with the ACTUAL
injected bias) for dc / hashmod40 / hashmod160 / hash, over the SAME 40 sampled
episodes the retention test uses, plus several follow-on diagnostics.

No GPU / model needed.
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
    "dc":        "/workspace/vla/lingbot_latents/relabel_dc",
    "hash":      "/workspace/vla/lingbot_latents/relabel_hash",
    "hashmod40": "/workspace/vla/lingbot_latents/relabel_hashmod40",
    "hashmod160":"/workspace/vla/lingbot_latents/relabel_hashmod160",
}
NMOD = {"hashmod40": 40, "hashmod160": 160}

KEY = 42
BETA_OUT = 0.1
ACTION_DIM = 7
HORIZON = 16
QUANT = 0.08
PROJ = (0, 1, 2)
N_DECOY = 16


# ---- exact reference primitives (copied verbatim from retention_test/relabel) ----
def dc_bias_for(prompt, key=KEY):
    return dc_keying.dc_bias(key, prompt, HORIZON, ACTION_DIM, BETA_OUT)  # (H,7)


def hash_ref_for(obs_seed, key=KEY):
    """retention_test.hash_ref_for : BETA_OUT * generate_keyed_reference(length=2)[0]."""
    cfg = InternalNoiseWatermarkConfig(
        secret_key=key, control_freq=float(HORIZON), beta=BETA_OUT,
        reference_mode="gaussian", keying_mode="obs",
        obs_proj_dims=PROJ, obs_quantization=QUANT)
    ctx = WatermarkContext(obs_seed=int(obs_seed))
    ref = generate_keyed_reference(length=2, action_dim=ACTION_DIM,
                                   sample_rate_hz=float(HORIZON), config=cfg, context=ctx)
    return BETA_OUT * ref[0]  # (7,)


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


# ---------------------------------------------------------------------------
def get_sample_eps():
    all_eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                     for p in glob.glob(f"{CLEAN}/episode_*.parquet"))
    rng = np.random.default_rng(0)
    sample_eps = sorted(rng.choice(all_eps, size=min(40, len(all_eps)),
                                   replace=False).tolist())
    return sample_eps


def load_tasks():
    tasks = {}
    with open(f"{DATA}/meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]
    return tasks


def load_ep(arm, ep):
    """Return (clean_action (T,7), relabel_action (T,7), state (T,8), task_idx)."""
    cdf = pd.read_parquet(f"{CLEAN}/episode_{ep:06d}.parquet")
    rdf = pd.read_parquet(f"{RELDIR[arm]}/data/chunk-000/episode_{ep:06d}.parquet")
    a_clean = np.stack(cdf["action"].values).astype(np.float64)
    a_rel   = np.stack(rdf["action"].values).astype(np.float64)
    state   = np.stack(cdf["observation.state"].values).astype(np.float64)
    task_idx = int(cdf["task_index"].iloc[0])
    return a_clean, a_rel, state, task_idx


# ---------------------------------------------------------------------------
def ref_for_arm(arm, prompt, eef0, key=KEY):
    """Exactly the retention_test reference for the FIRST chunk, shape (H,7)."""
    if arm == "dc":
        return dc_bias_for(prompt, key=key)  # (H,7)
    if arm in ("hashmod40", "hashmod160"):
        N = NMOD[arm]
        s = compute_obs_seed(eef0, quantization=QUANT, proj_dims=PROJ) % N
        v = hash_ref_for(s, key=key)
        return np.tile(v[None, :], (HORIZON, 1))
    # hash full
    s = compute_obs_seed(eef0, quantization=QUANT, proj_dims=PROJ)
    v = hash_ref_for(s, key=key)
    return np.tile(v[None, :], (HORIZON, 1))


def ceiling_for_arm(arm, sample_eps, tasks, key=KEY, decoy_seed_shift=None):
    """Metric-intrinsic ceiling: <b[:H], ref>/<ref,ref> summed over episodes.

    b = relabel_action - clean_action over the first H steps (true injected bias).
    ref = retention_test reference for that arm (optionally with a decoy key /
    decoy obs-seed shift for the noise calibration).
    """
    num = 0.0; den = 0.0
    for ep in sample_eps:
        a_clean, a_rel, state, task_idx = load_ep(arm, ep)
        prompt = tasks[task_idx]
        eef0 = state[0]
        b = (a_rel - a_clean)[:HORIZON]  # (H,7) true injected bias
        if decoy_seed_shift is not None and arm != "dc":
            # mirror retention_test decoy obs-seed perturbation
            eef_for_ref = eef0 + decoy_seed_shift
        else:
            eef_for_ref = eef0
        ref = ref_for_arm(arm, prompt, eef_for_ref, key=key)
        num += float((b * ref).sum())
        den += float((ref * ref).sum())
    return num / den, num, den


# ---------------------------------------------------------------------------
def main():
    sample_eps = get_sample_eps()
    tasks = load_tasks()
    print(f"[diag] sampled {len(sample_eps)} episodes: {sample_eps[:8]}...{sample_eps[-3:]}")
    print(f"[diag] same rng(0).choice protocol as retention_test\n")

    arms = ["dc", "hashmod40", "hashmod160", "hash"]

    # ============ (5) SANITY / BUG CHECK first ============
    print("="*70)
    print("(5) SANITY: recompute injected bias and compare to relabel-clean diff")
    print("="*70)
    for arm in arms:
        ep = sample_eps[0]
        a_clean, a_rel, state, task_idx = load_ep(arm, ep)
        prompt = tasks[task_idx]
        actual = a_rel - a_clean  # (T,7)
        if arm == "dc":
            c = dc_keying.dc_bias(KEY, prompt, HORIZON, ACTION_DIM, BETA_OUT)[0]
            recomputed = np.tile(c[None, :], (a_clean.shape[0], 1))
        else:
            N = NMOD.get(arm, None)
            recomputed = np.zeros_like(a_clean)
            for t in range(a_clean.shape[0]):
                s = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ)
                if N is not None:
                    s = s % N
                recomputed[t] = BETA_OUT * hash_ref_for_seed_inject(s)
        maxerr = float(np.abs(actual - recomputed).max())
        # also check a second episode
        ep2 = sample_eps[5]
        a_clean2, a_rel2, state2, ti2 = load_ep(arm, ep2)
        prompt2 = tasks[ti2]
        actual2 = a_rel2 - a_clean2
        if arm == "dc":
            c2 = dc_keying.dc_bias(KEY, prompt2, HORIZON, ACTION_DIM, BETA_OUT)[0]
            recomputed2 = np.tile(c2[None, :], (a_clean2.shape[0], 1))
        else:
            N = NMOD.get(arm, None)
            recomputed2 = np.zeros_like(a_clean2)
            for t in range(a_clean2.shape[0]):
                s = compute_obs_seed(state2[t], quantization=QUANT, proj_dims=PROJ)
                if N is not None:
                    s = s % N
                recomputed2[t] = BETA_OUT * hash_ref_for_seed_inject(s)
        maxerr2 = float(np.abs(actual2 - recomputed2).max())
        print(f"  {arm:11s}: ep{ep} max|actual-recomputed|={maxerr:.2e}  "
              f"ep{ep2} max={maxerr2:.2e}  -> {'OK' if max(maxerr,maxerr2)<1e-6 else 'MISMATCH!'}")

    # confirm hashmod really used the right modulus: distinct buckets in corpus subset
    print("\n  modulus confirmation (distinct first-frame buckets over sample):")
    for arm in arms:
        N = NMOD.get(arm, None)
        seeds = set()
        for ep in sample_eps:
            _, _, state, _ = load_ep(arm, ep)
            s = compute_obs_seed(state[0], quantization=QUANT, proj_dims=PROJ)
            if N is not None:
                s = s % N
            seeds.add(s)
        print(f"    {arm:11s}: {len(seeds)} distinct first-frame buckets (N={N})")

    # ============ (DECISIVE) CEILING TABLE ============
    print("\n" + "="*70)
    print("DECISIVE: metric-intrinsic CEILING (perfect student reproducing teacher)")
    print("  ceiling = sum<b[:H],ref> / sum<ref,ref>  over the 40 sampled eps")
    print("="*70)
    ceilings = {}
    for arm in arms:
        c, num, den = ceiling_for_arm(arm, sample_eps, tasks)
        ceilings[arm] = c
        print(f"  {arm:11s}: ceiling = {c:+.4f}   (num={num:+.3f} den={den:.3f})")
    print(f"\n  MEASURED retention (from trained students): "
          f"dc=0.478  hashmod40=0.128  hash=0.206")
    mono_ceiling = ceilings["hashmod40"] >= ceilings["hashmod160"] >= ceilings["hash"]
    print(f"  ceiling monotone in cardinality (hashmod40>=hashmod160>=hash)? "
          f"{mono_ceiling}")
    print(f"  ceiling NON-monotone like measured (hashmod40 < hash)? "
          f"{ceilings['hashmod40'] < ceilings['hash']}")

    # ============ (1) WITHIN-CHUNK BUCKET STABILITY ============
    print("\n" + "="*70)
    print("(1) within-chunk bucket stability over first H=16 steps")
    print("="*70)
    for arm in arms:
        N = NMOD.get(arm, None)
        ndistinct_list = []; frac_first_list = []
        for ep in sample_eps:
            _, _, state, _ = load_ep(arm, ep)
            seeds = []
            for t in range(min(HORIZON, state.shape[0])):
                s = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ)
                if N is not None:
                    s = s % N
                seeds.append(s)
            seeds = np.array(seeds)
            ndistinct_list.append(len(set(seeds.tolist())))
            frac_first_list.append(float((seeds == seeds[0]).mean()))
        print(f"  {arm:11s}: mean #distinct buckets in first 16 = "
              f"{np.mean(ndistinct_list):.2f}   "
              f"mean frac sharing frame0 bucket = {np.mean(frac_first_list):.3f}")

    # ============ (2) IS THE DIFFERENCE NOISE? decoy spread at ceiling ============
    print("\n" + "="*70)
    print("(2) decoy-calibrated noise scale AT THE CEILING level")
    print("    (replicate retention_test decoy keys/obs-seeds, but project the")
    print("     TRUE injected bias -> isolates metric+reference noise floor)")
    print("="*70)
    for arm in arms:
        true_c, _, _ = ceiling_for_arm(arm, sample_eps, tasks, key=KEY)
        decoy_cs = []
        decoy_keys = [KEY + 1000 + j for j in range(N_DECOY)]
        for j, dk in enumerate(decoy_keys):
            if arm == "dc" or arm in NMOD:
                dc_, _, _ = ceiling_for_arm(arm, sample_eps, tasks, key=dk)
            else:
                # hash full: decoy obs-seed shift, matching retention_test
                shift = (j + 1) * 0.5
                dc_, _, _ = ceiling_for_arm(arm, sample_eps, tasks, key=dk,
                                            decoy_seed_shift=shift)
            decoy_cs.append(dc_)
        decoy_cs = np.array(decoy_cs)
        z = (true_c - decoy_cs.mean()) / (decoy_cs.std() + 1e-9)
        print(f"  {arm:11s}: true_ceiling={true_c:+.4f}  "
              f"decoy_mean={decoy_cs.mean():+.4f}  decoy_std={decoy_cs.std():.4f}  "
              f"Z={z:+.2f}")
    print("\n  -> Compare measured hashmod40=0.128 vs hash=0.206: gap=0.078.")
    print("     Is that within ~1 decoy-std of the ceiling-level noise?")

    # ============ (3) BIAS STRUCTURE: DC component vs per-step ============
    print("\n" + "="*70)
    print("(3) bias structure: time-mean (DC) L2 vs per-step L2")
    print("="*70)
    for arm in arms:
        firstH_dc = []; firstH_step = []; whole_dc = []; whole_step = []
        for ep in sample_eps:
            a_clean, a_rel, state, task_idx = load_ep(arm, ep)
            b = a_rel - a_clean  # (T,7)
            bH = b[:HORIZON]
            firstH_dc.append(np.linalg.norm(bH.mean(0)))
            firstH_step.append(np.linalg.norm(bH, axis=1).mean())
            whole_dc.append(np.linalg.norm(b.mean(0)))
            whole_step.append(np.linalg.norm(b, axis=1).mean())
        print(f"  {arm:11s}: firstH  DC-L2={np.mean(firstH_dc):.4f}  "
              f"perStep-L2={np.mean(firstH_step):.4f}  "
              f"ratio(DC/step)={np.mean(firstH_dc)/np.mean(firstH_step):.3f}")
        print(f"  {'':11s}  whole   DC-L2={np.mean(whole_dc):.4f}  "
              f"perStep-L2={np.mean(whole_step):.4f}  "
              f"ratio(DC/step)={np.mean(whole_dc)/np.mean(whole_step):.3f}")

    # ============ (4) ALT-CEILING: whole-trajectory per-window keying ============
    print("\n" + "="*70)
    print("(4) ALT detector ceiling: whole-traj, each H-window keyed by ITS OWN")
    print("    first-frame bucket. ceiling = sum<b_win,ref_win>/sum<ref_win,ref_win>")
    print("="*70)
    alt_ceilings = {}
    for arm in arms:
        N = NMOD.get(arm, None)
        num = 0.0; den = 0.0
        for ep in sample_eps:
            a_clean, a_rel, state, task_idx = load_ep(arm, ep)
            prompt = tasks[task_idx]
            b = a_rel - a_clean  # (T,7)
            T = b.shape[0]
            for w0 in range(0, T - HORIZON + 1, HORIZON):
                eef = state[w0]
                if arm == "dc":
                    ref = dc_bias_for(prompt)  # (H,7)
                else:
                    s = compute_obs_seed(eef, quantization=QUANT, proj_dims=PROJ)
                    if N is not None:
                        s = s % N
                    v = hash_ref_for(s)
                    ref = np.tile(v[None, :], (HORIZON, 1))
                bw = b[w0:w0+HORIZON]
                num += float((bw * ref).sum())
                den += float((ref * ref).sum())
        alt_ceilings[arm] = num / den
        print(f"  {arm:11s}: alt-ceiling = {num/den:+.4f}")
    alt_mono = alt_ceilings["hashmod40"] >= alt_ceilings["hashmod160"] >= alt_ceilings["hash"]
    print(f"\n  alt-ceiling monotone (hashmod40>=hashmod160>=hash)? {alt_mono}")

    # ============ EXTRA: per-step ceiling (NO horizon broadcast at all) ============
    print("\n" + "="*70)
    print("(4b) per-step ALT ceiling: ref keyed PER TIMESTEP (full info, both")
    print("     injection and detection see each step's own bucket)")
    print("="*70)
    perstep_ceilings = {}
    for arm in arms:
        N = NMOD.get(arm, None)
        num = 0.0; den = 0.0
        for ep in sample_eps:
            a_clean, a_rel, state, task_idx = load_ep(arm, ep)
            prompt = tasks[task_idx]
            b = a_rel - a_clean
            T = b.shape[0]
            for t in range(T):
                if arm == "dc":
                    ref_t = dc_keying.dc_bias(KEY, prompt, HORIZON, ACTION_DIM, BETA_OUT)[0]
                else:
                    s = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ)
                    if N is not None:
                        s = s % N
                    ref_t = hash_ref_for(s)  # (7,)
                num += float((b[t] * ref_t).sum())
                den += float((ref_t * ref_t).sum())
        perstep_ceilings[arm] = num / den
        print(f"  {arm:11s}: per-step-ceiling = {num/den:+.4f}")
    ps_mono = perstep_ceilings["hashmod40"] >= perstep_ceilings["hashmod160"] >= perstep_ceilings["hash"]
    print(f"\n  per-step ceiling monotone? {ps_mono}")
    print("\n[diag] DONE")


if __name__ == "__main__":
    main()
