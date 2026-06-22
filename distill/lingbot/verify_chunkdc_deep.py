"""Deep-dive on check C: WHY does chunkdc all-phase ratio drop to ~0.63, and is the
residual a real BC-learnable signal? Also anchor against the dc survivor (retention~0.48).

Decompositions:
 (D1) All-phase window-mean bias for chunkdc = average of (up to 2) adjacent constant
      chunk-offsets weighted by overlap. We measure how much of that window-mean is a
      pure function of the START-frame conditioning bucket (E[wbias|cond]) vs how much is
      "leakage" from the NEXT chunk's offset (which the start-frame conditioning does NOT
      fix). The learnable part is E[wbias|cond]; the rest averages toward zero across the
      many windows that share a start bucket but straddle different next-chunk offsets.

 (D2) Anchor: run the SAME E[bias|cond] estimator on the dc arm (per-task constant; the
      empirically-confirmed survivor at retention~0.48) and on a perfectly-learnable
      synthetic (bias = pure function of start bucket, constant per window). This tells us
      what ratio value corresponds to "survives".

 (D3) Conditioning-bucket DRIFT within the H window: for all-phase windows, how often does
      the start-frame bucket differ from buckets later in the window (the thing that makes
      the offset only partially a function of the conditioning the policy is given).
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

DATA  = "/workspace/vla/lingbot_latents/libero_long"
CLEAN = f"{DATA}/data/chunk-000"
RELDIR = {
    "chunkdc40":  "/workspace/vla/lingbot_latents/relabel_chunkdc40",
    "chunkdc160": "/workspace/vla/lingbot_latents/relabel_chunkdc160",
    "hashmod40":  "/workspace/vla/lingbot_latents/relabel_hashmod40",
    "dc":         "/workspace/vla/lingbot_latents/relabel_dc",
}
NKEYS = {"chunkdc40": 40, "chunkdc160": 160, "hashmod40": 40}
KEY = 42; BETA_OUT = 0.1; ACTION_DIM = 7; HORIZON = 16; QUANT = 0.08; PROJ = (0, 1, 2)


def get_sample_eps():
    all_eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                     for p in glob.glob(f"{CLEAN}/episode_*.parquet"))
    rng = np.random.default_rng(0)
    return sorted(rng.choice(all_eps, size=40, replace=False).tolist())


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
    return a_clean, a_rel, state, int(cdf["task_index"].iloc[0])


def cond_mean(arm, sample_eps, all_phase, fold):
    """E[window-mean bias | start bucket], count-weighted RMS = learnable_L2.
    Also returns total window-mean RMS (the BC *target* magnitude), so ratio
    learnable/target tells us the fraction of the target that is conditioning-explained."""
    N = NKEYS.get(arm)
    sums = {}
    tgt_sq = 0.0; nw = 0
    for ep in sample_eps:
        a_clean, a_rel, state, _ = load_ep(arm, ep)
        b = a_rel - a_clean; T = b.shape[0]
        starts = range(0, T - HORIZON + 1) if all_phase else range(0, T - HORIZON + 1, HORIZON)
        for s in starts:
            wb = b[s:s + HORIZON].mean(0)
            tgt_sq += float(np.dot(wb, wb)); nw += 1
            if arm == "dc":
                cond = 0  # per-task constant -> single bucket (all windows in an ep same task)
            else:
                raw = compute_obs_seed(state[s], quantization=QUANT, proj_dims=PROJ)
                cond = (raw % N) if fold else raw
            d = sums.setdefault((ep if arm == "dc" else 0, cond), [np.zeros(ACTION_DIM), 0])
            d[0] += wb; d[1] += 1
    learn_sq = 0.0; tot = 0
    for k, (ss, c) in sums.items():
        cm = ss / c; learn_sq += float(np.dot(cm, cm)) * c; tot += c
    learn_l2 = float(np.sqrt(learn_sq / tot))
    tgt_l2 = float(np.sqrt(tgt_sq / nw))
    return learn_l2, tgt_l2, len(sums)


def synthetic_perfect(sample_eps, fold=True, N=40):
    """A synthetic 'perfectly learnable' mark: bias[t] = beta*dc_offset(start_bucket(t//H chunk)),
    i.e. EXACTLY chunkdc, but we measure E[bias|cond] of the per-step bias itself (no window
    averaging) -> ratio should be ~1 by construction; sanity anchor for the estimator."""
    sums = {}; tot_sq = 0.0; n = 0
    for ep in sample_eps:
        _, _, state, _ = load_ep("chunkdc40", ep)
        T = state.shape[0]
        for t in range(T):
            w = (t // HORIZON) * HORIZON
            bucket = compute_obs_seed(state[w], quantization=QUANT, proj_dims=PROJ) % N
            bias_t = BETA_OUT * dc_keying.dc_offset(KEY, bucket, ACTION_DIM)
            tot_sq += float(np.dot(bias_t, bias_t)); n += 1
            cond = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ) % N
            d = sums.setdefault(cond, [np.zeros(ACTION_DIM), 0]); d[0] += bias_t; d[1] += 1
    ls = 0.0; tt = 0
    for k, (ss, c) in sums.items():
        cm = ss / c; ls += float(np.dot(cm, cm)) * c; tt += c
    return float(np.sqrt(ls / tt)), float(np.sqrt(tot_sq / n))


def drift(sample_eps, N=40):
    """All-phase windows: frac of windows where start bucket spans into a DIFFERENT
    chunk offset region, and mean #distinct chunk-offsets contributing to a window."""
    fr = []; ndist = []
    for ep in sample_eps:
        _, _, state, _ = load_ep("chunkdc40", ep)
        T = state.shape[0]
        # chunk-offset id for each timestep = its chunk start index
        chunk_of = np.array([(t // HORIZON) for t in range(T)])
        for s in range(0, T - HORIZON + 1):
            ids = set(chunk_of[s:s + HORIZON].tolist())
            ndist.append(len(ids))
            fr.append(1.0 if len(ids) > 1 else 0.0)
    return float(np.mean(fr)), float(np.mean(ndist))


def main():
    eps = get_sample_eps()
    tasks = load_tasks()
    print("D1/D2. E[bias|cond] learnable_L2, window-mean TARGET L2, and their ratio")
    print("       (ratio = fraction of the BC target that the start-conditioning explains)")
    print(f"  {'arm':11s} {'phase':10s} {'learn_L2':>9s} {'target_L2':>10s} "
          f"{'learn/target':>13s} {'#groups':>8s}")
    for arm in ["dc", "chunkdc40", "chunkdc160", "hashmod40"]:
        for ap in [False, True]:
            ll, tl, ng = cond_mean(arm, eps, all_phase=ap, fold=True)
            ph = "ALL-PHASE" if ap else "H-ALIGNED"
            print(f"  {arm:11s} {ph:10s} {ll:9.4f} {tl:10.4f} {ll/tl:13.3f} {ng:8d}")

    sl, st = synthetic_perfect(eps)
    print(f"\n  estimator sanity (per-step chunkdc bias, E[bias|cond]): "
          f"learn={sl:.4f} total={st:.4f} ratio={sl/st:.3f} (expect ~1.0)")

    fr, nd = drift(eps)
    print(f"\nD3. all-phase window chunk-straddle: frac windows spanning >1 chunk-offset = "
          f"{fr:.3f}, mean #distinct chunk-offsets per window = {nd:.3f}")
    print("    (= 1 - 1/H of windows straddle a boundary -> next-chunk offset is the")
    print("     part the start-frame conditioning does NOT fix -> averages out, leaving")
    print("     ~1/2 of windows clean + boundary windows partially explained.)")
    print("\n[deep] DONE")


if __name__ == "__main__":
    main()
