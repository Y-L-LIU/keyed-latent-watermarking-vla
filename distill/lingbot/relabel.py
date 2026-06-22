"""STAGE 1: build a relabeled LingBot latent dataset for one arm (dc | hash).

The relabeled dataset symlinks the heavy assets (latents/, videos/, empty_emb.pt) and
copies only meta/ + a rewritten data/ where the parquet `action` column = clean_base_action
+ bias. The data loader reads actions from these parquets (HF dataset over data/*.parquet),
slicing actions[start:end] at all phases -> a constant-in-time DC offset survives BC, while
a zero-mean obs-tied reference averages to ~0.

Bias definitions (raw 7-dim action space, beta_out=0.1):
  dc:   new_a[t] = a[t] + dc_keying.dc_bias(key, prompt, H, 7, beta_out)[0]   (const per task)
  hash: new_a[t] = a[t] + beta_out * r(key, obs_seed(eef_pos[t]))             (obs-tied gaussian)

Usage:
  python3.11 relabel.py --arm dc   --out /workspace/vla/lingbot_latents/relabel_dc
  python3.11 relabel.py --arm hash --out /workspace/vla/lingbot_latents/relabel_hash
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/vla/lingbot-va")
sys.path.insert(0, "/workspace/vla/distill")

import dc_keying
from wan_va.wm.watermark import (
    compute_obs_seed, WatermarkContext, generate_keyed_reference,
    InternalNoiseWatermarkConfig,
)

SRC = "/workspace/vla/lingbot_latents/libero_long"
KEY = 42
BETA_OUT = 0.1
ACTION_DIM = 7
HORIZON = 16  # frame_chunk_size(4) * action_per_frame(4)
QUANT = 0.08
PROJ = (0, 1, 2)


def hash_ref_for_seed(obs_seed: int) -> np.ndarray:
    """First-row keyed gaussian (7,) for a given obs bucket seed. length=1 -> per-bucket vec."""
    cfg = InternalNoiseWatermarkConfig(
        secret_key=KEY, control_freq=float(HORIZON), beta=BETA_OUT,
        reference_mode="gaussian", keying_mode="obs",
        obs_proj_dims=PROJ, obs_quantization=QUANT,
    )
    ctx = WatermarkContext(obs_seed=int(obs_seed))
    # length=1 so generate_keyed_reference returns a single normalized draw per dim;
    # _generate_gaussian_reference subtracts mean over length -> with length=1 it is 0.
    # We instead draw a length-2 reference and take row 0 (gives a nonzero per-bucket vec).
    ref = generate_keyed_reference(
        length=2, action_dim=ACTION_DIM, sample_rate_hz=float(HORIZON),
        config=cfg, context=ctx,
    )  # (2, 7)
    return ref[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dc", "hash", "hashmod", "chunkdc"], required=True)
    ap.add_argument("--n-keys", type=int, default=0,
                    help="for hashmod/chunkdc: fold obs bucket into N_KEYS classes (pure cardinality knob)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.arm in ("hashmod", "chunkdc") and args.n_keys <= 0:
        ap.error(f"{args.arm} requires --n-keys > 0")

    out = args.out
    if os.path.exists(out):
        print(f"[relabel] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(out)

    # symlink heavy assets
    for name in ["latents", "videos", "empty_emb.pt"]:
        os.symlink(os.path.join(SRC, name), os.path.join(out, name))
    # copy meta (small)
    shutil.copytree(os.path.join(SRC, "meta"), os.path.join(out, "meta"))
    # also bring over .gitattributes/README harmlessly skipped

    # task index -> prompt
    tasks = {}
    with open(os.path.join(SRC, "meta", "tasks.jsonl")) as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]

    # rewrite data/ parquets
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    src_parquets = sorted(glob.glob(os.path.join(SRC, "data", "*", "*.parquet")))
    print(f"[relabel] arm={args.arm} rewriting {len(src_parquets)} parquet files")

    # precompute per-task DC vec cache
    dc_cache = {}
    rmses = []
    n_buckets = set()
    for pq in src_parquets:
        rel = os.path.relpath(pq, os.path.join(SRC, "data"))
        dst = os.path.join(out, "data", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        df = pd.read_parquet(pq)
        a = np.stack(df["action"].values).astype(np.float64)  # (T,7)
        state = np.stack(df["observation.state"].values).astype(np.float64)  # (T,8)
        task_idx = int(df["task_index"].iloc[0])
        prompt = tasks[task_idx]

        if args.arm == "dc":
            if task_idx not in dc_cache:
                dc_cache[task_idx] = dc_keying.dc_bias(KEY, prompt, HORIZON, ACTION_DIM, BETA_OUT)[0]
            bias = np.tile(dc_cache[task_idx][None, :], (a.shape[0], 1))  # (T,7) const
        elif args.arm == "chunkdc":
            # CORRECTED obs-keyed DC (the faithful pi0.5 analog): per H-aligned chunk a
            # CONSTANT offset keyed on the chunk-START observation bucket (mod N_KEYS).
            # Constant within the chunk + a function of the conditioning the policy sees
            # -> non-zero E[bias|obs] -> learnable by BC (unlike per-timestep hashmod).
            bias = np.zeros_like(a)
            for w in range(0, a.shape[0], HORIZON):
                bucket = compute_obs_seed(state[w], quantization=QUANT, proj_dims=PROJ) % args.n_keys
                n_buckets.add(bucket)
                c = BETA_OUT * dc_keying.dc_offset(KEY, bucket, ACTION_DIM)  # (7,) const per bucket, beta-scaled
                bias[w:w + HORIZON] = c[None, :]
        else:  # hash / hashmod: per-row obs-tied gaussian (hashmod folds the bucket mod N_KEYS)
            bias = np.zeros_like(a)
            for t in range(a.shape[0]):
                seed = compute_obs_seed(state[t], quantization=QUANT, proj_dims=PROJ)
                if args.arm == "hashmod":
                    seed = seed % args.n_keys   # pure cardinality knob, same vector function as hash
                n_buckets.add(seed)
                bias[t] = BETA_OUT * hash_ref_for_seed(seed)

        new_a = (a + bias).astype(np.float32)
        rmse = float(np.sqrt(((new_a - a) ** 2).mean()))
        rmses.append(rmse)
        df["action"] = list(new_a)
        df.to_parquet(dst)

    print(f"[relabel] mean per-episode rmse(new vs clean) = {np.mean(rmses):.4f} "
          f"(min {np.min(rmses):.4f} max {np.max(rmses):.4f})")
    if args.arm in ("hash", "hashmod", "chunkdc"):
        print(f"[relabel] distinct obs buckets across corpus = {len(n_buckets)}")
        # report DC component of hash bias = per-task time-mean (should be ~small for hash)
    # quick DC-of-hash diagnostic on episode 0
    df0 = pd.read_parquet(src_parquets[0])
    a0 = np.stack(df0["action"].values).astype(np.float64)
    if args.arm in ("hash", "hashmod"):
        st0 = np.stack(df0["observation.state"].values).astype(np.float64)
        b0 = np.stack([BETA_OUT * hash_ref_for_seed(
            compute_obs_seed(st0[t], quantization=QUANT, proj_dims=PROJ)) for t in range(len(st0))])
        print(f"[relabel] ep0 hash-bias time-mean (DC leak) L2 = {np.linalg.norm(b0.mean(0)):.4f} "
              f"vs per-step L2 = {np.linalg.norm(b0, axis=1).mean():.4f}")
    print(f"[relabel] DONE -> {out}")


if __name__ == "__main__":
    main()
