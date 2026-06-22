"""STAGE 3/4: bias-retention test on a trained LingBot student.

retention = <(student(o) - base(o)), bias(o)> / <bias(o), bias(o)>   (averaged over obs)
  ~1  -> bias fully learned (SURVIVES distillation)
  ~0  -> not learned (does NOT survive)

Matched-observation protocol: decode the first frame of each sampled episode, run the
server _infer at frame_st_id=0 with an IDENTICAL per-obs base-noise seed for base and
student, so the residual isolates the learned mapping difference (not noise variance).

The intended bias is computed IDENTICALLY to the relabel injection (same dc_keying /
watermark primitives), at beta_out=0.1, key=42:
  dc   : dc_keying.dc_bias(key, prompt, H, 7, 0.1)               -> (H,7) const-in-time
  hash : 0.1 * generate_keyed_reference(obs_seed(eef_pos_0))[0]  -> (7,) broadcast over H

A decoy-calibrated Z is reported: retention numerator projected on the true-key bias vs
~16 decoy keys (and, for hash, decoy obs-seeds), Z = (r_true - mean_decoy)/std_decoy.

Run (1 GPU):
  CUDA_VISIBLE_DEVICES=2 STUDENT_CKPT=.../transformer ARM=dc N_EPS=40 \
    torchrun --nproc_per_node=1 --master_port 29532 retention_test.py
"""
from __future__ import annotations
import os, sys, json, time, glob
import numpy as np
import torch

sys.path.insert(0, "/workspace/vla/lingbot-va")
sys.path.insert(0, "/workspace/vla/distill")

import dc_keying
from wan_va.wm.watermark import (
    compute_obs_seed, WatermarkContext, generate_keyed_reference,
    InternalNoiseWatermarkConfig,
)

DATA = "/workspace/vla/lingbot_latents/libero_long"
BASE_CKPT = "/workspace/vla/models/lingbot-va-posttrain-libero-long"
KEY = 42
BETA_OUT = 0.1
ACTION_DIM = 7
HORIZON = 16
QUANT = 0.08
PROJ = (0, 1, 2)
N_DECOY = 16


def decode_first_frame(ep_idx, chunk=0):
    import av
    obs = {}
    for key in ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]:
        path = f"{DATA}/videos/chunk-{chunk:03d}/{key}/episode_{ep_idx:06d}.mp4"
        c = av.open(path)
        for frame in c.decode(video=0):
            obs[key] = frame.to_ndarray(format="rgb24")
            break
        c.close()
    return obs


def dc_bias_for(prompt, key=KEY):
    return dc_keying.dc_bias(key, prompt, HORIZON, ACTION_DIM, BETA_OUT)  # (H,7)


def hash_ref_for(obs_seed, key=KEY):
    cfg = InternalNoiseWatermarkConfig(
        secret_key=key, control_freq=float(HORIZON), beta=BETA_OUT,
        reference_mode="gaussian", keying_mode="obs",
        obs_proj_dims=PROJ, obs_quantization=QUANT)
    ctx = WatermarkContext(obs_seed=int(obs_seed))
    ref = generate_keyed_reference(length=2, action_dim=ACTION_DIM,
                                   sample_rate_hz=float(HORIZON), config=cfg, context=ctx)
    return BETA_OUT * ref[0]  # (7,)


def build_server(ckpt_path):
    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    cfg = VA_CONFIGS["libero"]
    cfg.wan22_pretrained_model_name_or_path = ckpt_path
    cfg.rank = int(os.environ.get("RANK", 0))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.enable_offload = True
    cfg.save_root = "/workspace/vla/lingbot_out/distill_relabel/_retn_scratch"
    os.makedirs(cfg.save_root, exist_ok=True)
    return VA_Server(cfg)


def infer_chunk(server, prompt, obs, seed):
    """Return postprocessed action chunk flattened to (H,7) with a fixed base-noise seed."""
    server._reset(prompt=prompt)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        actions_np, _ = server._infer({"obs": [obs]}, frame_st_id=0)  # (7,F,H)
    a = actions_np  # (7,4,4)
    a = a.reshape(ACTION_DIM, -1).T  # (H=16, 7)
    return a.astype(np.float64)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    arm = os.environ["ARM"]
    if arm.startswith("hashmod"):
        arm = "hashmod"  # ckpt/json names carry the N suffix; logic keys off N_KEYS
    elif arm.startswith("chunkdc"):
        arm = "chunkdc"
    N_KEYS = int(os.environ.get("N_KEYS", 0))
    if arm in ("hashmod", "chunkdc") and N_KEYS <= 0:
        raise SystemExit(f"{arm} requires N_KEYS>0")
    # PERSTEP=1: key the reference per timestep (state[t]'s own bucket) instead of the
    # first-frame bucket broadcast over H. The first-frame metric caps the obs-tied
    # ceiling at ~0.62 (only ~62% of in-chunk steps share frame0's bucket) and is blind
    # to cardinality; per-step keying restores ceiling 1.0 so the entropy effect shows.
    PERSTEP = int(os.environ.get("PERSTEP", 0))
    student_ckpt = os.environ["STUDENT_CKPT"]  # dir containing transformer/ (assembled)
    n_eps = int(os.environ.get("N_EPS", 40))
    out_json = os.environ.get("OUT_JSON", f"/workspace/vla/distill/lingbot/retention_{arm}.json")

    # tasks
    tasks = {}
    with open(f"{DATA}/meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]
    import pandas as pd

    # sample episodes spread across the corpus (deterministic)
    all_eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                     for p in glob.glob(f"{DATA}/data/chunk-000/episode_*.parquet"))
    rng = np.random.default_rng(0)
    sample_eps = sorted(rng.choice(all_eps, size=min(n_eps, len(all_eps)), replace=False).tolist())
    print(f"[retn] arm={arm} n_eps={len(sample_eps)} student={student_ckpt}")

    print("[retn] building BASE server...")
    base_server = build_server(BASE_CKPT)
    print("[retn] running base over sample...")

    # cache per-episode (prompt, obs, eef_pos, base_action)
    recs = []
    for i, ep in enumerate(sample_eps):
        df = pd.read_parquet(f"{DATA}/data/chunk-000/episode_{ep:06d}.parquet")
        task_idx = int(df["task_index"].iloc[0]); prompt = tasks[task_idx]
        st_all = np.stack(df["observation.state"].values).astype(np.float64)
        eef0 = st_all[0]                       # (8,)
        stateH = st_all[:HORIZON]              # (<=H,8) per-step states for per-step keying
        obs = decode_first_frame(ep)
        seed = 100000 + ep
        a_base = infer_chunk(base_server, prompt, obs, seed)
        recs.append(dict(ep=ep, task_idx=task_idx, prompt=prompt, eef0=eef0, stateH=stateH,
                         obs=obs, seed=seed, a_base=a_base))
        if (i + 1) % 10 == 0:
            print(f"  base {i+1}/{len(sample_eps)}")
    del base_server
    torch.cuda.empty_cache()

    print("[retn] building STUDENT server...")
    stu_server = build_server(student_ckpt)
    print("[retn] running student over sample...")
    for i, r in enumerate(recs):
        r["a_stu"] = infer_chunk(stu_server, r["prompt"], r["obs"], r["seed"])
        if (i + 1) % 10 == 0:
            print(f"  student {i+1}/{len(recs)}")
    del stu_server
    torch.cuda.empty_cache()

    # --- retention ---
    # numerator = <resid, bias>, denom = <bias,bias>, both summed over (H,7)
    def retention_for_key(key, use_decoy_seed=False, decoy_idx=0):
        num = 0.0; den = 0.0
        for r in recs:
            resid = r["a_stu"] - r["a_base"]  # (H,7)
            if PERSTEP and arm in ("hashmod", "hash"):
                # per-timestep reference: each step keyed by its OWN state's bucket,
                # matching the per-timestep injection (ceiling 1.0). Align to resid length.
                Ht = min(HORIZON, resid.shape[0], r["stateH"].shape[0])
                bias = np.zeros((resid.shape[0], resid.shape[1]))
                for t in range(Ht):
                    st = compute_obs_seed(r["stateH"][t], quantization=QUANT, proj_dims=PROJ)
                    if arm == "hashmod":
                        st = st % N_KEYS
                    bias[t] = hash_ref_for(st, key=key)
                num += float((resid[:Ht] * bias[:Ht]).sum())
                den += float((bias[:Ht] * bias[:Ht]).sum())
                continue
            if arm == "dc":
                bias = dc_bias_for(r["prompt"], key=key)  # (H,7)
            elif arm == "chunkdc":
                # CORRECTED obs-keyed DC: per-chunk constant offset keyed on the chunk-start
                # (first-frame) observation bucket mod N_KEYS -> matches the chunkdc injection,
                # ceiling 1.0 with this per-chunk reference.
                s = compute_obs_seed(r["eef0"], quantization=QUANT, proj_dims=PROJ) % N_KEYS
                v = BETA_OUT * dc_keying.dc_offset(key, s, ACTION_DIM)  # (7,) beta-scaled to match injection
                bias = np.tile(v[None, :], (HORIZON, 1))
            elif arm == "hashmod":
                # same obs-tied gaussian as hash, bucket folded mod N_KEYS (cardinality sweep)
                s = compute_obs_seed(r["eef0"], quantization=QUANT, proj_dims=PROJ) % N_KEYS
                v = hash_ref_for(s, key=key)
                bias = np.tile(v[None, :], (HORIZON, 1))
            else:
                if use_decoy_seed:
                    # decoy obs-seed: perturb the bucket so the obs-tie is wrong
                    base_seed = compute_obs_seed(r["eef0"], quantization=QUANT, proj_dims=PROJ)
                    seedp = compute_obs_seed(r["eef0"] + (decoy_idx + 1) * 0.5,
                                             quantization=QUANT, proj_dims=PROJ)
                    v = hash_ref_for(seedp, key=key)
                else:
                    s = compute_obs_seed(r["eef0"], quantization=QUANT, proj_dims=PROJ)
                    v = hash_ref_for(s, key=key)
                bias = np.tile(v[None, :], (HORIZON, 1))
            num += float((resid * bias).sum())
            den += float((bias * bias).sum())
        return num / den, num, den

    retn_true, num_true, den_true = retention_for_key(KEY)

    # decoy calibration: wrong keys (and for hash, also wrong obs-seeds)
    decoy_retentions = []
    decoy_keys = [KEY + 1000 + j for j in range(N_DECOY)]
    for j, dk in enumerate(decoy_keys):
        if arm in ("dc", "hashmod", "chunkdc"):
            rr, _, _ = retention_for_key(dk)
        else:
            rr, _, _ = retention_for_key(dk, use_decoy_seed=True, decoy_idx=j)
        decoy_retentions.append(rr)
    decoy_retentions = np.array(decoy_retentions)
    z = (retn_true - decoy_retentions.mean()) / (decoy_retentions.std() + 1e-9)

    # also report residual rms and bias rms for context
    resid_rms = float(np.sqrt(np.mean([(r["a_stu"] - r["a_base"])**2 for r in recs])))
    base_rms = float(np.sqrt(np.mean([r["a_base"]**2 for r in recs])))

    result = dict(
        arm=arm, n_keys=N_KEYS, perstep=PERSTEP, n_eps=len(recs), key=KEY, beta_out=BETA_OUT,
        retention=retn_true, survives=bool(retn_true > 0.3),
        decoy_mean=float(decoy_retentions.mean()),
        decoy_std=float(decoy_retentions.std()),
        z=float(z), n_decoy=N_DECOY,
        resid_rms=resid_rms, base_rms=base_rms,
        student_ckpt=student_ckpt,
    )
    print("\n==================== RETENTION RESULT ====================")
    print(json.dumps(result, indent=2))
    print("=========================================================")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[retn] wrote {out_json}")


if __name__ == "__main__":
    main()
