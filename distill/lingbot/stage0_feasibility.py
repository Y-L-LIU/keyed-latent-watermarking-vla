"""STAGE 0 feasibility: build VA_Server with base ckpt, decode a dataset video frame,
run one _infer to get a base action chunk; report horizon + action_dim. Also read+rewrite
one episode parquet `action` column round-trip.

Run:
  CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 29531 \
    wan_va/wm/_stage0.py   (we run via -m by path; see launcher)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

REPO = "/workspace/vla/lingbot-va"
sys.path.insert(0, REPO)
sys.path.insert(0, "/workspace/vla/distill")

DATA = "/workspace/vla/lingbot_latents/libero_long"


def decode_first_frames(ep_idx, chunk=0, n=1):
    """Decode first n frames of agentview + eye_in_hand for an episode -> obs dicts."""
    import av
    out_frames = {}
    for key in ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]:
        path = f"{DATA}/videos/chunk-{chunk:03d}/{key}/episode_{ep_idx:06d}.mp4"
        container = av.open(path)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
        container.close()
        out_frames[key] = np.stack(frames)  # (n,H,W,3) uint8
    # build list of per-timestep obs dicts
    obs_list = []
    for t in range(n):
        obs_list.append({k: out_frames[k][t] for k in out_frames})
    return obs_list


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS

    cfg = VA_CONFIGS["libero"]
    cfg.wan22_pretrained_model_name_or_path = os.environ.get(
        "BASE_CKPT", "/workspace/vla/models/lingbot-va-posttrain-libero-long")
    cfg.rank = rank; cfg.local_rank = local_rank; cfg.world_size = world_size
    cfg.enable_offload = True
    cfg.save_root = "/workspace/vla/lingbot_out/distill_relabel/_stage0_scratch"
    os.makedirs(cfg.save_root, exist_ok=True)
    print(f"[stage0] action_dim={cfg.action_dim} frame_chunk_size={cfg.frame_chunk_size} "
          f"action_per_frame={cfg.action_per_frame} used_ch={cfg.used_action_channel_ids}")

    print("[stage0] building server...")
    t0 = time.time()
    server = VA_Server(cfg)
    print(f"[stage0] server built in {time.time()-t0:.1f}s")

    # read tasks.jsonl to get a prompt for episode 0
    import json
    tasks = {}
    with open(f"{DATA}/meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]
    import pandas as pd
    df = pd.read_parquet(f"{DATA}/data/chunk-000/episode_000000.parquet")
    task_idx = int(df["task_index"].iloc[0])
    prompt = tasks[task_idx]
    print(f"[stage0] episode 0 task_index={task_idx} prompt={prompt!r}")

    obs_list = decode_first_frames(0, n=1)
    print(f"[stage0] decoded frame shapes: "
          f"{[(k, v.shape, v.dtype) for k, v in obs_list[0].items()]}")

    server._reset(prompt=prompt)
    # seed the base noise deterministically so it is reproducible
    torch.manual_seed(1234)
    with torch.no_grad():
        actions_np, _ = server._infer({"obs": [obs_list[0]]}, frame_st_id=0)
    raw = server._last_raw_actions  # [1, C, F, H, 1]
    print(f"[stage0] raw_actions shape={tuple(raw.shape)} dtype={raw.dtype}")
    print(f"[stage0] postprocessed action chunk shape={actions_np.shape}")  # (7, F, H)
    print(f"[stage0] action chunk (used ch) first few:\n{actions_np[:, 0, :2]}")

    # parquet round-trip test (do NOT overwrite the real file; write to scratch)
    a = np.stack(df["action"].values)  # (T,7)
    print(f"[stage0] parquet action col: shape={a.shape} dtype={a.dtype}")
    a2 = a + 0.1
    df2 = df.copy()
    df2["action"] = list(a2.astype(np.float32))
    scratch = "/workspace/vla/lingbot_out/distill_relabel/_stage0_scratch/ep0_test.parquet"
    df2.to_parquet(scratch)
    df3 = pd.read_parquet(scratch)
    a3 = np.stack(df3["action"].values)
    rmse = float(np.sqrt(((a3 - a) ** 2).mean()))
    print(f"[stage0] parquet round-trip rmse (expect ~0.1) = {rmse:.4f}")
    os.remove(scratch)
    print("[stage0] FEASIBILITY OK")


if __name__ == "__main__":
    main()
