"""End-to-end watermark smoke test for LingBot-VA.

Runs the full pipeline on dummy observations (no LIBERO env needed):
  1. Load model, populate KV-cache with random frames
  2. Run action denoising with watermark injection
  3. Run MAP inversion on the observed actions
  4. Score recovered noise with WMF detector (true key vs null keys)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m wan_va.wm.run_smoke_test --config-name libero
"""
from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--secret-key", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--map-iters", type=int, default=50)
    parser.add_argument("--map-lr", type=float, default=0.1)
    parser.add_argument("--map-steps", type=int, default=20,
                        help="Action denoising steps for MAP (fewer = less memory)")
    parser.add_argument("--obs-sigma", type=float, default=1e-3)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to split model across (1 or 2)")
    args = parser.parse_args()

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    from wan_va.wm.watermark import (
        InternalNoiseWatermarkConfig,
        WatermarkContext,
    )
    from wan_va.wm.observation import ChannelObservation
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPConfig, FMLatentMAPSolver
    from wan_va.wm.scoring import score_chunk

    # --- 1. Load model ---
    config = VA_CONFIGS[args.config_name]
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    # Disable CFG for MAP (halves cache memory)
    config.guidance_scale = 1.0
    config.action_guidance_scale = 1.0

    if args.num_gpus > 1:
        config.device_map = "balanced"

    print("Loading model...")
    server = VA_Server(config)

    frame_chunk_size = server.job_config.frame_chunk_size
    action_per_frame = server.job_config.action_per_frame
    action_dim = server.job_config.action_dim
    active_ids = server.job_config.used_action_channel_ids

    print(f"  frame_chunk_size={frame_chunk_size}, action_per_frame={action_per_frame}")
    print(f"  action_dim={action_dim}, active_channels={active_ids}")

    # --- 2. Initialize session + populate KV-cache with watermarked actions ---
    server._reset(prompt="pick up the red cube and place it on the plate")

    h, w = server.job_config.height, server.job_config.width
    dummy_frame = {
        cam_key: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for cam_key in server.job_config.obs_cam_keys
    }

    # --- 3. Run inference with watermark injection ---
    wm_config = InternalNoiseWatermarkConfig(
        secret_key=args.secret_key,
        control_freq=16.0,
        beta=args.beta,
    )
    wm_context = WatermarkContext(chunk_index=0, episode_nonce=12345)

    print("Running inference with watermark injection...")
    with torch.no_grad():
        server._infer({'obs': [dummy_frame]}, frame_st_id=0,
                      wm_config=wm_config, wm_context=wm_context)

    y_raw = server._last_raw_actions  # [1, C, F, H, 1] bfloat16
    z_wm = server._last_wm_noise      # [1, C, F, H, 1] bfloat16
    print(f"  y_raw shape: {y_raw.shape}, norm: {y_raw.norm().item():.4f}")
    print(f"  z_wm norm: {z_wm.norm().item():.4f}")
    print(f"  beta={args.beta}, key={args.secret_key}")

    # Offload VAE to free memory for MAP
    server.streaming_vae.vae.to('cpu')
    torch.cuda.empty_cache()
    mem_free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU free after offload: {mem_free:.1f} GB")

    # --- 4. MAP Inversion ---
    print(f"\nRunning MAP inversion ({args.map_iters} iters, {args.map_steps} denoise steps)...")

    # Freeze model — only z needs gradients for MAP
    server.transformer.requires_grad_(False)

    # Set MAP denoising steps (fewer than forward for memory)
    server.job_config.action_num_inference_steps = args.map_steps

    obs_op = ChannelObservation(channel_idx=tuple(active_ids))
    y_obs = obs_op.apply(y_raw)  # [1, C_obs, F, H, 1]

    # Free unneeded tensors before MAP
    del y_raw
    torch.cuda.empty_cache()

    map_cfg = FMLatentMAPConfig(
        num_iters=args.map_iters,
        lr=args.map_lr,
        obs_sigma=args.obs_sigma,
        prior_weight=args.prior_weight,
    )

    def decode_fn(z):
        return server.sample_actions_from_noise(z, frame_st_id=0)

    solver = FMLatentMAPSolver(decode_fn, obs_op, map_cfg)

    t0 = time.time()
    result = solver.solve(y_obs=y_obs, z_init=z_wm)
    t_map = time.time() - t0

    z_map = result["z_map"]
    print(f"  MAP done in {t_map:.1f}s")
    print(f"  final obs MSE: {result['final_obs_mse']:.6f}")
    print(f"  z_map norm: {z_map.norm().item():.4f}")

    # Compare z_map with original z_wm
    cosine_sim = torch.nn.functional.cosine_similarity(
        z_map.float().flatten(), z_wm.float().detach().flatten(), dim=0).item()
    l2_dist = (z_map.float() - z_wm.float().detach()).norm().item()
    print(f"  cosine(z_map, z_wm): {cosine_sim:.4f}")
    print(f"  ||z_map - z_wm||: {l2_dist:.4f}")

    # --- 6. WMF Scoring ---
    print(f"\nRunning WMF scoring...")
    z_map_np = z_map[0].float().cpu().numpy()  # [C, F, H, 1]

    true_score = score_chunk(
        z_map_np,
        config=wm_config,
        context=wm_context,
        sample_rate_hz=16.0,
        active_channel_ids=active_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
        null_count=32,
        subspace_rank=3,
    )

    # Also score the original watermarked noise (oracle, for comparison)
    z_wm_np = z_wm[0].float().cpu().numpy()
    oracle_score = score_chunk(
        z_wm_np,
        config=wm_config,
        context=wm_context,
        sample_rate_hz=16.0,
        active_channel_ids=active_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
        null_count=32,
        subspace_rank=3,
    )

    # Score with wrong key (should be near 0)
    wrong_config = InternalNoiseWatermarkConfig(
        secret_key=args.secret_key + 999,
        control_freq=16.0,
        beta=args.beta,
    )
    wrong_key_score = score_chunk(
        z_map_np,
        config=wrong_config,
        context=wm_context,
        sample_rate_hz=16.0,
        active_channel_ids=active_ids,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
        null_count=32,
        subspace_rank=3,
    )

    print(f"\n{'='*50}")
    print(f"  WMF score (true key, MAP z):    {true_score:.4f}")
    print(f"  WMF score (true key, oracle z): {oracle_score:.4f}")
    print(f"  WMF score (wrong key, MAP z):   {wrong_key_score:.4f}")
    print(f"{'='*50}")

    if true_score > wrong_key_score + 1.0:
        print("\n=== WATERMARK DETECTION: PASS ===")
        print(f"  True key score ({true_score:.2f}) >> wrong key ({wrong_key_score:.2f})")
    elif true_score > wrong_key_score:
        print("\n=== WATERMARK DETECTION: MARGINAL ===")
        print(f"  True key score ({true_score:.2f}) > wrong key ({wrong_key_score:.2f}), but gap is small")
    else:
        print("\n=== WATERMARK DETECTION: FAIL ===")
        print(f"  True key score ({true_score:.2f}) <= wrong key ({wrong_key_score:.2f})")


if __name__ == "__main__":
    main()
