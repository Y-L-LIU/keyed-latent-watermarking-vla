"""Gradient flow test for sample_actions_from_noise.

Loads the actual model, runs a single chunk through the action denoising loop,
and verifies that gradients flow back to the initial noise z_init.

Usage:
    python -m wan_va.wm.test_gradient_flow --config-name libero
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS

    # Build server (loads model) — skip init_distributed for single-GPU test
    config = VA_CONFIGS[args.config_name]
    if args.port is not None:
        config.port = args.port
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    server = VA_Server(config)

    frame_chunk_size = server.job_config.frame_chunk_size
    action_per_frame = server.job_config.action_per_frame
    action_dim = server.job_config.action_dim

    print("Model loaded. Testing gradient flow...")
    print("  frame_chunk_size={}, action_per_frame={}, action_dim={}".format(
        frame_chunk_size, action_per_frame, action_dim))

    # Initialize session: set up cache, embeddings, action_mask
    # For MAP gradient test, disable video CFG to halve cache memory (batch_size=1)
    server.job_config.guidance_scale = 1.0
    server.job_config.action_guidance_scale = 1.0
    server._reset(prompt="pick up the red cup")
    print("Session reset (no CFG, batch_size=1 cache). Cache created.")

    # Run a dummy first inference to populate KV-cache with observation context.
    # obs format: {'obs': [list of dicts with cam_key -> HWC uint8 image]}
    h, w = server.job_config.height, server.job_config.width
    dummy_frame = {
        cam_key: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for cam_key in server.job_config.obs_cam_keys
    }
    dummy_obs = {'obs': [dummy_frame]}
    print("Running first inference to populate KV-cache...")
    with torch.no_grad():
        server._infer(dummy_obs, frame_st_id=0)
    print("KV-cache populated.")

    # Free memory: offload VAE to CPU (not in action denoising graph), clear CUDA cache
    server.streaming_vae.vae.to('cpu')
    torch.cuda.empty_cache()
    mem_free = torch.cuda.mem_get_info()[0] / 1e9
    print("After offload: {:.1f} GB free".format(mem_free))

    # Now test gradient flow through sample_actions_from_noise
    # Use fewer steps for MAP (full 50 is for inference quality, MAP works fine with 20)
    map_steps = 20
    server.job_config.action_num_inference_steps = map_steps
    z_init = torch.randn(
        1, action_dim, frame_chunk_size, action_per_frame, 1,
        device=server.device, dtype=server.dtype, requires_grad=True)

    print("\nRunning sample_actions_from_noise ({} steps)...".format(
        server.job_config.action_num_inference_steps))

    try:
        actions = server.sample_actions_from_noise(z_init, frame_st_id=0)
        print("Output shape: {}".format(actions.shape))

        loss = actions.sum()
        loss.backward()

        if z_init.grad is not None:
            grad_norm = z_init.grad.norm().item()
            print("z_init.grad norm: {:.6f}".format(grad_norm))
            if grad_norm > 0:
                print("\n=== GRADIENT FLOW TEST: PASS ===")
            else:
                print("\n=== GRADIENT FLOW TEST: FAIL (zero grad) ===")
        else:
            print("\n=== GRADIENT FLOW TEST: FAIL (grad is None) ===")

    except Exception as e:
        print("\nError during test: {}".format(e))
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
