"""LIBERO watermark evaluation for LingBot-VA.

Runs rollouts with watermark injection + inline MAP inversion on watermarked
chunks. Saves per-episode NPZ with rollout data AND MAP results for offline
re-scoring.

Usage (single GPU):
    torchrun --nproc_per_node=1 --master_port=29501 \
        wan_va/wm/eval_libero_watermark.py \
        --suite libero_10 --test-num 5 --out-dir outputs/wm_libero10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _extract_obs(obs):
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def _suite_max_steps(suite: str) -> int:
    table = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    return table.get(suite, 400)


def save_episode_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def run_map_on_chunk(server, raw_actions_tensor, frame_st_id, map_cfg, num_steps=None):
    """Run MAP inversion on a single chunk using the current KV-cache state.

    Must be called RIGHT AFTER _infer() and BEFORE _compute_kv_cache() so the
    cache contains the predictions from that chunk.

    Args:
        server: VA_Server instance with KV-cache in correct state
        raw_actions_tensor: [1, C, F, H, 1] the observed actions (MAP target)
        frame_st_id: the frame_st_id used for that chunk
        map_cfg: FMLatentMAPConfig instance
        num_steps: override denoising steps for MAP (faster with fewer steps)

    Returns:
        dict with z_map, final_obs_mse
    """
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPSolver
    from wan_va.wm.observation import ChannelObservation

    torch.cuda.empty_cache()

    active_channels = tuple(server.job_config.used_action_channel_ids)
    obs_op = ChannelObservation(channel_idx=active_channels)

    y_obs = obs_op.apply(raw_actions_tensor.to(server.device).float())

    def decode_fn(z):
        return server.sample_actions_from_noise(z, frame_st_id=frame_st_id, num_steps=num_steps)

    solver = FMLatentMAPSolver(decode_fn, obs_op, map_cfg)
    z_shape = raw_actions_tensor.shape

    result = solver.solve(y_obs=y_obs, z_init=None, z_shape=z_shape)
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--suite", type=str, default="libero_10",
                        choices=["libero_goal", "libero_spatial", "libero_object", "libero_10"])
    parser.add_argument("--task-range", type=int, nargs=2, default=None)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="outputs/wm_libero10")
    parser.add_argument("--secret-key", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--chunk-period", type=int, default=6)
    parser.add_argument("--chunk-start-min", type=int, default=2)
    parser.add_argument("--map-iters", type=int, default=30)
    parser.add_argument("--map-steps", type=int, default=10)
    parser.add_argument("--map-lr", type=float, default=0.08)
    parser.add_argument("--map-prior-weight", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-memory", type=str, nargs="+", default=None,
                        help="Per-GPU max memory, e.g. '40GiB' '8GiB'")
    parser.add_argument("--skip-map", action="store_true")
    args = parser.parse_args()

    # Distributed init (required by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    from wan_va.wm.watermark import InternalNoiseWatermarkConfig, WatermarkContext, should_watermark_chunk
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPConfig
    from wan_va.wm.scoring import score_chunk

    # --- Load model ---
    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.enable_offload = True  # VAE + text_encoder to CPU
    if args.num_gpus > 1:
        config.device_map = "balanced"  # transformer split across GPUs

    print(f"[Rank {rank}] Loading model (num_gpus={args.num_gpus}, offload=True)...")
    server = VA_Server(config)

    frame_chunk_size = server.job_config.frame_chunk_size
    action_per_frame = server.job_config.action_per_frame
    active_channel_ids = list(server.job_config.used_action_channel_ids)

    wm_config = InternalNoiseWatermarkConfig(
        secret_key=args.secret_key,
        control_freq=float(frame_chunk_size * action_per_frame),
        beta=args.beta,
        chunk_selection_period=args.chunk_period,
        chunk_start_min=args.chunk_start_min,
    )

    map_cfg = FMLatentMAPConfig(
        num_iters=args.map_iters,
        lr=args.map_lr,
        obs_sigma=1e-3,
        prior_weight=args.map_prior_weight,
    )

    # --- Setup LIBERO ---
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[args.suite]()
    num_tasks = benchmark_instance.get_num_tasks()

    if args.task_range:
        task_start, task_end = args.task_range
    else:
        task_start, task_end = 0, num_tasks

    max_steps = _suite_max_steps(args.suite)
    out_dir = Path(args.out_dir) / args.suite

    print(f"Suite: {args.suite}, tasks: [{task_start}, {task_end}), trials: {args.test_num}")
    print(f"Max steps: {max_steps}, output: {out_dir}")
    print(f"Watermark: key={args.secret_key}, beta={args.beta}, period={args.chunk_period}, start_min={args.chunk_start_min}")
    print(f"MAP: iters={args.map_iters}, steps={args.map_steps}, lr={args.map_lr}, prior_weight={args.map_prior_weight}")

    # Summary tracking
    summary = {"suite": args.suite, "tasks": [], "total_success": 0, "total_episodes": 0}

    # --- Run evaluations ---
    for task_idx in range(task_start, task_end):
        task = benchmark_instance.get_task(task_idx)
        prompt = task.language
        init_states = benchmark_instance.get_task_init_states(task_idx)
        env_args = {
            "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
            "camera_heights": 128,
            "camera_widths": 128,
        }

        task_successes = 0
        print(f"\n{'='*60}")
        print(f"[Task {task_idx}/{task_end}] {prompt}")
        print(f"{'='*60}")

        for episode_idx in range(args.test_num):
            episode_nonce = task_idx * 10000 + episode_idx

            # Resume: skip if NPZ already exists
            npz_path = out_dir / f"task{task_idx:02d}_ep{episode_idx:02d}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                task_successes += int(data["success"])
                print(f"  [T{task_idx} E{episode_idx}] SKIP (exists) success={bool(data['success'])}")
                continue

            t0 = time.time()

            # Create env
            env = OffScreenRenderEnv(**env_args)
            env.reset()
            env.set_init_state(init_states[episode_idx % init_states.shape[0]])
            for _ in range(5):
                obs_raw, _, _, _ = env.step([0.0] * 7)

            # Reset model
            server._reset(prompt=prompt)

            executed_actions_list = []
            chunk_wm_noises = []
            chunk_raw_actions = []
            chunk_frame_st_ids = []
            chunk_watermarked_flags = []
            chunk_map_z = []
            chunk_map_mse = []
            chunk_wmf_scores = []

            done = False
            chunk_index = 0
            global_step = 0
            first = True
            obs_dict = _extract_obs(obs_raw)
            prev_raw_actions = None

            while global_step < max_steps and not done:
                wm_context = WatermarkContext(
                    chunk_index=chunk_index,
                    episode_nonce=episode_nonce,
                )
                is_wm_chunk = should_watermark_chunk(wm_config, wm_context)
                current_frame_st_id = server.frame_st_id if not first else 0

                with torch.no_grad():
                    if first:
                        actions_out, _ = server._infer(
                            {'obs': [obs_dict]}, frame_st_id=0,
                            wm_config=wm_config, wm_context=wm_context)
                    else:
                        server._compute_kv_cache({
                            'obs': key_frame_list,
                            'state': prev_raw_actions,
                        })
                        current_frame_st_id = server.frame_st_id
                        actions_out, _ = server._infer(
                            {'obs': [key_frame_list[-1]]},
                            frame_st_id=server.frame_st_id,
                            wm_config=wm_config, wm_context=wm_context)

                # Save rollout traces
                wm_noise = server._last_wm_noise.detach().clone() if hasattr(server, '_last_wm_noise') and server._last_wm_noise is not None else None
                raw_actions_t = server._last_raw_actions.detach().clone()

                chunk_wm_noises.append(wm_noise[0].float().cpu().numpy() if wm_noise is not None else np.zeros((30, frame_chunk_size, action_per_frame, 1), dtype=np.float32))
                chunk_raw_actions.append(raw_actions_t[0].float().cpu().numpy())
                chunk_frame_st_ids.append(current_frame_st_id)
                chunk_watermarked_flags.append(is_wm_chunk)

                # --- MAP inversion on watermarked chunks ---
                if is_wm_chunk and not args.skip_map:
                    try:
                        map_result = run_map_on_chunk(
                            server, raw_actions_t, current_frame_st_id, map_cfg,
                            num_steps=args.map_steps)
                        z_map = map_result["z_map"][0].float().cpu().numpy()
                        mse = map_result["final_obs_mse"]

                        # WMF scoring on recovered noise
                        wmf = score_chunk(
                            z_map,
                            config=wm_config,
                            context=wm_context,
                            sample_rate_hz=wm_config.control_freq,
                            active_channel_ids=active_channel_ids,
                            frame_chunk_size=frame_chunk_size,
                            action_per_frame=action_per_frame,
                            null_count=32,
                            subspace_rank=3,
                        )
                    except Exception as e:
                        print(f"    MAP failed chunk {chunk_index}: {e}")
                        z_map = np.zeros((30, frame_chunk_size, action_per_frame, 1), dtype=np.float32)
                        mse = -1.0
                        wmf = 0.0

                    chunk_map_z.append(z_map)
                    chunk_map_mse.append(mse)
                    chunk_wmf_scores.append(wmf)
                    print(f"    MAP chunk {chunk_index}: MSE={mse:.6f}, WMF={wmf:.3f}")

                # Postprocess actions
                actions_np = server.postprocess_action(raw_actions_t)
                prev_raw_actions = actions_np

                # Execute actions in env (match client.py: key frame every kf_interval steps)
                key_frame_list = []
                start_f = 1 if first else 0
                kf_interval = action_per_frame // frame_chunk_size  # =4//4=1, same as client

                for f_idx in range(start_f, frame_chunk_size):
                    for a_idx in range(action_per_frame):
                        if global_step >= max_steps or done:
                            break
                        ee_action = actions_np[:, f_idx, a_idx]
                        executed_actions_list.append(ee_action.copy())
                        obs_raw, _, done_flag, _ = env.step(ee_action.tolist())
                        done = bool(done_flag)
                        global_step += 1

                        if (a_idx + 1) % kf_interval == 0:
                            obs_dict = _extract_obs(obs_raw)
                            key_frame_list.append(obs_dict)

                    if done or global_step >= max_steps:
                        break

                first = False
                chunk_index += 1

            env.close()

            success = bool(done)
            task_successes += int(success)
            executed_actions = np.stack(executed_actions_list, axis=0) if executed_actions_list else np.zeros((0, 7), dtype=np.float32)

            # Save NPZ per episode (incremental)
            npz_path = out_dir / f"task{task_idx:02d}_ep{episode_idx:02d}.npz"
            variant = "watermarked" if args.beta > 0 else "plain"
            save_data = dict(
                task_id=np.array(task_idx),
                episode_idx=np.array(episode_idx),
                episode_nonce=np.array(episode_nonce),
                success=np.array(success),
                variant=np.array(variant),
                task_description=np.array(prompt),
                total_steps=np.array(global_step),
                num_chunks=np.array(chunk_index),
                secret_key=np.array(args.secret_key),
                beta=np.array(args.beta),
                executed_actions=executed_actions,
                chunk_frame_st_ids=np.array(chunk_frame_st_ids),
                chunk_watermarked_flags=np.array(chunk_watermarked_flags),
                chunk_wm_noises=np.stack(chunk_wm_noises) if chunk_wm_noises else np.zeros((0,)),
                chunk_raw_actions=np.stack(chunk_raw_actions) if chunk_raw_actions else np.zeros((0,)),
            )
            if chunk_map_z:
                save_data["map_z"] = np.stack(chunk_map_z)
                save_data["map_mse"] = np.array(chunk_map_mse)
                save_data["wmf_scores"] = np.array(chunk_wmf_scores)

            save_episode_npz(npz_path, **save_data)

            elapsed = time.time() - t0
            wmf_avg = np.mean(chunk_wmf_scores) if chunk_wmf_scores else 0.0
            print(f"  [T{task_idx} E{episode_idx}] {'SUCC' if success else 'FAIL'} "
                  f"steps={global_step} chunks={chunk_index} wm_chunks={len(chunk_map_z)} "
                  f"avg_wmf={wmf_avg:.3f} {elapsed:.1f}s -> {npz_path.name}")

        # Task summary
        task_rate = task_successes / args.test_num
        summary["tasks"].append({"task_idx": task_idx, "prompt": prompt,
                                  "success_rate": task_rate, "successes": task_successes})
        summary["total_success"] += task_successes
        summary["total_episodes"] += args.test_num
        print(f"\n  Task {task_idx} success rate: {task_rate:.1%} ({task_successes}/{args.test_num})")

        # Save running summary
        summary_path = out_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    overall_rate = summary["total_success"] / max(summary["total_episodes"], 1)
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall_rate:.1%} ({summary['total_success']}/{summary['total_episodes']})")
    print(f"Results: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
