"""LIBERO watermark evaluation with controller postprocessing perturbations.

Tests watermark detection robustness under real-world output-controller
perturbations (clip, smooth, jitter, delay) applied before env execution.

Usage:
    torchrun --nproc_per_node=1 --master_port=29501 \
        wan_va/wm/eval_libero_watermark_robustness.py \
        --suite libero_10 --test-num 5 \
        --out-dir outputs/wm_libero10_robust \
        --controller-postprocess smooth --controller-smooth-alpha 0.7
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan_va.wm.eval_libero_watermark import (
    _extract_obs,
    _suite_max_steps,
    run_map_on_chunk,
    save_episode_npz,
)


# ---------------------------------------------------------------------------
# Controller postprocessor
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RobustnessConfig:
    controller_postprocess: str
    controller_clip_limit: float
    controller_smooth_alpha: float
    controller_jitter_std: float
    controller_delay_steps: int
    seed: int
    run_tag: str | None = None


class ControllerPostprocessor:
    """Per-step action perturbation simulating a downstream controller."""

    def __init__(self, *, config: RobustnessConfig, action_dim: int):
        self.config = config
        self.action_dim = int(action_dim)
        self._episode_nonce: int | None = None
        self._smooth_prev = np.zeros((self.action_dim,), dtype=np.float32)
        self._delay_pending: deque[np.ndarray] = deque()
        self._delay_last_emitted: np.ndarray | None = None

    def reset_episode(self, episode_nonce: int) -> None:
        self._episode_nonce = int(episode_nonce)
        self._smooth_prev = np.zeros((self.action_dim,), dtype=np.float32)
        self._delay_pending = deque()
        self._delay_last_emitted = None

    def apply_step(
        self, action: np.ndarray, *, episode_nonce: int, step: int
    ) -> np.ndarray:
        if self._episode_nonce != int(episode_nonce):
            self.reset_episode(episode_nonce)

        action = np.asarray(action, dtype=np.float32).copy()
        mode = self.config.controller_postprocess

        if mode == "none":
            return action

        if mode == "clip":
            limit = float(self.config.controller_clip_limit)
            return np.clip(action, -limit, limit)

        if mode == "smooth":
            alpha = float(self.config.controller_smooth_alpha)
            self._smooth_prev = alpha * action + (1.0 - alpha) * self._smooth_prev
            return self._smooth_prev.copy()

        if mode == "jitter":
            rng = np.random.default_rng(
                np.random.SeedSequence([
                    int(self.config.seed),
                    int(episode_nonce) & 0xFFFFFFFF,
                    int(step) & 0xFFFFFFFF,
                ])
            )
            noise = rng.normal(0.0, float(self.config.controller_jitter_std), size=action.shape)
            return action + noise.astype(np.float32)

        if mode == "delay":
            delay_steps = max(0, int(self.config.controller_delay_steps))
            self._delay_pending.append(action.copy())
            if len(self._delay_pending) > delay_steps:
                emitted = self._delay_pending.popleft()
                self._delay_last_emitted = emitted
                return emitted
            else:
                if self._delay_last_emitted is None:
                    self._delay_last_emitted = action.copy()
                return self._delay_last_emitted.copy()

        raise ValueError(f"Unsupported controller_postprocess={mode!r}")


def _default_run_tag(config: RobustnessConfig) -> str:
    mode = config.controller_postprocess
    if mode == "none":
        return "controller_none"
    if mode == "clip":
        return f"controller_clip_{config.controller_clip_limit:g}"
    if mode == "smooth":
        return f"controller_smooth_{config.controller_smooth_alpha:g}"
    if mode == "jitter":
        return f"controller_jitter_{config.controller_jitter_std:g}"
    if mode == "delay":
        return f"controller_delay_{config.controller_delay_steps}"
    raise ValueError(f"Unsupported controller_postprocess={mode!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Base eval args
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--suite", type=str, default="libero_10",
                        choices=["libero_goal", "libero_spatial", "libero_object", "libero_10"])
    parser.add_argument("--task-range", type=int, nargs=2, default=None)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="outputs/wm_libero10_robust")
    parser.add_argument("--secret-key", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--chunk-period", type=int, default=6)
    parser.add_argument("--chunk-start-min", type=int, default=2)
    parser.add_argument("--map-iters", type=int, default=30)
    parser.add_argument("--map-steps", type=int, default=10)
    parser.add_argument("--map-lr", type=float, default=0.08)
    parser.add_argument("--map-prior-weight", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-memory", type=str, nargs="+", default=None)
    parser.add_argument("--skip-map", action="store_true")
    # Robustness args
    parser.add_argument("--controller-postprocess", type=str, default="none",
                        choices=["none", "clip", "smooth", "jitter", "delay"])
    parser.add_argument("--controller-clip-limit", type=float, default=1.0)
    parser.add_argument("--controller-smooth-alpha", type=float, default=0.5)
    parser.add_argument("--controller-jitter-std", type=float, default=0.01)
    parser.add_argument("--controller-delay-steps", type=int, default=1)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    args = parser.parse_args()

    rob_config = RobustnessConfig(
        controller_postprocess=args.controller_postprocess,
        controller_clip_limit=args.controller_clip_limit,
        controller_smooth_alpha=args.controller_smooth_alpha,
        controller_jitter_std=args.controller_jitter_std,
        controller_delay_steps=args.controller_delay_steps,
        seed=args.controller_seed,
        run_tag=args.run_tag,
    )
    run_tag = rob_config.run_tag or _default_run_tag(rob_config)

    # Distributed init
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
    config.enable_offload = True
    if args.num_gpus > 1:
        config.device_map = "balanced"

    print(f"[Rank {rank}] Loading model (num_gpus={args.num_gpus}, offload=True)...")
    server = VA_Server(config)

    frame_chunk_size = server.job_config.frame_chunk_size
    action_per_frame = server.job_config.action_per_frame
    active_channel_ids = list(server.job_config.used_action_channel_ids)
    action_dim = len(active_channel_ids)

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

    postprocessor = ControllerPostprocessor(config=rob_config, action_dim=action_dim)

    # --- Setup LIBERO ---
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[args.suite]()
    num_tasks = benchmark_instance.get_num_tasks()

    if args.task_range:
        task_start, task_end = args.task_range
    else:
        task_start, task_end = 0, num_tasks

    max_steps = _suite_max_steps(args.suite)
    out_dir = Path(args.out_dir) / args.suite / run_tag

    print(f"Suite: {args.suite}, tasks: [{task_start}, {task_end}), trials: {args.test_num}")
    print(f"Max steps: {max_steps}, output: {out_dir}")
    print(f"Watermark: key={args.secret_key}, beta={args.beta}, period={args.chunk_period}")
    print(f"MAP: iters={args.map_iters}, steps={args.map_steps}, lr={args.map_lr}")
    print(f"Controller: {args.controller_postprocess} (tag={run_tag})")

    # Save robustness config
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness_config.json").write_text(
        json.dumps(dataclasses.asdict(rob_config), indent=2) + "\n"
    )

    summary = {"suite": args.suite, "run_tag": run_tag,
               "controller_postprocess": args.controller_postprocess,
               "tasks": [], "total_success": 0, "total_episodes": 0}

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

            npz_path = out_dir / f"task{task_idx:02d}_ep{episode_idx:02d}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                task_successes += int(data["success"])
                print(f"  [T{task_idx} E{episode_idx}] SKIP (exists) success={bool(data['success'])}")
                continue

            t0 = time.time()

            env = OffScreenRenderEnv(**env_args)
            env.reset()
            env.set_init_state(init_states[episode_idx % init_states.shape[0]])
            for _ in range(5):
                obs_raw, _, _, _ = env.step([0.0] * 7)

            server._reset(prompt=prompt)
            postprocessor.reset_episode(episode_nonce)

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

                wm_noise = server._last_wm_noise.detach().clone() if hasattr(server, '_last_wm_noise') and server._last_wm_noise is not None else None
                raw_actions_t = server._last_raw_actions.detach().clone()

                chunk_wm_noises.append(wm_noise[0].float().cpu().numpy() if wm_noise is not None else np.zeros((30, frame_chunk_size, action_per_frame, 1), dtype=np.float32))
                chunk_raw_actions.append(raw_actions_t[0].float().cpu().numpy())
                chunk_frame_st_ids.append(current_frame_st_id)
                chunk_watermarked_flags.append(is_wm_chunk)

                # MAP inversion on watermarked chunks (uses CLEAN raw actions)
                if is_wm_chunk and not args.skip_map:
                    try:
                        map_result = run_map_on_chunk(
                            server, raw_actions_t, current_frame_st_id, map_cfg,
                            num_steps=args.map_steps)
                        z_map = map_result["z_map"][0].float().cpu().numpy()
                        mse = map_result["final_obs_mse"]

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

                # Postprocess actions (model output → physical space)
                actions_np = server.postprocess_action(raw_actions_t)
                prev_raw_actions = actions_np

                # Execute with controller perturbation
                key_frame_list = []
                start_f = 1 if first else 0
                kf_interval = action_per_frame // frame_chunk_size

                for f_idx in range(start_f, frame_chunk_size):
                    for a_idx in range(action_per_frame):
                        if global_step >= max_steps or done:
                            break
                        ee_action = actions_np[:, f_idx, a_idx]
                        ee_action = postprocessor.apply_step(
                            ee_action, episode_nonce=episode_nonce, step=global_step)
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
                controller_postprocess=np.array(args.controller_postprocess),
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

        task_rate = task_successes / args.test_num
        summary["tasks"].append({"task_idx": task_idx, "prompt": prompt,
                                  "success_rate": task_rate, "successes": task_successes})
        summary["total_success"] += task_successes
        summary["total_episodes"] += args.test_num
        print(f"\n  Task {task_idx} success rate: {task_rate:.1%} ({task_successes}/{args.test_num})")

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    overall_rate = summary["total_success"] / max(summary["total_episodes"], 1)
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall_rate:.1%} ({summary['total_success']}/{summary['total_episodes']})")
    print(f"Controller: {args.controller_postprocess} (tag={run_tag})")
    print(f"Results: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
