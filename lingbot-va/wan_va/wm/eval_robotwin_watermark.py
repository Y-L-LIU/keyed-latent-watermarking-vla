"""RoboTwin watermark evaluation for LingBot-VA.

Runs rollouts with watermark injection + inline MAP inversion on watermarked
chunks in the RoboTwin-2.0 simulator. Saves per-episode NPZ with rollout data
AND MAP results for offline re-scoring.

Adapted from eval_libero_watermark.py for the RoboTwin SAPIEN environment.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
        wan_va/wm/eval_robotwin_watermark.py \
        --robotwin-root /path/to/RoboTwin \
        --task-names adjust_bottle stack_bowls_two \
        --test-num 5 --out-dir outputs/wm_robotwin
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ALL_ROBOTWIN_TASKS = [
    "stack_bowls_three", "handover_block", "hanging_mug", "scan_object",
    "lift_pot", "put_object_cabinet", "stack_blocks_three", "place_shoe",
    "adjust_bottle", "place_mouse_pad", "dump_bin_bigbin", "move_pillbottle_pad",
    "pick_dual_bottles", "shake_bottle", "place_fan", "turn_switch",
    "shake_bottle_horizontally", "place_container_plate", "rotate_qrcode",
    "place_object_stand", "put_bottles_dustbin", "move_stapler_pad",
    "place_burger_fries", "place_bread_basket",
    "pick_diverse_bottles", "open_microwave", "beat_block_hammer",
    "press_stapler", "click_bell", "move_playingcard_away", "open_laptop",
    "move_can_pot",
    "stack_bowls_two", "place_a2b_right", "stamp_seal", "place_object_basket",
    "handover_mic", "place_bread_skillet", "stack_blocks_two", "place_cans_plasticbox",
    "click_alarmclock", "blocks_ranking_size", "place_phone_stand",
    "place_can_basket", "place_object_scale", "place_a2b_left", "grab_roller",
    "place_dual_shoes", "place_empty_cup", "blocks_ranking_rgb",
]

DEFAULT_10_TASKS = [
    "adjust_bottle", "stack_bowls_two", "place_a2b_right", "open_laptop",
    "press_stapler", "place_shoe", "handover_block", "click_bell",
    "place_phone_stand", "stack_blocks_two",
]


def _format_obs(observation, prompt):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def _add_eef_pose(new_pose, init_pose):
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_pose(new_pose, init_pose):
    left_pose = _add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = _add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def _euler2quat(r, p, y):
    return R.from_euler('xyz', [r, p, y]).as_quat()


def _convert_action_to_ee(action_step, init_eef_pose):
    """Convert a single model action step to ee action for RoboTwin env.

    action_step: [16] — [left_eef(7), left_gripper(1), right_eef(7), right_gripper(1)]
    init_eef_pose: [16] — initial eef pose from env
    """
    ee_action = action_step.copy()
    if len(ee_action) == 14:
        ee_action = np.concatenate([
            ee_action[:3],
            _euler2quat(ee_action[3], ee_action[4], ee_action[5]),
            ee_action[6:10],
            _euler2quat(ee_action[10], ee_action[11], ee_action[12]),
            ee_action[13:14]
        ])
    elif len(ee_action) == 16:
        ee_action = _add_init_pose(ee_action, init_eef_pose)
        ee_action = np.concatenate([
            ee_action[:3],
            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
            ee_action[7:11],
            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
            ee_action[15:16]
        ])
    else:
        raise NotImplementedError(f"Unsupported action dim: {len(ee_action)}")
    return ee_action


def save_episode_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def run_map_on_chunk(server, raw_actions_tensor, frame_st_id, map_cfg, num_steps=None, map_shift=None,
                     num_starts=1, rng_seed=0):
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPSolver, run_map_restarts
    from wan_va.wm.observation import ChannelObservation

    torch.cuda.empty_cache()

    active_channels = tuple(server.job_config.used_action_channel_ids)
    obs_op = ChannelObservation(channel_idx=active_channels)

    y_obs = obs_op.apply(raw_actions_tensor.to(server.device).float())

    def decode_fn(z):
        return server.sample_actions_from_noise(z, frame_st_id=frame_st_id, num_steps=num_steps, shift_override=map_shift)

    z_shape = raw_actions_tensor.shape
    if int(num_starts) > 1:
        result = run_map_restarts(
            decode_fn, obs_op, y_obs=y_obs, z_shape=z_shape, cfg=map_cfg,
            num_starts=int(num_starts), rng_seed=int(rng_seed))
    else:
        solver = FMLatentMAPSolver(decode_fn, obs_op, map_cfg)
        result = solver.solve(y_obs=y_obs, z_init=None, z_shape=z_shape)
    torch.cuda.empty_cache()
    return result


def _setup_robotwin(robotwin_root: str):
    """Add RoboTwin to sys.path and import necessary modules."""
    robotwin_root = Path(robotwin_root)
    if str(robotwin_root) not in sys.path:
        sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)


def _class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit(f"No Task: {task_name}")
    return env_instance


def _load_task_config(robotwin_root, task_config="demo_clean"):
    """Load RoboTwin task config."""
    import yaml
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(et):
        robot_file = _embodiment_types[et]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])

    return args


def _get_embodiment_config(robot_file):
    import yaml
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_valid_seeds(task_env, task_name, args, num_seeds, start_seed=10000):
    """Run expert check to find valid seeds for evaluation."""
    from envs.utils.create_actor import UnStableError

    valid_seeds = []
    now_seed = start_seed
    now_id = 0
    args["render_freq"] = 0
    args["eval_mode"] = True

    while len(valid_seeds) < num_seeds:
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            continue
        except Exception as e:
            task_env.close_env()
            now_seed += 1
            print(f"  Expert check seed {now_seed-1} error: {e}")
            continue

        if task_env.plan_success and task_env.check_success():
            valid_seeds.append((now_seed, now_id, episode_info))
            now_id += 1
        now_seed += 1

    return valid_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--robotwin-root", type=str, required=True)
    parser.add_argument("--task-names", type=str, nargs="+", default=None,
                        help="Task names to evaluate. Default: 10 representative tasks")
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="outputs/wm_robotwin")
    parser.add_argument("--secret-key", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--chunk-period", type=int, default=6)
    parser.add_argument("--chunk-start-min", type=int, default=2)
    parser.add_argument("--map-iters", type=int, default=30)
    parser.add_argument("--map-steps", type=int, default=10)
    parser.add_argument("--map-shift", type=float, default=None, help="Override scheduler shift for MAP decode_fn (e.g. 0.05)")
    parser.add_argument("--map-lr", type=float, default=0.08)
    parser.add_argument("--map-prior-weight", type=float, default=1.0)
    parser.add_argument("--map-optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--map-num-starts", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-memory", type=str, nargs="+", default=None)
    parser.add_argument("--skip-map", action="store_true")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--skip-expert-check", action="store_true",
                        help="Skip expert seed validation (use sequential seeds)")
    parser.add_argument("--start-seed", type=int, default=10000)
    args = parser.parse_args()

    # Distributed init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    # Setup RoboTwin
    _setup_robotwin(args.robotwin_root)
    from evaluation.robotwin.test_render import Sapien_TEST
    Sapien_TEST()

    from description.utils.generate_episode_instructions import generate_episode_descriptions

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
    if "WAN_FORCE_OFFLOAD" in os.environ:
        config.enable_offload = os.environ["WAN_FORCE_OFFLOAD"] == "1"
    if args.num_gpus > 1:
        config.device_map = "balanced"

    print(f"[Rank {rank}] Loading model (num_gpus={args.num_gpus}, offload={config.enable_offload})...")
    server = VA_Server(config)

    frame_chunk_size = server.job_config.frame_chunk_size
    action_per_frame = server.job_config.action_per_frame
    active_channel_ids = list(server.job_config.used_action_channel_ids)
    num_active_channels = len(active_channel_ids)

    # Keyframe interval: match RoboTwin client behavior (action_per_frame // 4)
    kf_interval = action_per_frame // 4

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
        optimizer=args.map_optimizer,
    )

    # Task list
    task_names = args.task_names if args.task_names else DEFAULT_10_TASKS
    task_config_args = _load_task_config(args.robotwin_root, args.task_config)

    out_dir = Path(args.out_dir)

    print(f"Tasks: {task_names}")
    print(f"Trials per task: {args.test_num}, output: {out_dir}")
    print(f"Watermark: key={args.secret_key}, beta={args.beta}, period={args.chunk_period}, start_min={args.chunk_start_min}")
    print(f"MAP: iters={args.map_iters}, steps={args.map_steps}, shift={args.map_shift}, lr={args.map_lr}, prior_weight={args.map_prior_weight}, opt={args.map_optimizer}, starts={args.map_num_starts}")
    print(f"Model: frame_chunk_size={frame_chunk_size}, action_per_frame={action_per_frame}, kf_interval={kf_interval}")

    summary = {"tasks": [], "total_success": 0, "total_episodes": 0}

    # --- Run evaluations ---
    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"[Task {task_idx}/{len(task_names)}] {task_name}")
        print(f"{'='*60}")

        task_env = _class_decorator(task_name)
        task_args = dict(task_config_args)
        task_args["task_name"] = task_name
        task_args["task_config"] = args.task_config
        task_args["eval_mode"] = True
        task_args["render_freq"] = 0
        task_args["eval_video_log"] = False

        if args.skip_expert_check:
            valid_seeds = [(args.start_seed + i, i, None) for i in range(args.test_num)]
            print(f"  Using {args.test_num} sequential seeds (expert check skipped)")
        else:
            print(f"  Finding {args.test_num} valid seeds...")
            valid_seeds = _find_valid_seeds(
                task_env, task_name, task_args,
                num_seeds=args.test_num,
                start_seed=args.start_seed
            )
            print(f"  Found {len(valid_seeds)} valid seeds")

        task_successes = 0
        task_out_dir = out_dir / task_name

        for episode_idx, (seed, ep_id, episode_info) in enumerate(valid_seeds):
            episode_nonce = task_idx * 10000 + episode_idx

            # Resume: skip if NPZ exists
            npz_path = task_out_dir / f"task{task_idx:02d}_ep{episode_idx:02d}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                task_successes += int(data["success"])
                print(f"  [T{task_idx} E{episode_idx}] SKIP (exists) success={bool(data['success'])}")
                continue

            t0 = time.time()

            # Setup env with the valid seed (retry on unstable seeds)
            from envs.utils.create_actor import UnStableError
            actual_seed = seed
            setup_ok = False
            for retry_offset in range(100):
                try:
                    task_env.suc = 0
                    task_env.test_num = 0
                    task_env.setup_demo(now_ep_num=ep_id, seed=actual_seed, is_test=True, **task_args)
                    setup_ok = True
                    break
                except UnStableError as e:
                    try:
                        task_env.close_env()
                    except Exception:
                        pass
                    actual_seed += 1
                    if retry_offset < 5 or retry_offset % 10 == 0:
                        print(f"    Seed {actual_seed-1} unstable, trying {actual_seed}...")
                    continue
            if not setup_ok:
                print(f"  [T{task_idx} E{episode_idx}] SKIP (no stable seed found)")
                continue

            if episode_info is not None:
                # Generate instruction from expert run info
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(task_name, episode_info_list, args.test_num)
                instruction = np.random.choice(results[0]["seen"])
            else:
                # No expert info: use task description from instruction JSON
                import json as _json
                _instr_path = os.path.join(args.robotwin_root, "description", "task_instruction", f"{task_name}.json")
                with open(_instr_path) as _f:
                    _task_instr = _json.load(_f)
                instruction = _task_instr.get("full_description", task_name.replace("_", " "))
            task_env.set_instruction(instruction=instruction)
            prompt = task_env.get_instruction()

            # Get initial observation and eef pose
            initial_obs = task_env.get_obs()
            init_eef_pose = (
                initial_obs['endpose']['left_endpose']
                + [initial_obs['endpose']['left_gripper']]
                + initial_obs['endpose']['right_endpose']
                + [initial_obs['endpose']['right_gripper']]
            )
            init_eef_pose = np.array(init_eef_pose, dtype=np.float64)

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
            first_obs = _format_obs(initial_obs, prompt)
            prev_raw_actions = None

            while task_env.take_action_cnt < task_env.step_lim and not done:
                wm_context = WatermarkContext(
                    chunk_index=chunk_index,
                    episode_nonce=episode_nonce,
                )
                is_wm_chunk = should_watermark_chunk(wm_config, wm_context)
                current_frame_st_id = server.frame_st_id if not first else 0

                with torch.no_grad():
                    if first:
                        actions_out, _ = server._infer(
                            {'obs': [first_obs]}, frame_st_id=0,
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
                            num_steps=args.map_steps, map_shift=args.map_shift,
                            num_starts=args.map_num_starts,
                            rng_seed=args.secret_key * 1000003 + episode_nonce * 997 + chunk_index)
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

                # Postprocess actions: raw tensor → numpy [num_active_channels, F, H]
                actions_np = server.postprocess_action(raw_actions_t)
                prev_raw_actions = actions_np

                # Execute actions in env (match RoboTwin client behavior)
                key_frame_list = []
                start_f = 1 if first else 0

                for f_idx in range(start_f, frame_chunk_size):
                    for a_idx in range(action_per_frame):
                        if task_env.take_action_cnt >= task_env.step_lim or done:
                            break

                        raw_action_step = actions_np[:, f_idx, a_idx].flatten()
                        ee_action = _convert_action_to_ee(raw_action_step, init_eef_pose)
                        executed_actions_list.append(ee_action.copy())
                        task_env.take_action(ee_action, action_type='ee')
                        global_step += 1

                        # Keyframe capture: every kf_interval steps within a frame
                        if (a_idx + 1) % kf_interval == 0:
                            obs = _format_obs(task_env.get_obs(), prompt)
                            key_frame_list.append(obs)

                    if task_env.take_action_cnt >= task_env.step_lim:
                        break

                # Check success after chunk execution
                if task_env.eval_success:
                    done = True

                first = False
                chunk_index += 1

            task_env.close_env()

            success = bool(done)
            task_successes += int(success)
            executed_actions = np.stack(executed_actions_list, axis=0) if executed_actions_list else np.zeros((0, 16), dtype=np.float32)

            # Save NPZ per episode
            variant = "watermarked" if args.beta > 0 else "plain"
            save_data = dict(
                task_id=np.array(task_idx),
                episode_idx=np.array(episode_idx),
                episode_nonce=np.array(episode_nonce),
                success=np.array(success),
                variant=np.array(variant),
                task_name=np.array(task_name),
                task_description=np.array(prompt),
                total_steps=np.array(global_step),
                num_chunks=np.array(chunk_index),
                secret_key=np.array(args.secret_key),
                beta=np.array(args.beta),
                seed=np.array(seed),
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
        task_rate = task_successes / max(args.test_num, 1)
        summary["tasks"].append({
            "task_name": task_name,
            "task_idx": task_idx,
            "success_rate": task_rate,
            "successes": task_successes,
        })
        summary["total_success"] += task_successes
        summary["total_episodes"] += args.test_num
        print(f"\n  Task {task_name} success rate: {task_rate:.1%} ({task_successes}/{args.test_num})")

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
