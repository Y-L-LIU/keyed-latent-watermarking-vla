"""PATH B distillation detection eval (LingBot-VA), the pi0.5 score_latentdc_obskey analog.

A distilled STUDENT (LoRA-merged transformer) rolls out PLAIN in LIBERO (no injection).
A second BASE server tracks the student's trajectory in lockstep (same frames + same executed
actions -> identical cache structure, only weights differ). On a spread-out selection of chunks
we MAP-recover, through the BASE, the initial action noise that reproduces the student's observed
chunk actions -- the surviving DC seed if the watermark was inherited. We save per-chunk the
recovered noise + the student's chunk-start proprio state; an offline raw matched-filter scorer
(score_pathb.py) computes the per-episode Z vs same-obs/wrong-key decoys and the cross-student AUC
(watermarked-student vs clean-student), exactly mirroring pi0.5.

Usage (1 GPU, two transformers ~24GB):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa MASTER_ADDR=127.0.0.1 MASTER_PORT=29560 \
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  PYTHONPATH=/workspace/vla/openpi/third_party/libero:/workspace/vla/lingbot-va:/workspace/vla/distill \
  python3.11 -m wan_va.wm.eval_pathb_detection \
      --student-ckpt /workspace/vla_out/student_pathb_n40/checkpoints/checkpoint_step_1500/transformer \
      --out outputs/pathb_det/n40 --suite libero_10 --n-eps 3 --map-chunks 8
"""
from __future__ import annotations

import argparse
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


def _proprio_state(obs):
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:1]
    return np.concatenate([eef, quat, grip])


def _suite_max_steps(suite):
    return {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}.get(suite, 400)


def load_server(ckpt, config_name, rank, local_rank, world_size, save_root):
    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    config = VA_CONFIGS[config_name]
    config.rank = rank; config.local_rank = local_rank; config.world_size = world_size
    config.enable_offload = True
    config.save_root = save_root
    os.makedirs(os.path.join(save_root, "real"), exist_ok=True)
    if ckpt is not None:
        config.wan22_pretrained_model_name_or_path = ckpt
    return VA_Server(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="libero")
    ap.add_argument("--student-ckpt", default=None, help="LoRA-merged transformer dir (None = base, sanity)")
    ap.add_argument("--base-ckpt", default="/workspace/vla/models/lingbot-va-posttrain-libero-long")
    ap.add_argument("--suite", default="libero_10",
                    choices=["libero_goal", "libero_spatial", "libero_object", "libero_10"])
    ap.add_argument("--n-eps", type=int, default=3, help="episodes PER TASK")
    ap.add_argument("--task-range", type=int, nargs=2, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--n-keys", type=int, default=40)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--map-chunks", type=int, default=8, help="how many chunks to MAP per episode (spread across the rollout)")
    ap.add_argument("--map-skip", type=int, default=2, help="skip the first N chunks before MAPping")
    ap.add_argument("--map-stride", type=int, default=3, help="MAP every Nth chunk (spreads obs buckets)")
    ap.add_argument("--map-iters", type=int, default=40)   # Adam needs a few more iters than huge-grad SGD
    ap.add_argument("--map-steps", type=int, default=10)
    ap.add_argument("--map-lr", type=float, default=0.05)   # Adam lr
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    # No-op the server's per-chunk debug dumps (latents/actions/obs .pt) -- never read back, and
    # over a rollout sweep they fill the /workspace quota (two servers -> double the writes).
    import wan_va.wan_va_server as _server_mod
    _server_mod.save_async = lambda *a, **k: None

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from wan_va.wm.watermark import compute_obs_seed
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPConfig
    from wan_va.wm.eval_libero_watermark import run_map_on_chunk

    # Two servers: student (rollout) + base (MAP). Distinct save_roots.
    print("[pathb-det] loading BASE server (MAP)...", flush=True)
    scratch_root = Path(os.environ.get("PATHB_DET_SCRATCH_ROOT", "/tmp/pathb_det_scratch"))
    base = load_server(args.base_ckpt, args.config_name, rank, local_rank, world_size,
                       str(scratch_root / f"base_{os.getpid()}"))
    print(f"[pathb-det] loading STUDENT server (rollout): {args.student_ckpt}", flush=True)
    student = load_server(args.student_ckpt, args.config_name, rank, local_rank, world_size,
                          str(scratch_root / f"student_{os.getpid()}"))

    F = student.job_config.frame_chunk_size
    Hf = student.job_config.action_per_frame
    max_steps = _suite_max_steps(args.suite)
    # Adam (not SGD): robust to the 1/obs_sigma^2 (~1e6) obs weight when inverting an off-manifold
    # distilled student through the base -- SGD diverges to NaN there.
    map_cfg = FMLatentMAPConfig(num_iters=args.map_iters, lr=args.map_lr, obs_sigma=1e-3,
                                prior_weight=1.0, optimizer="adam")

    bench = benchmark.get_benchmark_dict()[args.suite]()
    num_tasks = bench.get_num_tasks()
    t_lo, t_hi = args.task_range if args.task_range else (0, num_tasks)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pathb-det] suite={args.suite} tasks[{t_lo},{t_hi}) eps/task={args.n_eps} "
          f"map_chunks={args.map_chunks} (skip {args.map_skip}, stride {args.map_stride}) "
          f"N_KEYS={args.n_keys}", flush=True)

    def should_map(ci, mapped):
        return ci >= args.map_skip and (ci - args.map_skip) % args.map_stride == 0 and mapped < args.map_chunks

    for task_idx in range(t_lo, t_hi):
        task = bench.get_task(task_idx)
        prompt = task.language
        init_states = bench.get_task_init_states(task_idx)
        env_args = {"bddl_file_name": bench.get_task_bddl_file_path(task_idx),
                    "camera_heights": 128, "camera_widths": 128}

        for ep in range(args.n_eps):
            npz_path = out_dir / f"task{task_idx:02d}_ep{ep:02d}.npz"
            if npz_path.exists():
                print(f"  [T{task_idx} E{ep}] SKIP (exists)"); continue
            episode_nonce = task_idx * 10000 + ep
            t0 = time.time()

            env = OffScreenRenderEnv(**env_args)
            env.reset()
            env.set_init_state(init_states[ep % init_states.shape[0]])
            for _ in range(5):
                obs_raw, _, _, _ = env.step([0.0] * 7)

            student._reset(prompt=prompt)
            base._reset(prompt=prompt)

            rec_noises, rec_states, rec_chunkidx = [], [], []
            done = False; ci = 0; g = 0; first = True; mapped = 0
            obs_dict = _extract_obs(obs_raw)
            prev_actions = None; key_frames = []

            while g < max_steps and not done:
                state = _proprio_state(obs_raw)
                do_map = should_map(ci, mapped)

                # BASE mirrors the STUDENT every chunk (same frames + same executed actions) so its
                # cache + frame_st_id stay in lockstep with the student's trajectory -> a MAP at any
                # later chunk inverts through a base conditioned on the exact observed history.
                with torch.no_grad():
                    if first:
                        student._infer({'obs': [obs_dict]}, frame_st_id=0)
                        base._infer({'obs': [obs_dict]}, frame_st_id=0)
                        cur_st_id = 0
                    else:
                        student._compute_kv_cache({'obs': key_frames, 'state': prev_actions})
                        base._compute_kv_cache({'obs': key_frames, 'state': prev_actions})
                        cur_st_id = base.frame_st_id  # == student.frame_st_id (same frames)
                        student._infer({'obs': [key_frames[-1]]}, frame_st_id=student.frame_st_id)
                        base._infer({'obs': [key_frames[-1]]}, frame_st_id=base.frame_st_id)

                student_raw = student._last_raw_actions.detach().clone()

                # MAP the STUDENT's chunk actions through the BASE (right after base._infer,
                # before the next base._compute_kv_cache -- the cache holds this chunk's preds).
                if do_map:
                    try:
                        mr = run_map_on_chunk(base, student_raw, cur_st_id, map_cfg, num_steps=args.map_steps)
                        z_map = mr["z_map"][0].float().cpu().numpy()
                        rec_noises.append(z_map.astype(np.float32))
                        rec_states.append(state.astype(np.float32))
                        rec_chunkidx.append(ci)
                        mapped += 1
                    except Exception as e:
                        print(f"    MAP fail chunk {ci}: {e}", flush=True)

                actions_np = student.postprocess_action(student_raw)
                prev_actions = actions_np
                key_frames = []
                start_f = 1 if first else 0
                for f_idx in range(start_f, F):
                    for a_idx in range(Hf):
                        if g >= max_steps or done:
                            break
                        ee = actions_np[:, f_idx, a_idx]
                        obs_raw, _, dflag, _ = env.step(ee.tolist())
                        done = bool(dflag); g += 1
                        key_frames.append(_extract_obs(obs_raw))
                    if done or g >= max_steps:
                        break
                first = False; ci += 1
            env.close()

            success = bool(done)
            np.savez_compressed(
                str(npz_path),
                task_id=np.array(task_idx), episode_idx=np.array(ep),
                episode_nonce=np.array(episode_nonce), success=np.array(success),
                n_keys=np.array(args.n_keys), secret_key=np.array(args.secret_key),
                chunk_recovered_noise=np.stack(rec_noises) if rec_noises else np.zeros((0,)),
                chunk_observation_state=np.stack(rec_states) if rec_states else np.zeros((0,)),
                chunk_index=np.array(rec_chunkidx),
                total_chunks=np.array(ci),
            )
            print(f"  [T{task_idx} E{ep}] {'SUCC' if success else 'FAIL'} chunks={ci} "
                  f"mapped={mapped} {time.time()-t0:.1f}s -> {npz_path.name}", flush=True)

    print("[pathb-det] DONE", flush=True)


if __name__ == "__main__":
    main()
