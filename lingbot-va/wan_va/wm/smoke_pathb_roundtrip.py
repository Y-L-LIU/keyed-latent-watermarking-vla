"""PATH B round-trip smoke test (base model only).

The cheap gate before building the expensive generation-based relabel: confirm that a
DC obs-keyed seed injected into LingBot's action initial-noise is MAP-recoverable through
the BASE model and detectable by a RAW matched filter vs same-observation/wrong-key decoys.

This is the LingBot analog of the pi0.5 "teacher is watermarked, Z=4.47" sanity. It mirrors
score_latentdc_obskey.py exactly (raw MF sum(ref*z), Z = (true - decoy.mean)/decoy.std,
seed-retention = r.z / r.r), with the LingBot server + MAP solver in the middle.

NOTHING here is action-space: the watermark enters as the chunk's initial noise (reference_mode
="dc", obs_seed = compute_obs_seed(state) % N_KEYS), exactly the seed site used in the paper.

Usage (1 GPU):
    torchrun --nproc_per_node=1 --master_port=29533 \
        wan_va/wm/smoke_pathb_roundtrip.py \
        --suite libero_10 --task 0 --num-eps 1 --max-chunks 4 --beta 1.0 --n-keys 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/workspace/vla/distill")

import dc_keying  # noqa: E402  (byte-identical to watermark._dc_offset_vector)


def _extract_obs(obs):
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def _proprio_state(obs):
    """8-dim proprio whose first 3 dims are eef xyz (the obs-keying projection)."""
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)      # (3,)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)    # (4,)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:1]
    return np.concatenate([eef, quat, grip])                                   # (8,)


def raw_mf(z_map, state, *, key, n_decoys, n_keys, q, proj, active, length):
    """Raw matched-filter detection of a DC obs-keyed seed in recovered noise.

    Identical math to distill/score_latentdc_obskey.py:
        ref(k) = dc_offset(k, compute_obs_seed(state)%N_KEYS, D) tiled over length, L2-normed
        score(k) = sum(ref(k) * z)
        Z = (score(key) - decoys.mean) / decoys.std        (same obs, wrong key)
        retention = sum(ref(key)*z) / sum(ref(key)*ref(key))
    """
    z = np.nan_to_num(np.asarray(z_map, dtype=np.float64))     # [C, F, H, 1]
    if z.ndim == 4:
        z = z[..., 0]                                          # [C, F, H]
    D = len(active)
    z2 = z[active].reshape(D, length).T                        # [length, D]

    idx = int(_obs_bucket(state, q, proj) % n_keys)            # key-independent obs bucket

    def ref(k):
        c = dc_keying.dc_offset(k, idx, D)                     # (D,)
        r = np.tile(c[None, :], (length, 1))                  # [length, D]
        return r / (np.linalg.norm(r) + 1e-8)

    keys = [key] + [key + 1 + j for j in range(n_decoys)]
    sums = {k: float(np.sum(ref(k) * z2)) for k in keys}   # per-key MF for THIS chunk
    dec = np.array([sums[k] for k in keys[1:]])
    Z = (sums[key] - dec.mean()) / (dec.std() + 1e-8)      # per-chunk Z (display only)
    return sums, Z, float(dec.mean()), float(dec.std()), idx


def _obs_bucket(state, q, proj):
    from wan_va.wm.watermark import compute_obs_seed
    return compute_obs_seed(state, quantization=q, proj_dims=proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="libero")
    ap.add_argument("--suite", default="libero_10",
                    choices=["libero_goal", "libero_spatial", "libero_object", "libero_10"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--num-eps", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--n-keys", type=int, default=40)
    ap.add_argument("--n-decoys", type=int, default=16)
    ap.add_argument("--q", type=float, default=0.08)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--map-iters", type=int, default=30)
    ap.add_argument("--map-steps", type=int, default=10)
    ap.add_argument("--map-lr", type=float, default=0.08)
    args = ap.parse_args()
    proj = tuple(int(x) for x in args.proj_dims.split(","))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    from wan_va.wm.watermark import InternalNoiseWatermarkConfig, WatermarkContext
    from wan_va.wm.fm_latent_map_solver import FMLatentMAPConfig
    from wan_va.wm.eval_libero_watermark import run_map_on_chunk

    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.enable_offload = True
    # absolute scratch save_root (server dumps per-chunk latents/actions debug tensors);
    # relative './train_out' is cwd-fragile under `python -m`.
    config.save_root = "/workspace/vla/lingbot_out/smoke_train_out"
    os.makedirs(os.path.join(config.save_root, "real"), exist_ok=True)
    print(f"[smoke] loading base server ({args.config_name})...", flush=True)
    server = VA_Server(config)

    F = server.job_config.frame_chunk_size
    Hf = server.job_config.action_per_frame
    length = F * Hf
    active = list(server.job_config.used_action_channel_ids)
    D = len(active)
    print(f"[smoke] F={F} H={Hf} length={length} active={active} (D={D}) "
          f"beta={args.beta} N_KEYS={args.n_keys} decoys={args.n_decoys}", flush=True)

    # reference_mode="dc": the byte-identical pi0.5 latent-DC seed; periodic strategy with
    # count>=period -> EVERY chunk watermarked (smoke wants max signal, not selection).
    wm_config = InternalNoiseWatermarkConfig(
        secret_key=args.secret_key,
        control_freq=float(length),
        beta=args.beta,
        reference_mode="dc",
        keying_mode="obs",
        obs_proj_dims=proj,
        obs_quantization=args.q,
        chunk_selection_strategy="periodic",
        chunk_selection_period=1,
        chunk_selection_count=1,
        chunk_start_min=0,
    )
    map_cfg = FMLatentMAPConfig(num_iters=args.map_iters, lr=args.map_lr,
                               obs_sigma=1e-3, prior_weight=1.0)

    benchmark_instance = benchmark.get_benchmark_dict()[args.suite]()
    task = benchmark_instance.get_task(args.task)
    prompt = task.language
    init_states = benchmark_instance.get_task_init_states(args.task)
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(args.task),
        "camera_heights": 128, "camera_widths": 128,
    }
    print(f"[smoke] task {args.task}: {prompt}", flush=True)

    keys = [args.secret_key] + [args.secret_key + 1 + j for j in range(args.n_decoys)]
    ep_Z = []                       # per-episode accumulated Z (the faithful pi0.5 metric)
    for ep in range(args.num_eps):
        episode_nonce = args.task * 10000 + ep
        acc = {k: 0.0 for k in keys}   # accumulate per-key MF across the episode's chunks
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        env.set_init_state(init_states[ep % init_states.shape[0]])
        for _ in range(5):
            obs_raw, _, _, _ = env.step([0.0] * 7)

        server._reset(prompt=prompt)
        obs_dict = _extract_obs(obs_raw)
        first = True
        prev_raw_actions = None
        key_frame_list = []

        for chunk_index in range(args.max_chunks):
            state = _proprio_state(obs_raw)
            bucket = int(_obs_bucket(state, args.q, proj) % args.n_keys)
            wm_context = WatermarkContext(
                chunk_index=chunk_index, episode_nonce=episode_nonce, obs_seed=bucket)
            current_frame_st_id = server.frame_st_id if not first else 0

            t0 = time.time()
            with torch.no_grad():
                if first:
                    server._infer({'obs': [obs_dict]}, frame_st_id=0,
                                  wm_config=wm_config, wm_context=wm_context)
                else:
                    server._compute_kv_cache({'obs': key_frame_list, 'state': prev_raw_actions})
                    current_frame_st_id = server.frame_st_id
                    server._infer({'obs': [key_frame_list[-1]]}, frame_st_id=server.frame_st_id,
                                  wm_config=wm_config, wm_context=wm_context)

            raw_actions_t = server._last_raw_actions.detach().clone()
            wm_noise = server._last_wm_noise.detach().clone()

            # MAP-recover through the SAME (base) model, right after _infer.
            map_result = run_map_on_chunk(server, raw_actions_t, current_frame_st_id,
                                          map_cfg, num_steps=args.map_steps)
            z_map = map_result["z_map"][0].float().cpu().numpy()
            mse = map_result["final_obs_mse"]

            sums, Zc, dmean, dstd, idx = raw_mf(
                z_map, state, key=args.secret_key, n_decoys=args.n_decoys,
                n_keys=args.n_keys, q=args.q, proj=proj, active=active, length=length)
            for k in keys:
                acc[k] += sums[k]               # accumulate per-key MF across chunks

            # sanity: how well did MAP recover the *injected* noise directly (oracle ceiling)?
            inj = wm_noise[0].float().cpu().numpy()
            inj_a = inj[active].reshape(D, length)
            rec_a = z_map[active].reshape(D, length)
            cos = float((inj_a * rec_a).sum() /
                        (np.linalg.norm(inj_a) * np.linalg.norm(rec_a) + 1e-9))
            print(f"  ep{ep} chunk{chunk_index} bucket={idx:3d} MAP_mse={mse:.5f} "
                  f"inj~rec_cos={cos:+.3f} | MF_true={sums[args.secret_key]:+.3f} "
                  f"decoy={dmean:+.3f}±{dstd:.3f} chunkZ={Zc:+.2f} ({time.time()-t0:.1f}s)",
                  flush=True)

            # advance env one chunk so the next chunk sees a different observation/bucket
            actions_np = server.postprocess_action(raw_actions_t)
            prev_raw_actions = actions_np
            key_frame_list = []
            start_f = 1 if first else 0
            kf_interval = max(1, Hf // F)
            done = False
            for f_idx in range(start_f, F):
                for a_idx in range(Hf):
                    ee_action = actions_np[:, f_idx, a_idx]
                    obs_raw, _, done_flag, _ = env.step(ee_action.tolist())
                    done = bool(done_flag)
                    if (a_idx + 1) % kf_interval == 0:
                        key_frame_list.append(_extract_obs(obs_raw))
                if done:
                    break
            first = False
            if done:
                break

        # per-episode accumulated Z (FAITHFUL pi0.5 metric: accumulate MF over chunks, then Z)
        dec_acc = np.array([acc[k] for k in keys[1:]])
        Z_ep = (acc[args.secret_key] - dec_acc.mean()) / (dec_acc.std() + 1e-8)
        ep_Z.append(Z_ep)
        print(f"  ep{ep} ACCUMULATED: sum_true={acc[args.secret_key]:+.2f} "
              f"decoy_acc={dec_acc.mean():+.2f}±{dec_acc.std():.2f}  -> Z_ep={Z_ep:+.2f}",
              flush=True)
        env.close()

    ep_Z = np.array(ep_Z)
    print("\n" + "=" * 64)
    print(f"PATH B round-trip (base teacher, beta={args.beta}): episodes={len(ep_Z)}")
    print(f"  per-episode accumulated Z: mean={ep_Z.mean():+.2f} min={ep_Z.min():+.2f} "
          f"max={ep_Z.max():+.2f}")
    # gate: teacher must be clearly detectable (the 'watermark is present & recoverable'
    # sanity, analogous to pi0.5 teacher Z=4.47). 3 sigma over the episode is decisive.
    gate = ep_Z.mean() > 3.0
    print(f"  GATE (mean accumulated Z>3): {'PASS' if gate else 'FAIL'}")
    print(f"  SMOKE_{'PASS' if gate else 'FAIL'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
