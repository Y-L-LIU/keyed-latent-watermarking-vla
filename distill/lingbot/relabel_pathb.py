"""PATH B (faithful, seed-injection) generation relabel for LingBot-VA.

The pi0.5 analog (relabel_latent_dc_obskey.py), ported to the WAN world-action model.
For each demo, replay it autoregressively through the BASE server on the DEMO's own
observations (obs byte-identical to the clean dataset -- only the action column changes),
injecting a DC obs-keyed seed into each chunk's INITIAL ACTION NOISE
(z_wm = sqrt(1-b^2) z + b * dc_offset(key, obs_bucket % N_KEYS), reference_mode="dc"),
and record the teacher's GENERATED actions as the new BC target. The student then distills
this watermarked behavior; detection MAP-recovers the seed (eval_libero_watermark path).

This is the SEED-space method (consistent with pi0.5), NOT the action-space bias the earlier
dc/hash/hashmod/chunkdc arms used. N_KEYS is the pure entropy knob, identical to pi0.5.

The heavy assets (latents/, videos/, empty_emb.pt, meta/) are symlinked/copied from the clean
libero_long dataset unchanged; only data/*.parquet `action` columns are rewritten -> the student
trains on the SAME demo latents, only the action labels carry the watermark (matched-observation
design, exactly like pi0.5's "obs byte-identical, only actions swapped").

Usage (1 GPU, sharded by episode range for parallelism):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa MASTER_ADDR=127.0.0.1 MASTER_PORT=29540 \
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  PYTHONPATH=/workspace/vla/openpi/third_party/libero:/workspace/vla/lingbot-va:/workspace/vla/distill \
  python3.11 relabel_pathb.py --out /workspace/vla/lingbot_latents/relabel_pathb_n40 \
      --n-keys 40 --beta 1.0 --ep-range 0 50 [--validate]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append("/workspace/vla/lingbot-va")
sys.path.insert(0, "/workspace/vla/distill")

SRC = "/workspace/vla/lingbot_latents/libero_long"
KEY = 42
QUANT = 0.08
PROJ = (0, 1, 2)
CAMS = ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]

# 10 episodes per task x 10 tasks (each task = a contiguous 50-block); this is the
# sparse-but-broad corpus used for the 10-task redo. Output episodes are re-numbered
# to contiguous positions 0..99 because the trainer's episode_data_index is positional.
TASK_STARTS_10X10 = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
NPER_10X10 = 10
ORIG_EPS_10X10 = [t + i for t in TASK_STARTS_10X10 for i in range(NPER_10X10)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def decode_mp4(path):
    """Decode an mp4 to a list of HWC uint8 RGB frames (PyAV, display order)."""
    import av
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def make_obs(ag_frame, eh_frame):
    """Match eval _extract_obs format: dict of HWC uint8 arrays under the cam keys.
    Demo frames come from the stored mp4 (same orientation the latents were extracted from),
    so NO vertical flip (the eval flips only because the LIVE env is upside-down vs the videos)."""
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(ag_frame),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(eh_frame),
    }


def _trim_meta(out, max_eps):
    """Trim the copied meta to the first `max_eps` episodes (contiguous eps 0..max_eps-1).

    The loader's episode_data_index is built from meta.episodes (keyed by episode_index) and the
    hf action view from data/*.parquet, so meta MUST match the generated episode set. Contiguous
    0..N-1 keeps episode_index values and latent filenames valid (no re-indexing needed)."""
    import json as _json
    md = os.path.join(out, "meta")
    kept_lengths = []
    for fn in ["episodes.jsonl", "episodes_stats.jsonl", "episodes_ori.jsonl"]:
        p = os.path.join(md, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        lines = lines[:max_eps]
        if fn == "episodes.jsonl":
            kept_lengths = [int(_json.loads(l).get("length", 0)) for l in lines]
        with open(p, "w") as f:
            f.writelines(lines)
    info_p = os.path.join(md, "info.json")
    with open(info_p) as f:
        info = _json.load(f)
    info["total_episodes"] = max_eps
    info["total_frames"] = int(sum(kept_lengths)) if kept_lengths else info.get("total_frames")
    info["total_videos"] = max_eps * 2   # 2 cameras per episode
    info["splits"] = {"train": f"0:{max_eps}"}
    with open(info_p, "w") as f:
        _json.dump(info, f, indent=4)


def build_reindex_plan():
    ep_by_idx = {
        d["episode_index"]: d
        for d in read_jsonl(os.path.join(SRC, "meta", "episodes.jsonl"))
    }
    plan = []
    global_start = 0
    for pos, orig_ep in enumerate(ORIG_EPS_10X10):
        ep_meta = ep_by_idx[orig_ep]
        length = int(ep_meta["length"])
        plan.append({
            "pos": pos,
            "orig_ep": orig_ep,
            "global_start": global_start,
            "length": length,
            "meta": ep_meta,
        })
        global_start += length
    return plan


def build_reindexed_dataset_skeleton(out):
    """Build the 10x10 sparse corpus skeleton with contiguous output episode ids."""
    if os.path.exists(out):
        print(f"[relabel-pathb] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data", "chunk-000"))
    os.makedirs(os.path.join(out, "meta"))
    os.symlink(os.path.join(SRC, "empty_emb.pt"), os.path.join(out, "empty_emb.pt"))
    for cam in CAMS:
        os.makedirs(os.path.join(out, "latents", "chunk-000", cam))
        os.makedirs(os.path.join(out, "videos", "chunk-000", cam))

    plan = build_reindex_plan()
    stat_by_idx = {
        d["episode_index"]: d
        for d in read_jsonl(os.path.join(SRC, "meta", "episodes_stats.jsonl"))
    }
    shutil.copyfile(os.path.join(SRC, "meta", "tasks.jsonl"), os.path.join(out, "meta", "tasks.jsonl"))

    new_eps, new_stats = [], []
    for item in plan:
        pos, orig_ep = item["pos"], item["orig_ep"]
        ep = json.loads(json.dumps(item["meta"]))
        ep["episode_index"] = pos
        new_eps.append(ep)
        st = json.loads(json.dumps(stat_by_idx[orig_ep]))
        st["episode_index"] = pos
        new_stats.append(st)

        ac = ep["action_config"][0]
        s0, e0 = int(ac["start_frame"]), int(ac["end_frame"])
        for cam in CAMS:
            src_lat = os.path.join(SRC, "latents", "chunk-000", cam, f"episode_{orig_ep:06d}_{s0}_{e0}.pth")
            dst_lat = os.path.join(out, "latents", "chunk-000", cam, f"episode_{pos:06d}_{s0}_{e0}.pth")
            if not os.path.exists(src_lat):
                raise FileNotFoundError(src_lat)
            os.symlink(src_lat, dst_lat)

            src_vid = os.path.join(SRC, "videos", "chunk-000", cam, f"episode_{orig_ep:06d}.mp4")
            if os.path.exists(src_vid):
                os.symlink(src_vid, os.path.join(out, "videos", "chunk-000", cam, f"episode_{pos:06d}.mp4"))

    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        for ep in new_eps:
            f.write(json.dumps(ep) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as f:
        for st in new_stats:
            f.write(json.dumps(st) + "\n")

    info = json.load(open(os.path.join(SRC, "meta", "info.json")))
    info["total_episodes"] = len(plan)
    info["total_frames"] = int(sum(item["length"] for item in plan))
    info["total_videos"] = len(plan) * len(CAMS)
    info["splits"] = {"train": f"0:{len(plan)}"}
    with open(os.path.join(out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    print(f"[relabel-pathb] reindexed 10x10 skeleton -> {out}: "
          f"{len(plan)} eps, total_frames={info['total_frames']}")


def build_dataset_skeleton(out, max_eps=None, reindex=False):
    """Symlink heavy assets + copy meta from the clean dataset (only data/ will be rewritten)."""
    if reindex:
        if max_eps is not None:
            raise ValueError("--max-eps is only valid for the legacy contiguous skeleton")
        build_reindexed_dataset_skeleton(out)
        return
    if os.path.exists(out):
        print(f"[relabel-pathb] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(out)
    for name in ["latents", "videos", "empty_emb.pt"]:
        os.symlink(os.path.join(SRC, name), os.path.join(out, name))
    shutil.copytree(os.path.join(SRC, "meta"), os.path.join(out, "meta"))
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    if max_eps:
        _trim_meta(out, max_eps)
        print(f"[relabel-pathb] trimmed meta to first {max_eps} episodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-keys", type=int, required=True, help="obs-bucket key cardinality (entropy knob); 0 = per-task (DC only)")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--reference-mode", choices=["dc", "gaussian", "bandpass"], default="dc",
                    help="seed reference to inject; dc is the distillation-positive control, gaussian is the zero-mean deployed-style control")
    ap.add_argument("--secret-key", type=int, default=KEY)
    ap.add_argument("--config-name", default="libero")
    ap.add_argument("--ep-range", type=int, nargs=2, default=None, help="[lo, hi) episode indices to process")
    ap.add_argument("--reindex", action="store_true",
                    help="use the 10-task x 10-episode sparse corpus, output as contiguous episodes 0..99")
    ap.add_argument("--pos-range", type=int, nargs=2, default=None,
                    help="[lo, hi) output positions to process with --reindex")
    ap.add_argument("--q", type=float, default=QUANT)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--skeleton-only", action="store_true", help="just build symlink/meta skeleton, no generation")
    ap.add_argument("--no-skeleton", action="store_true", help="assume skeleton exists (sharded runs write episodes into a pre-built dir)")
    ap.add_argument("--max-eps", type=int, default=None, help="trim meta to first N episodes (contiguous 0..N-1) for a small finetune corpus")
    ap.add_argument("--validate", action="store_true", help="MAP-recover the generated chunks and report matched-filter Z")
    args = ap.parse_args()
    if args.reindex and args.ep_range:
        ap.error("--ep-range is ambiguous with --reindex; use --pos-range instead")
    if args.pos_range and not args.reindex:
        ap.error("--pos-range requires --reindex")
    proj = tuple(int(x) for x in args.proj_dims.split(","))

    # skeleton build needs no GPU/server -> handle it before loading the model.
    if args.skeleton_only:
        build_dataset_skeleton(args.out, max_eps=args.max_eps, reindex=args.reindex)
        print("[relabel-pathb] skeleton only, done."); return

    # --- distributed / server ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    from wan_va.wm.watermark import (
        InternalNoiseWatermarkConfig, WatermarkContext, compute_obs_seed)

    # The server dumps per-chunk latents_*.pt / actions_*.pt / obs_data_*.pt via save_async for
    # debug -- never read back. Over the corpus that is tens of GB / tens of thousands of files
    # and BLEW THE /workspace QUOTA (Errno 122). No-op it (nothing depends on these dumps).
    import wan_va.wan_va_server as _server_mod
    _server_mod.save_async = lambda *a, **k: None

    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    # OFFLOAD=0 keeps VAE+text_encoder on GPU (~46G/proc, one proc/GPU) -> no CPU VAE encode,
    # which was the 8-way contention bottleneck (load 270). OFFLOAD=1 for memory-tight solo runs.
    config.enable_offload = os.environ.get("OFFLOAD", "1") == "1"
    config.save_root = "/workspace/vla_out/pathb_scratch"   # off the quota'd /workspace (dumps no-op'd anyway)
    os.makedirs(os.path.join(config.save_root, "real"), exist_ok=True)
    print(f"[relabel-pathb] loading base server ({args.config_name})...", flush=True)
    server = VA_Server(config)

    F = server.job_config.frame_chunk_size
    Hf = server.job_config.action_per_frame
    length = F * Hf
    active = list(server.job_config.used_action_channel_ids)
    D = len(active)

    # reference_mode="dc": pi0.5-faithful positive control. reference_mode="gaussian":
    # zero-mean deployed-style control. obs keying via context.obs_seed (bucket % N_KEYS).
    def make_wm(bucket):
        cfg = InternalNoiseWatermarkConfig(
            secret_key=args.secret_key, control_freq=float(length), beta=args.beta,
            reference_mode=args.reference_mode, keying_mode="obs", obs_proj_dims=proj, obs_quantization=args.q,
            chunk_selection_strategy="periodic", chunk_selection_period=1,
            chunk_selection_count=1, chunk_start_min=0)
        return cfg, WatermarkContext(obs_seed=int(bucket))

    # MAP infra for optional validation (round-trip MF on generated chunks).
    if args.validate:
        sys.path.insert(0, "/workspace/vla/distill")
        import dc_keying
        from wan_va.wm.fm_latent_map_solver import FMLatentMAPConfig
        from wan_va.wm.eval_libero_watermark import run_map_on_chunk
        map_cfg = FMLatentMAPConfig(num_iters=30, lr=0.08, obs_sigma=1e-3, prior_weight=1.0)

        def val_mf(z_map, bucket):
            z = np.nan_to_num(np.asarray(z_map, dtype=np.float64))
            if z.ndim == 4:
                z = z[..., 0]
            z2 = z[active].reshape(D, length).T
            keys = [args.secret_key] + [args.secret_key + 1 + j for j in range(16)]
            def ref(k):
                c = dc_keying.dc_offset(k, int(bucket), D)
                r = np.tile(c[None, :], (length, 1))
                return r / (np.linalg.norm(r) + 1e-8)
            s = {k: float(np.sum(ref(k) * z2)) for k in keys}
            dec = np.array([s[k] for k in keys[1:]])
            return s[args.secret_key], float(dec.mean()), float(dec.std())

    # --- dataset skeleton (sharded runs pass --no-skeleton; skeleton pre-built once) ---
    if not args.no_skeleton:
        build_dataset_skeleton(args.out, reindex=args.reindex)

    tasks = {}
    with open(os.path.join(SRC, "meta", "tasks.jsonl")) as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]

    # episode index from filename episode_NNNNNN.parquet
    def ep_of(pq):
        return int(Path(pq).stem.split("_")[-1])

    if args.reindex:
        units = build_reindex_plan()
        if args.pos_range:
            lo, hi = args.pos_range
            units = [u for u in units if lo <= u["pos"] < hi]
        for u in units:
            u["pq"] = os.path.join(SRC, "data", "chunk-000", f"episode_{u['orig_ep']:06d}.parquet")
    else:
        src_parquets = sorted(glob.glob(os.path.join(SRC, "data", "*", "*.parquet")))
        if args.ep_range:
            lo, hi = args.ep_range
            src_parquets = [p for p in src_parquets if lo <= ep_of(p) < hi]
        units = []
        for pq in src_parquets:
            ep = ep_of(pq)
            units.append({"pq": pq, "orig_ep": ep, "pos": ep, "global_start": None, "length": None})

    print(f"[relabel-pathb] {len(units)} episodes; mode={args.reference_mode} "
          f"N_KEYS={args.n_keys} beta={args.beta} F={F} H={Hf} D={D}", flush=True)

    rmses, all_val = [], []
    for pi, unit in enumerate(units):
        pq = unit["pq"]
        ep = unit["orig_ep"]
        pos = unit["pos"]
        if args.reindex:
            dst = os.path.join(args.out, "data", "chunk-000", f"episode_{pos:06d}.parquet")
        else:
            rel = os.path.relpath(pq, os.path.join(SRC, "data"))
            dst = os.path.join(args.out, "data", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"[relabel-pathb] ep{ep}->pos{pos} exists, skip"); continue

        df = pd.read_parquet(pq)
        if args.reindex:
            df = df.reset_index(drop=True)
            df["episode_index"] = pos
            df["index"] = range(int(unit["global_start"]), int(unit["global_start"]) + len(df))
        a_demo = np.stack(df["action"].values).astype(np.float64)        # (T,7)
        st_demo = np.stack(df["observation.state"].values).astype(np.float64)  # (T,8)
        T = a_demo.shape[0]
        task_idx = int(df["task_index"].iloc[0]); prompt = tasks[task_idx]
        episode_nonce = ep

        cam_ag = f"{SRC}/videos/chunk-000/observation.images.agentview_rgb/episode_{ep:06d}.mp4"
        cam_eh = f"{SRC}/videos/chunk-000/observation.images.eye_in_hand_rgb/episode_{ep:06d}.mp4"
        fa = decode_mp4(cam_ag); fe = decode_mp4(cam_eh)
        n_fr = min(len(fa), len(fe), T)

        server._reset(prompt=prompt)
        executed = []
        g = 0; first = True; chunk_index = 0
        prev_raw_actions = None; key_frame_list = []
        obs_dict = make_obs(fa[0], fe[0])
        t0 = time.time(); val_msgs = []

        while g < n_fr:
            bucket = int(compute_obs_seed(st_demo[g], quantization=args.q, proj_dims=proj)
                         % args.n_keys) if args.n_keys > 0 else \
                     int(compute_obs_seed([task_idx], quantization=1.0, proj_dims=(0,)))
            wm_config, wm_context = make_wm(bucket)
            current_frame_st_id = server.frame_st_id if not first else 0

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

            if args.validate and chunk_index < 4:
                mr = run_map_on_chunk(server, raw_actions_t, current_frame_st_id, map_cfg, num_steps=10)
                z_map = mr["z_map"][0].float().cpu().numpy()
                mft, dm, dsd = val_mf(z_map, bucket)
                val_msgs.append(f"c{chunk_index}(b{bucket}):MF={mft:+.2f}/{dm:+.2f}±{dsd:.2f}")
                all_val.append((mft, dm, dsd))

            actions_np = server.postprocess_action(raw_actions_t)   # [7, F, H]
            prev_raw_actions = actions_np
            key_frame_list = []
            start_f = 1 if first else 0
            for f_idx in range(start_f, F):
                for a_idx in range(Hf):
                    if g >= n_fr:
                        break
                    executed.append(actions_np[:, f_idx, a_idx].copy())
                    g += 1
                    key_frame_list.append(make_obs(fa[min(g, n_fr - 1)], fe[min(g, n_fr - 1)]))
                if g >= n_fr:
                    break
            first = False
            chunk_index += 1

        # assemble relabel action column, length T (pad tail with the demo's own actions)
        ex = np.stack(executed, axis=0).astype(np.float64) if executed else np.zeros((0, D))
        new_a = a_demo.copy()
        m = min(len(ex), T)
        new_a[:m] = ex[:m]
        new_a = new_a.astype(np.float32)
        rmse = float(np.sqrt(((new_a - a_demo) ** 2).mean())); rmses.append(rmse)
        df["action"] = list(new_a)
        df.to_parquet(dst)
        ep_label = f"ep{ep}->pos{pos}" if args.reindex else f"ep{ep}"
        msg = f"[relabel-pathb] {ep_label} ({pi+1}/{len(units)}) chunks={chunk_index} " \
              f"steps={m}/{T} rmse(vs demo)={rmse:.3f} {time.time()-t0:.1f}s"
        if val_msgs:
            msg += " | " + " ".join(val_msgs)
        print(msg, flush=True)

    print(f"[relabel-pathb] DONE -> {args.out}  mean rmse(vs demo)={np.mean(rmses):.3f}", flush=True)
    if all_val:
        v = np.array(all_val)
        print(f"[relabel-pathb] VALIDATE round-trip: MF_true mean={v[:,0].mean():+.2f} "
              f"decoy mean={v[:,1].mean():+.2f} (n={len(v)} chunks) "
              f"-> {'SEED RECOVERABLE' if v[:,0].mean() > v[:,1].mean() + 1.0 else 'WEAK/CHECK'}", flush=True)


if __name__ == "__main__":
    main()
