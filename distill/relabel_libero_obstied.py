#!/usr/bin/env python3.11
"""Relabel LIBERO HDF5 demos with a teacher policy's obs-tied-watermarked actions.

This builds the *distillation* training corpus for the distillation-robustness
experiment. The teacher is the base pi05_libero policy run with the new
observation-tied DENSE watermark (keyed reference r = r(k*, bucket(eef_pos)),
applied to every chunk, beta=1). We keep every observation byte-identical to the
original demo and overwrite only `data/<demo>/actions` with the teacher's
prediction, open-loop chunked at the policy's action horizon. A student later
behavior-clones this corpus; if the keyed warp is baked into the teacher's
input->action map, the student inherits it (the distillation-survival test).

With --keying clean the teacher injects nothing -> negative control corpus
(a student distilled from a clean teacher must show no key).

Reuses the exact training-time obs construction (_read_libero_image /
_read_libero_state) so the teacher sees in-distribution input.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import time

import h5py
import numpy as np

# openpi imports (PYTHONPATH must include openpi/src + third_party/libero)
from openpi.training.config import get_config
from openpi.training.data_loader import _read_libero_image, _read_libero_state
import openpi.policies.policy_config as policy_config
import openpi.policies.watermark as wm

SUITE_FILES = {
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
}


def build_watermark_config(args) -> wm.InternalNoiseWatermarkConfig | None:
    # "output" keying runs a CLEAN teacher; the obs-tied bias is added post-hoc in
    # action space (see output_bias_fn), so no latent watermark_config is used.
    if args.keying in ("clean", "output", "output_task", "output_task_dc"):
        return None
    proj = tuple(int(x) for x in args.proj_dims.split(",")) if args.proj_dims else None
    return wm.InternalNoiseWatermarkConfig(
        secret_key=args.secret_key,
        control_freq=args.control_freq,
        beta=args.beta,
        reference_mode=args.reference_mode,
        # DENSE: periodic with count>=period marks every chunk.
        chunk_selection_strategy="periodic",
        chunk_selection_period=1,
        chunk_selection_count=1,
        # Observation-tied keying.
        keying_mode="observation",
        obs_key="observation/state",
        obs_proj_dims=proj,
        obs_quantization=args.obs_quantization,
        episode_nonce_key=None,
    )


def _prompt_seed(prompt: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(str(prompt).encode("utf-8"), digest_size=8).digest(), "little")


def make_output_bias_fn(args):
    """Additive ACTION-space keyed bias for the perceptible contrast arms.

    bias(chunk) = beta_out * r_out(k, seed), with r_out a keyed Gaussian reference
    in executed-action space. The seed determines the KEYING FUNCTION the student
    must learn:
      output       -> seed = bucket(ee_pos): a pseudorandom function over ~161 spatial
                      buckets, UNLEARNABLE by a smooth policy (averages out).
      output_task  -> seed = hash(instruction): ~10 discrete offsets, LEARNABLE
                      (per-task action style) -> survives distillation.
    The contrast isolates learnability of the keying as the determinant of survival.
    """
    if args.keying not in ("output", "output_task", "output_task_dc"):
        return None
    proj = tuple(int(x) for x in args.proj_dims.split(",")) if args.proj_dims else None
    ref_cfg = wm.InternalNoiseWatermarkConfig(
        secret_key=args.secret_key, control_freq=args.control_freq,
        beta=1.0, reference_mode="gaussian",
    )

    def bias(state, prompt, horizon, action_dim):
        if args.keying == "output_task_dc":
            import dc_keying
            return dc_keying.dc_bias(args.secret_key, prompt, horizon, action_dim, args.beta_out).astype(np.float32)
        if args.keying == "output_task":
            obs_seed = _prompt_seed(prompt)
        else:
            obs_seed = wm.compute_obs_seed(np.asarray(state), quantization=args.obs_quantization, proj_dims=proj)
        ctx = wm.WatermarkContext(obs_seed=obs_seed)
        r = wm.generate_keyed_reference(
            length=horizon, action_dim=action_dim, sample_rate_hz=args.control_freq,
            config=ref_cfg, context=ctx,
        )
        return float(args.beta_out) * np.asarray(r, dtype=np.float32)

    return bias


def relabel_demo(policy, obs_group, actions_shape, prompt, horizon, output_bias_fn=None) -> np.ndarray:
    """Open-loop relabel: query teacher at chunk starts, write full H-chunks."""
    T = int(actions_shape[0])
    out = np.zeros(actions_shape, dtype=np.float32)
    action_dim = int(actions_shape[1])
    start = 0
    while start < T:
        state = _read_libero_state(obs_group, start)
        observation = {
            "observation/image": _read_libero_image(obs_group, start, ("agentview_image", "agentview_rgb")),
            "observation/wrist_image": _read_libero_image(
                obs_group, start, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")
            ),
            "observation/state": state,
            "prompt": str(prompt),
        }
        result = policy.infer(observation)
        chunk = np.asarray(result["actions"], dtype=np.float32)  # (H, action_dim_env)
        if output_bias_fn is not None:
            chunk = chunk + output_bias_fn(state, prompt, chunk.shape[0], action_dim)[: chunk.shape[0], :action_dim]
        n = min(horizon, T - start, chunk.shape[0])
        out[start : start + n] = chunk[:n, :action_dim]
        start += horizon
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", default="/workspace/vla/models/pi05_libero")
    ap.add_argument("--hdf5-dir", default="/workspace/vla/openpi/third_party/libero/libero/datasets")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keying", choices=["observation", "clean", "output", "output_task", "output_task_dc"], default="observation")
    ap.add_argument("--secret-key", type=int, default=42)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-out", type=float, default=0.1, help="obs-tied output-action bias scale (output mode)")
    ap.add_argument("--reference-mode", default="gaussian")
    ap.add_argument("--control-freq", type=float, default=20.0)
    ap.add_argument("--proj-dims", default="0,1,2")
    ap.add_argument("--obs-quantization", type=float, default=0.08)
    ap.add_argument("--task-range", type=int, nargs=2, default=None, help="[lo hi) task indices for sharding")
    ap.add_argument("--max-demos", type=int, default=None, help="cap demos/task for smoke tests")
    args = ap.parse_args()

    wm_cfg = build_watermark_config(args)
    output_bias_fn = make_output_bias_fn(args)
    mode = "obs-tied LATENT dense" if wm_cfg else ("obs-tied OUTPUT bias" if output_bias_fn else "CLEAN")
    print(f"[relabel] keying={args.keying} mode={mode} "
          f"k={args.secret_key} beta={args.beta} beta_out={args.beta_out} proj={args.proj_dims} q={args.obs_quantization}", flush=True)

    train_config = get_config(args.config_name)
    horizon = int(train_config.model.action_horizon)
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, watermark_config=wm_cfg
    )
    print(f"[relabel] teacher loaded, horizon={horizon}", flush=True)

    # Resolve the suite's demo files via the LIBERO benchmark registry (same order as training).
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    root = pathlib.Path(args.hdf5_dir)
    out_root = pathlib.Path(args.out_dir) / args.suite
    out_root.mkdir(parents=True, exist_ok=True)

    task_ids = list(range(suite.n_tasks))
    if args.task_range:
        lo, hi = args.task_range
        task_ids = [t for t in task_ids if lo <= t < hi]

    for task_id in task_ids:
        rel = suite.get_task_demonstration(task_id)  # e.g. libero_goal/<file>.hdf5
        src = root / rel
        dst = out_root / pathlib.Path(rel).name
        prompt = suite.get_task(task_id).language
        if dst.exists():
            print(f"[relabel] task {task_id} {dst.name} exists, skipping", flush=True)
            continue
        tmp = dst.with_suffix(".tmp.hdf5")
        shutil.copyfile(src, tmp)
        t0 = time.time()
        with h5py.File(tmp, "r+") as f:
            demo_keys = sorted(f["data"].keys())
            if args.max_demos:
                demo_keys = demo_keys[: args.max_demos]
            for di, dk in enumerate(demo_keys):
                demo = f["data"][dk]
                new_actions = relabel_demo(
                    policy, demo["obs"], demo["actions"].shape, prompt, horizon, output_bias_fn=output_bias_fn
                )
                demo["actions"][...] = new_actions
            f.flush()
        tmp.rename(dst)
        print(f"[relabel] task {task_id} {dst.name}: {len(demo_keys)} demos in {time.time()-t0:.1f}s", flush=True)

    print("[relabel] DONE", flush=True)


if __name__ == "__main__":
    main()
