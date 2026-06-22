"""Post-process saved pi05_libero rollouts to fill the 3 inversion-mode cells
that the original sweep didn't compute (full+MAP / partial+ODE / partial+MAP).

The sweep runs in default mode which produces:
  - chunk_recovered_noise (= full+ODE: reverse-Euler on raw 32-D action)
  - chunk_map_restart_recovered_noise (zeros, MAP not actually run)

This script imports helper functions from eval_libero_action_inversion.py and
reuses them on saved chunk obs + actions. Writes new fields to a sidecar npz
named `<orig>_extra_modes.npz` in the same directory.

Run: requires GPU (loads policy). One process per (suite, GPU) for parallelism.
"""
from __future__ import annotations
import argparse, sys, types, time
from pathlib import Path
import numpy as np

# Ensure openpi src + libero on path (mirror sweep env)
sys.path.insert(0, "/workspace/vla/openpi/src")
sys.path.insert(0, "/workspace/vla/openpi/third_party/libero")
sys.path.insert(0, "/workspace/vla/openpi/scripts")

import argparse as _ap


def _make_args(num_inversion_steps=10, latent_map_iters=100, latent_map_lr=0.1,
               latent_prior_weight=1.0, map_num_starts=4, obs_sigma=1e-4,
               num_decoder_steps=10, inversion_method="reverse", refinement_steps=0,
               refinement_learning_rate=0.05, refinement_latent_l2=1e-4,
               refinement_init_l2=1e-3, full_map_no_warm_start=False,
               latent_init_from_bridge=False, save_recovered_noise_cache_steps=(),
               posterior_step_size=1e-3, posterior_burnin=100, posterior_thinning=50,
               posterior_num_samples=8, posterior_map_tether_weight=1.0,
               posterior_grad_clip_norm=100.0, map_random_seed=0,
               fm_guide_scale=0.5, fm_guide_schedule="linear_decay") -> _ap.Namespace:
    return _ap.Namespace(
        num_inversion_steps=num_inversion_steps,
        num_decoder_steps=num_decoder_steps,
        inversion_method=inversion_method,
        refinement_steps=refinement_steps,
        refinement_learning_rate=refinement_learning_rate,
        refinement_latent_l2=refinement_latent_l2,
        refinement_init_l2=refinement_init_l2,
        latent_map_iters=latent_map_iters,
        latent_map_lr=latent_map_lr,
        latent_prior_weight=latent_prior_weight,
        map_num_starts=map_num_starts,
        map_random_seed=map_random_seed,
        obs_sigma=obs_sigma,
        full_map_no_warm_start=full_map_no_warm_start,
        latent_init_from_bridge=latent_init_from_bridge,
        save_recovered_noise_cache_steps=list(save_recovered_noise_cache_steps),
        posterior_step_size=posterior_step_size,
        posterior_burnin=posterior_burnin,
        posterior_thinning=posterior_thinning,
        posterior_num_samples=posterior_num_samples,
        posterior_map_tether_weight=posterior_map_tether_weight,
        posterior_grad_clip_norm=posterior_grad_clip_norm,
        fm_channel_inverse=False,
        fm_full_latent_map=False,
        fm_latent_map=False,
        fm_latent_posterior=False,
        fm_guide_scale=fm_guide_scale,
        fm_guide_schedule=fm_guide_schedule,
    )


def _reconstruct_obs(d, k):
    """Build the policy obs dict for chunk k from saved chunk_observation_* fields."""
    return {
        "observation/state": np.asarray(d["chunk_observation_state"][k], dtype=np.float32),
        "observation/image": np.asarray(d["chunk_observation_image"][k], dtype=np.uint8),
        "observation/wrist_image": np.asarray(d["chunk_observation_wrist_image"][k], dtype=np.uint8),
        "prompt": str(np.asarray(d["chunk_prompt"][k])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True, type=Path)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--modes", default="full_map,partial_ode,partial_map",
                    help="Comma-separated subset of {full_ode,full_map,partial_ode,partial_map}")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N npz (0 = all)")
    args = ap.parse_args()

    # Lazy import (after path setup) — eval script defines the helpers we need
    print("# importing eval_libero_action_inversion helpers ...", flush=True)
    import eval_libero_action_inversion as eli
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as cfg_mod

    print(f"# loading policy ({args.config_name}, ckpt={args.checkpoint_dir}) ...", flush=True)
    t0 = time.time()
    cfg = cfg_mod.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    print(f"# policy loaded in {time.time()-t0:.1f}s", flush=True)

    modes = set(args.modes.split(","))
    args_inv = _make_args()

    npzs = sorted(args.rollout_dir.rglob("*.npz"))
    if args.limit > 0: npzs = npzs[:args.limit]
    print(f"# processing {len(npzs)} npz", flush=True)

    for ni, p in enumerate(npzs):
        # Resolve symlinks so sidecar lands next to the real npz, not next to the symlink.
        real_p = p.resolve()
        sidecar = real_p.with_name(real_p.stem + "_extra_modes.npz")
        if sidecar.exists():
            print(f"# [{ni+1}/{len(npzs)}] skip (sidecar exists) {p.name}", flush=True)
            continue
        d = np.load(p, allow_pickle=True)
        # Only watermarked rollouts (plain has β=0 base noise; recovery still meaningful but lower-priority)
        chunk_idx_list = list(range(d["chunk_chunk_index"].shape[0]))
        out = {m: [] for m in modes}
        sel_flags = []
        t_ep = time.time()
        for k in chunk_idx_list:
            obs = _reconstruct_obs(d, k)
            raw = np.asarray(d["chunk_raw_actions"][k], dtype=np.float32)
            env_act = np.asarray(d["chunk_observed_actions"][k], dtype=np.float32)
            sel = bool(d["chunk_selected"][k]) if "chunk_selected" in d.files else True
            sel_flags.append(sel)

            # Each MAP variant requires its corresponding fm-* flag set on args.
            # We toggle them per-mode (script asserts the flag at call entry).
            if "full_ode" in modes:
                args_inv.fm_channel_inverse = False; args_inv.fm_full_latent_map = False
                args_inv.fm_latent_map = False; args_inv.fm_latent_posterior = False
                z = eli._recover_noise_from_actions(policy, obs=obs, raw_actions=raw, args=args_inv)
                out["full_ode"].append(np.asarray(z, dtype=np.float32))
            if "partial_ode" in modes:
                args_inv.fm_channel_inverse = False; args_inv.fm_full_latent_map = False
                args_inv.fm_latent_map = False; args_inv.fm_latent_posterior = False
                # zero-pad normalized env-visible 7D to 32D, then reverse-Euler
                y_obs_norm = eli._normalize_channel_observation(policy, env_act)
                a_full = np.zeros_like(raw, dtype=np.float32)
                a_full[..., :y_obs_norm.shape[-1]] = np.asarray(y_obs_norm, dtype=np.float32)
                z = eli._recover_noise_from_actions(policy, obs=obs, raw_actions=a_full, args=args_inv)
                out["partial_ode"].append(np.asarray(z, dtype=np.float32))
            if "full_map" in modes:
                args_inv.fm_channel_inverse = False; args_inv.fm_full_latent_map = True
                args_inv.fm_latent_map = False; args_inv.fm_latent_posterior = False
                payload = eli._recover_noise_from_full_action_latent(
                    policy, obs=obs, raw_action_chunk=raw, args=args_inv)
                out["full_map"].append(np.asarray(payload["recovered_noise"], dtype=np.float32))
            if "partial_map" in modes:
                args_inv.fm_channel_inverse = False; args_inv.fm_full_latent_map = False
                args_inv.fm_latent_map = True; args_inv.fm_latent_posterior = False
                payload = eli._recover_noise_from_channel_observation_latent(
                    policy, obs=obs, env_action_chunk=env_act, raw_action_chunk=raw, args=args_inv)
                out["partial_map"].append(np.asarray(payload["recovered_noise"], dtype=np.float32))

        save = {f"chunk_recovered_noise_{m}": np.stack(out[m]) for m in modes if out[m]}
        save["chunk_selected_subset"] = np.asarray(sel_flags, dtype=bool)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sidecar, **save)
        print(f"# [{ni+1}/{len(npzs)}] wrote {sidecar.name} ({len(chunk_idx_list)} chunks, {time.time()-t_ep:.1f}s)", flush=True)

    print("# done", flush=True)


if __name__ == "__main__":
    main()
