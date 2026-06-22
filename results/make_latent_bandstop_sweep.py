#!/usr/bin/env python
"""Latent-fingerprint band-stop sweep for paper Sec 5.1 (removability).

Companion to make_fig_bandstop_sweep.py (which sweeps only the OUTPUT-space
sine watermark). Here we compute the REAL detection score of the latent
fingerprint under the same band-stop attack, fully offline from saved
watermarked rollouts.

Per notch center fc:
  1. take the WATERMARKED rollout's per-chunk observed action chunk (T,7),
  2. band-stop it (4th-order Butterworth, width 1.4 Hz) at fc,
  3. re-run the partial-observation MAP latent recovery (the *detector*) on the
     band-stopped action -> recovered internal noise z,
  4. score z against the keyed reference with the SAME whitened matched filter
     (wmf, subspace rank 3, J=32 false keys) as the main pipeline,
  5. normalize each curve to its own no-attack score == 1.0.

This reuses the production recovery+scoring code from
scripts/eval_libero_action_inversion.py (no model reimplementation). The sine
curve is regenerated from make_fig_bandstop_sweep.py so both panels match.

GPU optional. Run from /workspace/vla/openpi so the `scripts`/`openpi` packages
resolve. Set OPENPI_DATA_HOME=/workspace/vla/openpi-cache for the cached tokenizer.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------------------------------------------------------
# Repo wiring. The shared eval module imports `libero` at top level (sim only);
# we never simulate here, so stub it before importing.
# ----------------------------------------------------------------------------
REPO_ROOT = "/workspace/vla/openpi"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.environ.setdefault("OPENPI_DATA_HOME", "/workspace/vla/openpi-cache")

for _name in ("libero", "libero.libero", "libero.libero.envs"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["libero.libero"].benchmark = types.ModuleType("benchmark")
sys.modules["libero.libero"].get_libero_path = lambda *a, **k: "/tmp"
sys.modules["libero.libero.envs"].OffScreenRenderEnv = object

from scripts import eval_libero_action_inversion as ev  # noqa: E402
from scripts import eval_libero_internal_watermark as oe  # noqa: E402

RESULTS_DIR = "/workspace/vla/results"
PI05_WM_DIR = "/workspace/vla/eval_out/base/libero_10/rollouts/none/task_rollout"
CKPT = "/workspace/vla/models/pi05_libero"
CONFIG_NAME = "pi05_libero"

# Sweep geometry — matches make_fig_bandstop_sweep.py exactly.
FS = 20.0
WIDTH = 1.4
CENTERS = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
WM_BAND = (1.0, 2.0)
# Canonical narrow band-stop used in tab:sine-baseline (0.8-2.2 Hz). We report
# the latent score at a notch centered on this band as the headline number.
CANON_BAND = (0.8, 2.2)


def _bandstop(a: np.ndarray, fc: float) -> np.ndarray:
    """4th-order Butterworth band-stop of width WIDTH centered at fc, over (T,7)."""
    lo = max(fc - WIDTH / 2, 0.05)
    hi = min(fc + WIDTH / 2, FS / 2 - 0.05)
    b, aa = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="bandstop")
    return filtfilt(b, aa, a, axis=0).astype(np.float32)


def _bandstop_band(a: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo = max(band[0], 0.05)
    hi = min(band[1], FS / 2 - 0.05)
    b, aa = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="bandstop")
    return filtfilt(b, aa, a, axis=0).astype(np.float32)


def _args_from_npz(z) -> SimpleNamespace:
    """Reconstruct the argparse-like namespace the recovery code reads, from the
    scalars stored in the rollout npz (so config exactly matches what produced
    the saved recovery: MAP 30 iters, lr 0.1, prior 1.0, 2 restarts, etc.)."""
    g = lambda k, d: (z[k].item() if k in z.files else d)  # noqa: E731
    return SimpleNamespace(
        # latent MAP recovery knobs
        fm_latent_map=True,
        fm_latent_posterior=False,
        fm_full_latent_map=False,
        fm_channel_inverse=bool(g("fm_channel_inverse", False)),
        latent_init_from_bridge=bool(g("latent_init_from_bridge", False)),
        obs_sigma=float(g("obs_sigma", 1e-4)),
        latent_map_iters=int(g("latent_map_iters", 30)),
        latent_map_lr=float(g("latent_map_lr", 0.1)),
        latent_prior_weight=float(g("latent_prior_weight", 1.0)),
        map_num_starts=int(g("map_num_starts", 2)),
        map_random_seed=int(g("map_random_seed", 0)),
        # posterior fields (unused for MAP, but read by helpers/defaults)
        posterior_step_size=float(g("posterior_step_size", 0.0)),
        posterior_burnin=int(g("posterior_burnin", 0)),
        posterior_thinning=int(g("posterior_thinning", 1)),
        posterior_num_samples=int(g("posterior_num_samples", 0)),
        posterior_map_tether_weight=float(g("posterior_map_tether_weight", 0.0)),
        posterior_grad_clip_norm=float(g("posterior_grad_clip_norm", 0.0)),
        # watermark / scoring config
        secret_key=int(g("secret_key", 17)),
        sample_rate_hz=float(g("sample_rate_hz", 20.0)),
        beta=float(g("beta", 1.0)),
        freq_min_hz=float(g("freq_min_hz", 1.0)),
        freq_max_hz=float(g("freq_max_hz", 2.0)),
        n_tones=int(g("n_tones", 4)),
        reference_mode=str(g("reference_mode", "gaussian")),
        detector=str(g("detector", "wmf")),
        subspace_rank=(None if int(g("subspace_rank", 3)) < 0 else int(g("subspace_rank", 3))),
        null_decoy_count=int(g("null_decoy_count", 32)),
        score_step_scope=str(g("score_step_scope", "full_chunk")),
        window_aggregator=str(g("window_aggregator", "sum")),
        max_score_windows=(None if int(g("max_score_windows", -1)) < 0 else int(g("max_score_windows", -1))),
        chunk_selection_strategy=str(g("chunk_selection_strategy", "stateful_online")),
        chunk_selection_period=int(g("chunk_selection_period", 1)),
        chunk_selection_count=int(g("chunk_selection_count", 5)),
        chunk_selection_total_slots=(
            None if int(g("chunk_selection_total_slots", -1)) < 0 else int(g("chunk_selection_total_slots", -1))
        ),
    )


def _load_episodes(max_ep: int):
    """Load watermarked pi0.5 rollouts that carry observed actions + references."""
    episodes = []
    for fp in sorted(glob.glob(os.path.join(PI05_WM_DIR, "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        if "variant" not in z.files or str(z["variant"]) != "watermarked":
            continue
        if "chunk_observed_actions" not in z.files or "chunk_reference" not in z.files:
            continue
        episodes.append((fp, z))
        if len(episodes) >= max_ep:
            break
    return episodes


def _obs_dict(z, idx: int, episode_nonce: int, chunk_index: int) -> dict:
    """Reconstruct the policy observation dict for one chunk from saved fields.
    Images are already resized to 224x224 in the npz."""
    return {
        "observation/image": np.asarray(z["chunk_observation_image"][idx], dtype=np.uint8),
        "observation/wrist_image": np.asarray(z["chunk_observation_wrist_image"][idx], dtype=np.uint8),
        "observation/state": np.asarray(z["chunk_observation_state"][idx], dtype=np.float32),
        "prompt": str(z["chunk_prompt"][idx]),
        "chunk_index": int(chunk_index),
        "episode_nonce": int(episode_nonce),
    }


def _build_traces(policy, z, args, episode_nonce, observed_transform):
    """Recover internal noise for every SELECTED chunk after applying
    `observed_transform` to that chunk's observed action, then return a list of
    InversionChunkTrace ready for scoring. The prefix VLM encode happens inside
    recovery; we keep it per-chunk (chunks differ in observation)."""
    selected = np.asarray(z["chunk_selected"], dtype=bool)
    chunk_idx = np.asarray(z["chunk_chunk_index"], dtype=np.int32)
    executed = np.asarray(z["chunk_executed_steps"], dtype=np.int32)
    obs_act = np.asarray(z["chunk_observed_actions"], dtype=np.float32)   # (n,10,7)
    raw_act = np.asarray(z["chunk_raw_actions"], dtype=np.float32)        # (n,10,32)
    reference = np.asarray(z["chunk_reference"], dtype=np.float32)        # (n,10,32)
    inj = np.asarray(z["chunk_injected_noise"], dtype=np.float32)

    traces = []
    for i in range(len(chunk_idx)):
        if not (bool(selected[i]) and int(executed[i]) > 0):
            # non-selected chunks contribute zeros and are skipped by the scorer
            traces.append(
                ev.InversionChunkTrace(
                    chunk_index=int(chunk_idx[i]),
                    executed_steps=int(executed[i]),
                    reference=np.zeros_like(reference[i]),
                    recovered_noise=np.zeros_like(reference[i]),
                    injected_noise=inj[i],
                    raw_actions=raw_act[i],
                    observed_actions=obs_act[i],
                    selected=bool(selected[i]),
                )
            )
            continue
        obs = _obs_dict(z, i, episode_nonce, int(chunk_idx[i]))
        env_chunk = observed_transform(obs_act[i])
        payload = ev._recover_noise_from_channel_observation_latent(
            policy,
            obs=obs,
            env_action_chunk=env_chunk,
            raw_action_chunk=raw_act[i],
            args=args,
        )
        rec = np.asarray(payload["recovered_noise"], dtype=np.float32)
        traces.append(
            ev.InversionChunkTrace(
                chunk_index=int(chunk_idx[i]),
                executed_steps=int(executed[i]),
                reference=reference[i],
                recovered_noise=rec,
                injected_noise=inj[i],
                raw_actions=raw_act[i],
                observed_actions=obs_act[i],
                selected=True,
            )
        )
    return traces


def _episode_true_key_score(traces, args, episode_nonce):
    """Whitened matched-filter score against the TRUE key, matching the main
    pipeline (retarget references to the true key, then wmf with J=32 nulls)."""
    action_dim = next(
        (int(t.reference.shape[1]) for t in traces if t.reference.ndim == 2 and t.reference.shape[1] > 0),
        32,
    )
    ref_cfg = oe.wm.InternalNoiseWatermarkConfig(
        secret_key=int(args.secret_key),
        control_freq=float(args.sample_rate_hz),
        beta=float(args.beta),
        freq_range=(float(args.freq_min_hz), float(args.freq_max_hz)),
        n_tones=int(args.n_tones),
        watermark_dims=tuple(range(action_dim)),
        reference_mode=str(args.reference_mode),
        chunk_selection_strategy=str(args.chunk_selection_strategy),
        chunk_selection_period=int(args.chunk_selection_period),
        chunk_selection_count=int(args.chunk_selection_count),
        chunk_selection_total_slots=args.chunk_selection_total_slots,
    )
    retargeted = ev._retarget_chunk_references(
        list(traces),
        reference_config=ref_cfg,
        sample_rate_hz=float(args.sample_rate_hz),
        episode_nonce=int(episode_nonce),
    )
    null_cfgs = ev._wrong_key_reference_configs(ref_cfg, count=int(args.null_decoy_count))
    score = ev._episode_score(
        retargeted,
        detector=str(args.detector),
        reference_mode=str(args.reference_mode),
        sample_rate_hz=float(args.sample_rate_hz),
        freq_range=(float(args.freq_min_hz), float(args.freq_max_hz)),
        aggregator=str(args.window_aggregator),
        score_step_scope=str(args.score_step_scope),
        max_windows=args.max_score_windows,
        reference_config=ref_cfg,
        episode_nonce=int(episode_nonce),
        null_decoy_count=int(args.null_decoy_count),
        subspace_rank=args.subspace_rank,
        null_reference_configs=null_cfgs,
    )
    return float(score)


def run_latent_sweep(max_ep: int):
    episodes = _load_episodes(max_ep)
    print(f"[latent] loaded {len(episodes)} watermarked pi0.5 episodes")
    if not episodes:
        raise RuntimeError("No watermarked pi0.5 episodes found.")

    args0 = _args_from_npz(episodes[0][1])
    print(f"[latent] recovery cfg: MAP iters={args0.latent_map_iters} lr={args0.latent_map_lr} "
          f"prior={args0.latent_prior_weight} restarts={args0.map_num_starts} obs_sigma={args0.obs_sigma}")
    print(f"[latent] scoring cfg: detector={args0.detector} subspace_rank={args0.subspace_rank} "
          f"J={args0.null_decoy_count} key={args0.secret_key}")

    tc = oe.training_config.get_config(CONFIG_NAME)
    policy = oe.policy_config.create_trained_policy(tc, CKPT)

    # transforms keyed by sweep label
    transforms = {"none": (lambda a: a)}
    for fc in CENTERS:
        transforms[fc] = (lambda a, fc=fc: _bandstop(a, fc))
    transforms["canon"] = (lambda a: _bandstop_band(a, CANON_BAND))

    # per-episode scores by label
    raw = {k: [] for k in transforms}
    for ep_i, (fp, z) in enumerate(episodes):
        args = _args_from_npz(z)
        episode_nonce = int(z["episode_nonce"])
        for label, tf in transforms.items():
            traces = _build_traces(policy, z, args, episode_nonce, tf)
            raw[label].append(_episode_true_key_score(traces, args, episode_nonce))
        print(f"[latent] ep {ep_i+1}/{len(episodes)} {os.path.basename(fp)} "
              f"none={raw['none'][-1]:.4f} canon={raw['canon'][-1]:.4f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in raw.items()}
    base = means["none"]
    norm = {k: (means[k] / base if base != 0 else float("nan")) for k in means}
    return means, norm, base, len(episodes)


# ----------------------------------------------------------------------------
# Sine sweep (regenerate via the existing module so both panels stay identical).
# ----------------------------------------------------------------------------
def _load_sine_module():
    spec = importlib.util.spec_from_file_location(
        "mfbs", os.path.join(RESULTS_DIR, "make_fig_bandstop_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ep", type=int, default=30, help="watermarked episodes to sweep")
    ap.add_argument("--csv", type=str, default=os.path.join(RESULTS_DIR, "latent_bandstop_sweep.csv"))
    args = ap.parse_args()

    means, norm, base, n_ep = run_latent_sweep(args.max_ep)

    print("\n[latent] pi0.5 normalized score vs notch center:")
    for fc in CENTERS:
        print(f"  fc={fc:.1f} Hz : raw={means[fc]:.4f}  norm={norm[fc]:.4f}")
    print(f"  no-attack raw = {base:.4f} (norm 1.000)")
    print(f"  canonical 0.8-2.2 Hz notch: raw={means['canon']:.4f}  norm={norm['canon']:.4f}")

    # write CSV
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "label", "center_hz", "raw_score", "norm_score", "n_episodes"])
        w.writerow(["pi05", "none", "none", f"{base:.6f}", "1.0", n_ep])
        for fc in CENTERS:
            w.writerow(["pi05", "notch", fc, f"{means[fc]:.6f}", f"{norm[fc]:.6f}", n_ep])
        w.writerow(["pi05", "canon_0.8-2.2", "1.5", f"{means['canon']:.6f}", f"{norm['canon']:.6f}", n_ep])
    print(f"[latent] wrote {args.csv}")
    return means, norm, base, n_ep


if __name__ == "__main__":
    main()
