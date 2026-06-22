"""Export per-episode RAW WMF scores for lingbot (diffusion world-action) rollouts,
in the same delivery schema as the openpi exporter.

Reuses lingbot's own validated scoring (`build_score_vector_from_noise`,
`wmf_score_from_vectors`) so `s_true` reproduces the stored `wmf_scores` exactly
(verified 8/8). Emits s_true + J false-key scores (J=32, keys = secret_key+1000+i,
leave-one-out — matching `score_chunk`'s null bank).

robotwin va config: D_raw=30, active channels = [0..6,28,7..13,29] (D_env=16),
obs_ratio=16/30, control_freq = F*H = 32, secret_key=42, subspace_rank=3.
"""
from __future__ import annotations

import argparse, csv, sys, types
from pathlib import Path
import numpy as np

# Register namespace stubs so only the leaf wm modules load (avoid heavy wan_va/__init__).
_BASE = Path(__file__).resolve().parents[1]  # .../wan_va
for _name, _path in [("wan_va", str(_BASE)), ("wan_va.wm", str(_BASE / "wm"))]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name); _m.__path__ = [_path]; sys.modules[_name] = _m

from wan_va.wm.scoring import build_score_vector_from_noise, wmf_score_from_vectors  # noqa: E402
from wan_va.wm.watermark import InternalNoiseWatermarkConfig, WatermarkContext  # noqa: E402

ACTIVE_CHANNELS = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
D_RAW = 30
F, H = 2, 16
CONTROL_FREQ = float(F * H)          # 32.0
SECRET_KEY = 42
SUBSPACE_RANK = 3
NULL_BASE_OFFSET = 1000              # null keys = SECRET_KEY + 1000 + i  (matches score_chunk)


def _cfg(key):
    return InternalNoiseWatermarkConfig(
        secret_key=key, control_freq=CONTROL_FREQ, beta=1.0, freq_range=(0.5, 3.0),
        n_tones=4, reference_mode="gaussian", chunk_selection_strategy="stateful_online",
        chunk_selection_period=6, chunk_selection_count=5, chunk_start_min=2)


def _chunk_vec(recovered_noise, key, ctx):
    return build_score_vector_from_noise(
        np.asarray(recovered_noise, np.float32), config=_cfg(key), context=ctx,
        sample_rate_hz=CONTROL_FREQ, active_channel_ids=ACTIVE_CHANNELS,
        frame_chunk_size=F, action_per_frame=H)


def score_episode(npz_path, J=32):
    """Return (s_true, [s_false_1..J], m, variant, stored_wmf_first)."""
    d = np.load(npz_path, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else (
        "watermarked" if float(d["beta"]) > 0 else "plain")
    nonce = int(d["episode_nonce"])
    # The injected key is per-run (newer runs do not use the legacy 42), so read it
    # from the episode and build the matched-filter reference for the key that was
    # actually used. Fall back to the legacy default for older NPZ without the field.
    true_key = int(d["secret_key"]) if "secret_key" in d.files else SECRET_KEY
    flags = np.asarray(d["chunk_watermarked_flags"], bool)
    wm_idx = np.where(flags)[0]
    wm_noises = d["chunk_wm_noises"]
    have_map = "map_z" in d.files
    map_z = d["map_z"] if have_map else None

    true_parts, null_parts = [], [[] for _ in range(J)]
    for i, ci in enumerate(wm_idx):
        if have_map and i < len(map_z):
            recovered = map_z[i]
        else:
            recovered = wm_noises[ci]      # plain / skip-map: sampler noise is the latent
        ctx = WatermarkContext(chunk_index=int(ci), episode_nonce=nonce)
        true_parts.append(_chunk_vec(recovered, true_key, ctx))
        for j in range(J):
            null_parts[j].append(_chunk_vec(recovered, true_key + NULL_BASE_OFFSET + j, ctx))

    m = len(true_parts)
    if m == 0:
        return None
    true_vec = np.concatenate(true_parts).astype(np.float64)
    null_mat = np.stack([np.concatenate(p) for p in null_parts]).astype(np.float64)
    s_true = wmf_score_from_vectors(true_vec, null_mat, subspace_rank=SUBSPACE_RANK)
    s_false = [wmf_score_from_vectors(null_mat[k], np.delete(null_mat, k, axis=0),
                                      subspace_rank=SUBSPACE_RANK) for k in range(J)]
    stored = float(np.asarray(d["wmf_scores"]).ravel()[0]) if "wmf_scores" in d.files and \
        np.asarray(d["wmf_scores"]).size else float("nan")
    return float(s_true), [float(x) for x in s_false], m, variant, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--attack-strength", default="")
    ap.add_argument("--dataset", default="robotwin10")
    ap.add_argument("--null-count", type=int, default=32)
    # libero override: channels=range(7), F=H=4, control_freq=16 (vs robotwin defaults).
    ap.add_argument("--preset", choices=["robotwin", "libero"], default="robotwin")
    args = ap.parse_args()

    global ACTIVE_CHANNELS, D_RAW, F, H, CONTROL_FREQ
    if args.preset == "libero":
        ACTIVE_CHANNELS = list(range(0, 7))
        D_RAW = 30
        F, H = 4, 4
        CONTROL_FREQ = float(F * H)   # 16.0

    J = args.null_count
    obs_ratio = len(ACTIVE_CHANNELS) / D_RAW
    header = (["episode_id", "variant", "model", "dataset", "obs", "obs_ratio", "recovery",
               "attack", "attack_strength", "m", "s_true"] + [f"s_false_{i+1}" for i in range(J)])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    npzs = sorted(Path(args.rollout_dir).rglob("*.npz"))
    n, max_mismatch = 0, 0.0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for p in npzs:
            r = score_episode(p, J=J)
            if r is None:
                continue
            s_true, s_false, m, variant, stored = r
            if m == 1 and not np.isnan(stored):           # self-check (per-chunk == episode for m=1)
                max_mismatch = max(max_mismatch, abs(s_true - stored))
            eid = f"lingbot|{args.dataset}|{args.attack}|partial|map|{p.parent.name}_{p.stem}_{variant}"
            w.writerow([eid, variant, "lingbot", args.dataset, "partial", f"{obs_ratio:.4f}",
                        "map", args.attack, args.attack_strength, m, f"{s_true:.8f}"]
                       + [f"{x:.8f}" for x in s_false])
            n += 1
    print(f"[lingbot-export] {n} episodes -> {out}  (max |s_true-stored| on m=1: {max_mismatch:.2e})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
