#!/usr/bin/env python3
"""Why does LingBot saturate (z~8-9 sigma) while pi0.5 sits near chance (z~1.5-2)?

Decompose the per-episode matched-filter margin into its two structural factors.
For the raw scorer, the per-episode true-key margin is
    z = (s_true - mean_decoy) / std_decoy
with s_true = sum_d (tv_d - mu_decoy_d) over the D-dim score vector tv. Because
the J=32 decoys are random keyed references, mean_decoy ~ 0 and the decoy SUM has
variance ~ D * sigma_c^2 (sigma_c = per-tap cosine noise floor). Hence

    z  ~=  sqrt(D) * (mu_sig / sigma_c)        [decomposition]

      D       = score-vector length  (matched-filter taps that carry signal)
      mu_sig  = mean per-tap signal  (tv_d - decoy mean): recovery fidelity
      sigma_c = per-tap decoy std    (cosine noise floor ~ 1/sqrt(window len))

This script runs the *actual* production scorers (asc.score_* internals) on a
sample of clean watermarked episodes per family and reports D, mu_sig, sigma_c,
per-tap SNR, the predicted sqrt(D)*SNR, and the measured z -- so we can see which
factor (taps vs fidelity) drives the LingBot >> pi0.5 gap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc          # noqa: E402
from wan_va.wm.scoring import build_score_vector_from_noise  # noqa: E402
from wan_va.wm.watermark import WatermarkContext as LCtx     # noqa: E402
from build_cost_utility_table import _score_episode, _null_episode  # noqa: E402

J = asc.J
DATA = asc.DATA
N_SAMPLE = 18   # watermarked episodes per family (scoring is J=32 ref-gens/chunk -> keep modest)


def lingbot_vectors(npz, preset):
    """Replicate score_lingbot's tv / nm construction, but KEEP the vectors."""
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else (
        "watermarked" if float(d["beta"]) > 0 else "plain")
    if variant != "watermarked":
        return None
    cf = float(preset["F"] * preset["H"])
    nonce = int(d["episode_nonce"]); key = int(d["secret_key"]) if "secret_key" in d.files else 42
    wm_idx = np.where(np.asarray(d["chunk_watermarked_flags"], bool))[0]
    map_z = d["map_z"] if "map_z" in d.files else None
    wm_noises = d["chunk_wm_noises"]
    tp, npr = [], [[] for _ in range(J)]
    for i, ci in enumerate(wm_idx):
        rec = map_z[i] if (map_z is not None and i < len(map_z)) else wm_noises[ci]
        ctx = LCtx(chunk_index=int(ci), episode_nonce=nonce)
        vec = lambda kk: build_score_vector_from_noise(
            np.asarray(rec, np.float32), config=asc._lcfg(kk, cf), context=ctx, sample_rate_hz=cf,
            active_channel_ids=preset["ACTIVE"], frame_chunk_size=preset["F"], action_per_frame=preset["H"])
        tp.append(vec(key))
        for j in range(J):
            npr[j].append(vec(key + 1000 + j))
    if not tp:
        return None
    tv = np.concatenate(tp).astype(np.float64)
    nm = np.stack([np.concatenate(p) for p in npr]).astype(np.float64)
    return tv, nm, len(wm_idx)


def openpi_vectors(npz):
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else "watermarked"
    if variant != "watermarked":
        return None
    key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    tv = _score_episode(d, score_step_scope="full_chunk", max_windows=5)
    if tv.shape[0] == 0:
        return None
    nm = np.stack([_null_episode(d, off + 1, score_step_scope="full_chunk", max_windows=5,
                                 secret_key=key, sample_rate_hz=srate) for off in range(J)])
    return tv.astype(np.float64), nm.astype(np.float64), int(tv.shape[0])


def decompose(tv, nm):
    """Per-episode decomposition quantities."""
    mu_decoy = nm.mean(axis=0)                 # per-tap decoy mean
    sig = tv - mu_decoy                        # per-tap signal
    st = float(sig.sum())                      # = raw_score(tv, nm)
    sf = np.array([float((nm[j] - np.delete(nm, j, axis=0).mean(axis=0)).sum()) for j in range(J)])
    z = (st - sf.mean()) / (sf.std(ddof=1) + asc.av.EPS)
    D = tv.shape[0]
    mu_sig = float(sig.mean())                 # mean per-tap signal
    sigma_c = float(nm.std(axis=0).mean())     # per-tap decoy std (noise floor)
    snr = mu_sig / (sigma_c + 1e-12)
    return dict(D=D, mu_sig=mu_sig, sigma_c=sigma_c, snr=snr, z=z,
                z_pred=np.sqrt(D) * snr)


FAMILIES = [
    ("LingBot/LIBERO-10", "lingbot", "libero",
     ["attack_c_data/rollouts/lingbot_libero/normal"]),
    ("LingBot/RoboTwin", "lingbot", "robotwin",
     ["eval_out/lingbot_rt_fix/clean"]),
    ("pi0.5/LIBERO-10", "openpi", None,
     ["eval_out/base/libero_10/rollouts/none/task_rollout"]),
    ("pi0.5/RoboTwin", "openpi", None,
     ["eval_out/openpi_robotwin_topup"]),
]


def main():
    hdr = f"{'family':20s}{'n':>4s}{'D':>7s}{'n_wm':>6s}{'mu_sig':>9s}{'sigma_c':>9s}{'SNR/tap':>9s}{'z_pred':>8s}{'z_meas':>8s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, pipe, preset, dirs in FAMILIES:
        npzs = []
        for dd in dirs:
            npzs += sorted((DATA / dd).rglob("*.npz"))
        recs = []
        for p in npzs:
            if len(recs) >= N_SAMPLE:
                break
            try:
                if pipe == "lingbot":
                    v = lingbot_vectors(p, asc.PRESETS[preset])
                else:
                    v = openpi_vectors(p)
            except Exception as e:
                print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
            if v is None:
                continue
            tv, nm, n_wm = v
            r = decompose(tv, nm); r["n_wm"] = n_wm
            recs.append(r)
        if not recs:
            print(f"{name:20s}  (no watermarked episodes found)"); continue
        agg = lambda k: float(np.mean([r[k] for r in recs]))
        D = agg("D"); n_wm = agg("n_wm"); mu = agg("mu_sig"); sc = agg("sigma_c")
        snr = agg("snr"); zp = agg("z_pred"); zm = agg("z")
        print(f"{name:20s}{len(recs):4d}{D:7.1f}{n_wm:6.1f}{mu:9.4f}{sc:9.4f}{snr:9.4f}{zp:8.2f}{zm:8.2f}")
        rows.append((name, len(recs), D, n_wm, mu, sc, snr, zp, zm))
    import pandas as pd
    pd.DataFrame(rows, columns=["family", "n", "D", "n_wm", "mu_sig", "sigma_c",
                                "snr_per_tap", "z_pred", "z_meas"]).to_csv(
        HERE / "diag_lingbot_vs_pi05.csv", index=False)
    print(f"\nwrote {HERE/'diag_lingbot_vs_pi05.csv'}")


if __name__ == "__main__":
    main()
