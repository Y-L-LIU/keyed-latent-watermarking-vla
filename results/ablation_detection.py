#!/usr/bin/env python3
"""Detection-pipeline ablation on LingBot/RoboTwin, clean vs. under attack.

Leave-one-component-out over the verifier's scoring/decision blocks, holding the
others fixed, on the same rollouts that feed \\Cref{tab:main}:

  Full   : MAP recovery + whitened matched filter (rank 3) + false-key Z
  -WMF   : replace the PCA-whitened subspace projection with a raw matched filter
           sum_d (feature_d - mu_decoy_d); no whitening, no subspace.
  -Calib : threshold the raw true-key WMF score directly (no per-episode
           normalisation against the 32 decoy keys).

Reported on CLEAN and averaged over post-processing ATTACKS (clip/ema/jitter/
delay). On clean the keyed signal is strong enough that the stripped variants
match or beat Full; the whitening and calibration earn their keep under attack,
where Full holds and the ablated variants degrade. (|G|=1 per-episode AUC is the
lens: |G|=16 saturates to 1.000 and hides the effect.)

The map-recovery chunk vectors are computed ONCE per episode and scored both ways
(whitened + raw), so Full and -WMF share one expensive pass. "Full" reproduces
the committed clean CSV (== tab:main) -- checked by a faithfulness gate.

Writes results/tab_ablation.tex; prints clean, per-attack, and attack-avg rows.
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402  (calibrate/auc/auc_group/tpr_point + consts)

# --- lingbot leaf-module import (same namespace-stub trick as the exporter) --- #
LV = Path("/workspace/vla/lingbot-va/wan_va")
for _name, _path in [("wan_va", str(LV)), ("wan_va.wm", str(LV / "wm"))]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name); _m.__path__ = [_path]; sys.modules[_name] = _m
from wan_va.wm.scoring import build_score_vector_from_noise, wmf_score_from_vectors  # noqa: E402
from wan_va.wm.watermark import InternalNoiseWatermarkConfig, WatermarkContext      # noqa: E402

# robotwin va config (verbatim from export_lingbot_per_episode_scores.py)
ACTIVE = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
F, H = 2, 16
CF = float(F * H)            # 32.0
RANK = 3
J = 32
NULL_OFF = 1000

EVAL = Path("/workspace/vla/eval_out/lingbot_rt_fix")
CLEAN_DIR = EVAL / "clean"
COMMITTED_CSV = Path("/workspace/vla/attack_c_data/per_episode_scores/lingbot_robotwin_partial_map_clean.csv")
# canonical strength per attack family (full-data tags only; jitter_0.01 has 1 npz)
ATTACK_TAGS = ["clip_1.0", "ema_0.5", "jitter_0.02", "delay_2"]
ATTACK_ALL = ["clip_0.5", "clip_1.0", "clip_2.0", "ema_0.3", "ema_0.5",
              "jitter_0.02", "delay_1", "delay_2", "delay_3"]


def _cfg(key):
    return InternalNoiseWatermarkConfig(
        secret_key=key, control_freq=CF, beta=1.0, freq_range=(0.5, 3.0),
        n_tones=4, reference_mode="gaussian", chunk_selection_strategy="stateful_online",
        chunk_selection_period=6, chunk_selection_count=5, chunk_start_min=2)


def _chunk_vec(rec, key, ctx):
    return build_score_vector_from_noise(
        np.asarray(rec, np.float32), config=_cfg(key), context=ctx,
        sample_rate_hz=CF, active_channel_ids=ACTIVE, frame_chunk_size=F, action_per_frame=H)


def _raw_score(feature, null_mat):
    """Un-whitened matched filter: sum of (feature - decoy mean). No PCA, no
    eigenvalue whitening -- isolates exactly the whitened-subspace block."""
    feature = np.asarray(feature, float)
    null_mat = np.asarray(null_mat, float)
    if feature.size == 0:
        return 0.0
    return float(np.sum(feature - null_mat.mean(axis=0)))


def _recover_vectors(d, recovery):
    """Build (true_vec, null_mat) for one episode under the given recovery."""
    nonce = int(d["episode_nonce"])
    true_key = int(d["secret_key"]) if "secret_key" in d.files else 42
    wm_idx = np.where(np.asarray(d["chunk_watermarked_flags"], bool))[0]
    map_z = d["map_z"] if "map_z" in d.files else None
    raw_act = d["chunk_raw_actions"]
    wm_noises = d["chunk_wm_noises"]
    true_parts, null_parts = [], [[] for _ in range(J)]
    for i, ci in enumerate(wm_idx):
        if recovery == "raw_action":
            rec = raw_act[ci]
        else:
            rec = map_z[i] if (map_z is not None and i < len(map_z)) else wm_noises[ci]
        ctx = WatermarkContext(chunk_index=int(ci), episode_nonce=nonce)
        true_parts.append(_chunk_vec(rec, true_key, ctx))
        for j in range(J):
            null_parts[j].append(_chunk_vec(rec, true_key + NULL_OFF + j, ctx))
    if not true_parts:
        return None
    tv = np.concatenate(true_parts).astype(np.float64)
    nm = np.stack([np.concatenate(p) for p in null_parts]).astype(np.float64)
    return tv, nm


def score_episode(npz_path):
    """One map-recovery pass -> both whitened and raw scalar scores."""
    # allow_pickle: trusted local artifacts written by our own rollout pipeline.
    d = np.load(npz_path, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else (
        "watermarked" if float(d["beta"]) > 0 else "plain")
    rv = _recover_vectors(d, "map")
    if rv is None:
        return None
    tv, nm = rv
    st_w = wmf_score_from_vectors(tv, nm, subspace_rank=RANK)
    sf_w = [wmf_score_from_vectors(nm[k], np.delete(nm, k, axis=0), subspace_rank=RANK) for k in range(J)]
    st_r = _raw_score(tv, nm)
    sf_r = [_raw_score(nm[k], np.delete(nm, k, axis=0)) for k in range(J)]
    return dict(variant=variant, st_w=float(st_w), sf_w=[float(x) for x in sf_w],
                st_r=float(st_r), sf_r=[float(x) for x in sf_r])


def build_dir(d):
    """Return (df_wmf, df_raw) for a rollout directory (one episode per npz)."""
    rows_w, rows_r = [], []
    for k, p in enumerate(sorted(Path(d).rglob("*.npz"))):
        r = score_episode(p)
        if r is None:
            continue
        eid = f"lingbot|robotwin10|{p.parent.name}_{p.stem}_{k}_{r['variant']}"
        rows_w.append([eid, r["variant"], r["st_w"]] + r["sf_w"])
        rows_r.append([eid, r["variant"], r["st_r"]] + r["sf_r"])
    cols = ["episode_id", "variant", "s_true"] + [f"s_false_{i+1}" for i in range(J)]
    return pd.DataFrame(rows_w, columns=cols), pd.DataFrame(rows_r, columns=cols)


def metrics(z_h1, z_null):
    rng = np.random.default_rng(av.RNG_SEED)
    rng_auc = np.random.default_rng(av.RNG_SEED + 1)
    a1 = av.auc(z_h1, z_null.ravel())
    a16 = av.auc_group(z_h1, z_null, av.G_TABLE, rng_auc)
    t16 = av.tpr_point(z_h1, z_null, av.G_TABLE, av.FPR_MAIN, rng)
    return a1, a16, t16


def calibrated(df):
    cal = av.calibrate(df)
    return metrics(cal.z_h1, cal.z_null)


def uncalibrated(df):
    """-Calib: raw WMF scores through the identical aggregation (no per-episode Z)."""
    s_false = df[av.FALSE_COLS].to_numpy(float)
    s_true = df["s_true"].to_numpy(float)
    is_wm = df["variant"].to_numpy() == "watermarked"
    return metrics(s_true[is_wm], s_false)


def variants_for_dir(d):
    """Return AUC(|G|=1) for Full / -WMF / -Calib on directory d."""
    df_w, df_r = build_dir(d)
    full = calibrated(df_w)[0]
    nowmf = calibrated(df_r)[0]
    nocal = uncalibrated(df_w)[0]
    return dict(full=full, nowmf=nowmf, nocal=nocal, n=len(df_w))


def main():
    # --- clean (authoritative Full from committed CSV == tab:main) + gate ----- #
    df_csv = pd.read_csv(COMMITTED_CSV)
    full_csv = calibrated(df_csv)[0]
    clean = variants_for_dir(CLEAN_DIR)
    gate = abs(full_csv - clean["full"])
    print(f"[gate] clean Full re-scored {clean['full']:.3f} vs committed {full_csv:.3f} "
          f"|d|={gate:.4f} ({'OK' if gate < 0.02 else 'CHECK'})\n")

    # --- attacks -------------------------------------------------------------- #
    per_attack = {}
    for tag in ATTACK_ALL:
        d = EVAL / "atk" / tag
        if not d.exists():
            continue
        per_attack[tag] = variants_for_dir(d)
        v = per_attack[tag]
        print(f"  {tag:12s} n={v['n']:3d}  Full {v['full']:.3f}  -WMF {v['nowmf']:.3f}  "
              f"-Calib {v['nocal']:.3f}")

    # canonical-strength attack average (one strength per family)
    canon = [per_attack[t] for t in ATTACK_TAGS if t in per_attack]
    avg = {k: float(np.mean([c[k] for c in canon])) for k in ("full", "nowmf", "nocal")}

    print(f"\n  {'variant':24s}{'Clean':>8s}{'Attack-avg':>12s}")
    print(f"  {'Full pipeline':24s}{clean['full']:8.3f}{avg['full']:12.3f}")
    print(f"  {'- WMF whitening':24s}{clean['nowmf']:8.3f}{avg['nowmf']:12.3f}")
    print(f"  {'- false-key calib':24s}{clean['nocal']:8.3f}{avg['nocal']:12.3f}")
    print(f"\n  attack-avg over canonical tags: {ATTACK_TAGS}")

    # --- LaTeX ---------------------------------------------------------------- #
    def row(name, c, a, bold=False):
        f = (lambda x: f"\\textbf{{{x:.3f}}}") if bold else (lambda x: f"{x:.3f}")
        return f"{name} & {f(c)} & {f(a)} \\\\"

    tex = r"""% auto-generated by results/ablation_detection.py -- do not hand-edit
\begin{table}[t]\centering
\caption{Detection ablation on LingBot/RoboTwin: per-episode AUC ($|G|{=}1$) on
clean rollouts and averaged over post-processing attacks (clip, EMA, jitter,
delay at canonical strengths), on the same pool as \Cref{tab:main}. Each row
removes one block of the verifier and holds the others fixed. On clean the keyed
signal is strong enough that the stripped variants match Full; the whitened
matched filter and the false-key calibration earn their keep under attack, where
Full holds and the ablated variants degrade.}
\label{tab:ablation}\small
\begin{tabular}{l cc}
\toprule
Verifier & Clean & Attack-avg \\
\midrule
""" + "\n".join([
        row(r"Full pipeline", clean["full"], avg["full"], bold=True),
        row(r"\quad$-$ whitened matched filter", clean["nowmf"], avg["nowmf"]),
        row(r"\quad$-$ false-key calibration", clean["nocal"], avg["nocal"]),
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (HERE / "tab_ablation.tex").write_text(tex)
    print(f"\n  wrote {HERE/'tab_ablation.tex'}")


if __name__ == "__main__":
    main()
