#!/usr/bin/env python3
"""Raw recovery fidelity = cosine(recovered_seed, injected_seed), SAME method for
all four families, measured BEFORE the matched filter / keyed scorer.

This is the single quantity the "why LingBot >> pi0.5" story should rest on.
Unlike diag_lingbot_vs_pi05.py (which reports post-filter per-tap mu_sig at
different tap granularities per pipeline), here we use one definition everywhere:

    cos_chunk = <z_rec_flat, z_inj_flat> / (||z_rec_flat|| * ||z_inj_flat||)

flattened over the FULL seed of a chunk, averaged over watermarked chunks, then
over episodes. For pi0.5 we also split the 32-dim latent into the env-actuated
dims (0:A) vs the unactuated padding (A:32) to test the "under-observed seed"
claim directly.  A = 7 (LIBERO 7-DoF) / 14 (RoboTwin dual-arm 7+7).
"""
from __future__ import annotations
import glob
import numpy as np

N_SAMPLE = 18

# LingBot keyed-channel presets (must match results/ablation_scorer_allcells.py).
LB_PRESETS = {
    "libero":   dict(ACTIVE=list(range(0, 7)), F=4, H=4),
    "robotwin": dict(ACTIVE=list(range(0, 7)) + [28] + list(range(7, 14)) + [29], F=2, H=16),
}


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _per_row_cos(A, B):
    """Mean cosine over rows: A,B are (n_rows, length); cosine within each row."""
    return float(np.mean([_cos(A[r], B[r]) for r in range(A.shape[0])]))


def lingbot_cos(npz, preset):
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else (
        "watermarked" if float(d.get("beta", 0)) > 0 else "plain")
    if variant != "watermarked" or "map_z" not in d.files:
        return None
    flags = np.asarray(d["chunk_watermarked_flags"], bool)
    wm_idx = np.where(flags)[0]
    inj = np.asarray(d["chunk_wm_noises"])     # (n_chunks, ...) injected keyed seed
    rec = np.asarray(d["map_z"])               # (n_wm, ...)    MAP-recovered seed
    n = min(len(wm_idx), len(rec))
    if n == 0:
        return None
    ACTIVE = preset["ACTIVE"]; length0 = preset["F"] * preset["H"]; ad = len(ACTIVE)
    full, perch = [], []
    for i in range(n):
        zr = np.asarray(rec[i], np.float32)
        zi = np.asarray(inj[wm_idx[i]], np.float32)
        while zr.ndim > 3:
            zr = zr[..., 0]; zi = zi[..., 0]
        ar = zr[ACTIVE].reshape(ad, length0)      # keyed channels only
        ai = zi[ACTIVE].reshape(ad, length0)
        full.append(_cos(ar, ai))                 # flattened over keyed channels
        perch.append(_per_row_cos(ar, ai))        # per-keyed-channel, then mean
    return {"keyed_flat": float(np.mean(full)), "keyed_perdim": float(np.mean(perch)),
            "n_wm": n}


def openpi_cos(npz, n_act):
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else "watermarked"
    if variant != "watermarked" or "chunk_injected_noise" not in d.files:
        return None
    inj = np.asarray(d["chunk_injected_noise"])      # (n_chunks, H, 32)
    rec = np.asarray(d["chunk_recovered_noise"])     # (n_chunks, H, 32)
    sel = np.asarray(d.get("chunk_selected", np.ones(inj.shape[0], bool)), bool)
    inj, rec = inj[sel], rec[sel]
    n = inj.shape[0]
    if n == 0:
        return None
    # flattened cosine over dim-subsets (matches the doc's "all 32 / observed / padding")
    allf = [_cos(rec[k], inj[k]) for k in range(n)]
    actf = [_cos(rec[k, :, :n_act], inj[k, :, :n_act]) for k in range(n)]
    padf = [_cos(rec[k, :, n_act:], inj[k, :, n_act:]) for k in range(n)]
    # per-keyed-dim cosine over the H time-steps (parallel to LingBot per-channel)
    perdim = [_per_row_cos(rec[k].T, inj[k].T) for k in range(n)]          # all 32 dims
    perdim_act = [_per_row_cos(rec[k, :, :n_act].T, inj[k, :, :n_act].T) for k in range(n)]
    return {"all_flat": float(np.mean(allf)), "act_flat": float(np.mean(actf)),
            "pad_flat": float(np.mean(padf)), "keyed_perdim": float(np.mean(perdim)),
            "act_perdim": float(np.mean(perdim_act)), "n_wm": n}


FAMILIES = [
    ("LingBot/LIBERO-10", "lingbot", "libero",
     "attack_c_data/rollouts/lingbot_libero/normal"),
    ("LingBot/RoboTwin", "lingbot", "robotwin",
     "eval_out/lingbot_rt_fix/clean"),
    ("pi0.5/LIBERO-10", "openpi", 7,
     "eval_out/base/libero_10/rollouts/none/task_rollout"),
    ("pi0.5/RoboTwin", "openpi", 14,
     "eval_out/openpi_robotwin_topup"),
]


def main():
    print("Raw recovery cosine cos(recovered_seed, injected_seed), same method per column.\n")
    hdr = (f"{'family':20s}{'n':>4s}{'n_wm':>6s}"
           f"{'keyed_perdim':>13s}{'keyed_flat':>11s}"
           f"{'act_perdim':>11s}{'act_flat':>9s}{'pad_flat':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for name, pipe, arg, root in FAMILIES:
        fs = sorted(glob.glob(root + "/**/*.npz", recursive=True))
        recs = []
        for p in fs:
            if len(recs) >= N_SAMPLE:
                break
            try:
                r = lingbot_cos(p, LB_PRESETS[arg]) if pipe == "lingbot" else openpi_cos(p, arg)
            except Exception as e:
                print(f"   skip {p.split('/')[-1]}: {type(e).__name__} {e}")
                continue
            if r:
                recs.append(r)
        if not recs:
            print(f"{name:20s}  (no watermarked episodes)")
            continue
        m = lambda k: float(np.mean([r[k] for r in recs if k in r])) if any(k in r for r in recs) else float("nan")
        def cell(k):
            v = m(k)
            return f"{v:>{ {'keyed_perdim':13,'keyed_flat':11,'act_perdim':11,'act_flat':9,'pad_flat':9}[k] }.3f}" \
                if not np.isnan(v) else f"{'-':>{ {'keyed_perdim':13,'keyed_flat':11,'act_perdim':11,'act_flat':9,'pad_flat':9}[k] }s}"
        print(f"{name:20s}{len(recs):4d}{m('n_wm'):6.1f}"
              f"{cell('keyed_perdim')}{cell('keyed_flat')}"
              f"{cell('act_perdim')}{cell('act_flat')}{cell('pad_flat')}")


if __name__ == "__main__":
    main()
