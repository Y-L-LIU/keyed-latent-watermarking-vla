#!/usr/bin/env python3
"""Re-score pi0.5/LIBERO ATTACK rollouts with the matched filter restricted to
the WORK dims (0..6) -- the "inject-ALL32 / detect 0-6" detector.

The committed rollouts already injected all 32 latent dims (the main experiment);
this is a PURE OFFLINE re-score of the stored chunk_recovered_noise +
chunk_reference, slicing both to columns 0:7 before each per-window cosine. No
rollouts are run. The all-32 detector (current paper main result) is already in
attack_c_data/per_episode_scores_raw/<stem>.csv -- this writes the work-7 arm to
attack_c_data/per_episode_scores_raw_work7/<stem>.csv (committed schema, s_true
= raw matched filter on dims 0..6) so analyze_verification / analyze_identification
run on it unchanged, and prints a per-attack comparison:

  detAUC @|G|=1, @|G|=16   (separability)
  identR1 @|G|=1, @|G|=16  (closed-set rank-1 over the 33-key gallery)
  DIR@1%  @|G|=16          (open-set, uses the plain impostor pool)

for both detectors side by side. Lag-0 (no sync search), matching the committed
table.

Usage: rescore_detect_work7.py [--probe N] [--no-write]
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_scorer_allcells as asc          # noqa: E402  (sets up openpi import paths)
import build_raw_perep as bp                     # noqa: E402  (CONDS / META / episode_id / HEADER)
import analyze_verification as av                # noqa: E402
import analyze_identification as ai              # noqa: E402

op_wm = asc.op_wm
J = asc.J
DATA = asc.DATA
WORK_DIMS = list(range(0, 7))                     # LIBERO actuated dims (7-DoF arm+gripper)
OUT = DATA / "attack_c_data" / "per_episode_scores_raw_work7"
COMMITTED_RAW = DATA / "attack_c_data" / "per_episode_scores_raw"


# --- dim-restricted scorers (byte-identical to build_cost_utility_table's
#     _score_episode/_null_episode except the cosine operands are sliced to dims) #
def _score_episode_dims(d, dims, max_windows=5):
    selected = d["chunk_selected"]; recovered = d["chunk_recovered_noise"]
    reference = d["chunk_reference"]; executed = d["chunk_executed_steps"]
    out = []; count = 0
    for i in range(len(selected)):
        if not selected[i] or executed[i] <= 0:
            continue
        if count >= max_windows:
            break
        steps = reference[i].shape[0]            # full_chunk
        out.append(asc._cosine_sim(recovered[i][:steps][:, dims], reference[i][:steps][:, dims]))
        count += 1
    return np.asarray(out, dtype=np.float32)


def _null_episode_dims(d, off, dims, key, srate, max_windows=5):
    selected = d["chunk_selected"]; recovered = d["chunk_recovered_noise"]
    reference = d["chunk_reference"]; executed = d["chunk_executed_steps"]
    chunk_idx = d["chunk_index"] if "chunk_index" in d.files else d["chunk_chunk_index"]
    nonce = int(d["episode_nonce"])
    action_dim = int(reference.shape[-1]); horizon = int(reference.shape[1])
    cfg = op_wm.InternalNoiseWatermarkConfig(secret_key=key + off, control_freq=srate)
    out = []; count = 0
    for i in range(len(selected)):
        if not selected[i] or executed[i] <= 0:
            continue
        if count >= max_windows:
            break
        ctx = op_wm.WatermarkContext(chunk_index=int(chunk_idx[i]), episode_nonce=nonce)
        ref = op_wm.generate_keyed_reference(length=horizon, action_dim=action_dim,
                                             sample_rate_hz=srate, config=cfg, context=ctx)
        steps = ref.shape[0]
        out.append(asc._cosine_sim(recovered[i][:steps][:, dims], ref[:steps][:, dims]))
        count += 1
    return np.asarray(out, dtype=np.float32)


def score_openpi_dims(npz, dims):
    d = np.load(npz, allow_pickle=True)
    variant = str(d["variant"]) if "variant" in d.files else "watermarked"
    key = int(d["secret_key"]) if "secret_key" in d.files else 12345
    srate = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 50.0
    tv = _score_episode_dims(d, dims)
    if tv.shape[0] == 0:
        return None
    nm = np.stack([_null_episode_dims(d, off + 1, dims, key, srate) for off in range(J)])
    return asc._finish(variant, tv, nm, asc._wmf_score)   # (variant, st_w, sf_w, st_r, sf_r)


# --- only the pi0.5/LIBERO conditions from the committed build list ---------- #
def pi05_libero_conds():
    return [c for c in bp.CONDS if c[0] == "pi0.5/libero_10"]


def build_work7_csv(cond, probe=0):
    cell, attack, strength, stem, dirs = cond
    m = bp.META[cell]
    npz = []
    for d in dirs:
        npz += sorted((DATA / d).rglob("*.npz"))
    if probe:
        npz = npz[:probe]
    rows = []
    for p in npz:
        try:
            r = score_openpi_dims(p, WORK_DIMS)
        except Exception as e:
            print(f"   skip {p.name}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r is None:
            continue
        variant, _st_w, _sf_w, st_r, sf_r = r
        eid = bp.episode_id(cell, attack, p, variant)
        base = [eid, variant, m["model"], m["dataset"], "partial", m["obs_ratio"], "map", attack, strength, 1]
        rows.append(base + [f"{st_r:.8f}"] + [f"{x:.8f}" for x in sf_r])
    return stem, rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(bp.HEADER); w.writerows(rows)


# --- metrics ---------------------------------------------------------------- #
def metrics(df):
    rng = np.random.default_rng(av.RNG_SEED)
    cal = av.calibrate(df)
    det1 = av.auc(cal.z_h1, cal.z_null.ravel())
    det16 = av.auc_group(cal.z_h1, cal.z_null, 16, rng)
    ical = ai.calibrate(df)
    r1_1 = ai.cmc_curve(ical, 1, rng)[0][0]
    r1_16 = ai.cmc_curve(ical, 16, rng)[0][0]
    dir16 = ai.dir_at_far(ical, 16, [0.01], rng)[0.01]
    nwm = int((df["variant"] == "watermarked").sum()); npl = int((df["variant"] == "plain").sum())
    return dict(det1=det1, det16=det16, r1_1=r1_1, r1_16=r1_16, dir16=dir16, nwm=nwm, npl=npl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    if not args.probe and not args.no_write:
        OUT.mkdir(exist_ok=True)

    conds = pi05_libero_conds()
    print(f"pi0.5/LIBERO  work-dims=detect{WORK_DIMS[0]}-{WORK_DIMS[-1]} vs all-32 (committed)  lag-0\n")
    hdr = (f"{'attack':14s}{'n(wm/pl)':>10s} | "
           f"{'det@G1':>14s}{'det@G16':>14s} | {'idR1@G1':>14s}{'idR1@G16':>14s} | {'DIR1%@G16':>14s}")
    print(hdr); print("-" * len(hdr))
    print("                          |   all32   work7  all32   work7 |  all32   work7  all32   work7 |  all32   work7")
    for cond in conds:
        stem = cond[3]; attack = cond[1]; strength = cond[2]
        stem_w7, rows = build_work7_csv(cond, probe=args.probe)
        if not rows:
            print(f"{attack+' '+strength:14s}  (no rows)"); continue
        if not args.probe and not args.no_write:
            write_csv(OUT / f"{stem_w7}.csv", rows)
        # work-7 metrics from the rows just built
        w7df = pd.DataFrame([r for r in rows], columns=bp.HEADER)
        for c in ["s_true"] + [f"s_false_{i+1}" for i in range(J)]:
            w7df[c] = w7df[c].astype(float)
        mw = metrics(w7df)
        # all-32 baseline from committed CSV
        cpath = COMMITTED_RAW / f"{stem}.csv"
        if cpath.exists():
            ma = metrics(pd.read_csv(cpath))
        else:
            ma = dict(det1=float("nan"), det16=float("nan"), r1_1=float("nan"),
                      r1_16=float("nan"), dir16=float("nan"), nwm=0, npl=0)
        lbl = f"{attack} {strength}".strip()
        ncol = f"{mw['nwm']}/{mw['npl']}"
        print(f"{lbl:14s}{ncol:>10s} | "
              f"{ma['det1']:7.3f}{mw['det1']:8.3f}{ma['det16']:7.3f}{mw['det16']:8.3f} | "
              f"{ma['r1_1']:7.3f}{mw['r1_1']:8.3f}{ma['r1_16']:7.3f}{mw['r1_16']:8.3f} | "
              f"{ma['dir16']:7.3f}{mw['dir16']:8.3f}")
    if not args.probe and not args.no_write:
        print(f"\nwrote work-7 per-episode CSVs -> {OUT}")


if __name__ == "__main__":
    main()
