#!/usr/bin/env python3
"""Per-cell watermark-detection metrics for the lingbot RoboTwin SNR=0.05 re-run.

Reads the NPZ episodes written by the lingbot RoboTwin SNR=0.05 rollout under
eval_out/lingbot_rt_fix/ and scores each episode as sum(wmf_scores) -- the official
§12.5 per-episode convention used by aggregate_compression_results.py -- then reports
AUC / TPR@1% / TPR@10% / success per cell (clean, each attack tag, attacks-pooled,
descendant). wm vs plain are the positive/negative classes.

Safe to run at ANY time while the campaign is still going: it just reports whatever
episodes have landed so far. Re-run it to refresh. Also writes a markdown snapshot.

On-disk layout it parses (relative to --root):
  clean/<variant>/<task>/.../*.npz
  atk/<tag>/<variant>/<task>/.../*.npz        (tag e.g. clip_2.0, ema_0.5, jitter_0.01, delay_3)
  desc/<variant>/<task>/.../*.npz
with <variant> in {wm, plain}. Per-episode score = sum(npz['wmf_scores']).

Usage:
  python3 analyze_lingbot_rt_snr05.py                  # print table + write RESULTS_snr05.md
  python3 analyze_lingbot_rt_snr05.py --metric mean    # mean(wmf) per episode instead of sum
  python3 analyze_lingbot_rt_snr05.py --root <dir> --out <file.md>
"""
import argparse, glob, os, time
import numpy as np


# --- identical to aggregate_compression_results.py so numbers match the official pipeline ---
def roc_auc(pos, neg) -> float:
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort() + 1
    pos_rank_sum = ranks[: pos.size].sum()
    return float((pos_rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def tpr_at_fpr(pos, neg, target_fpr: float) -> float:
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    k = int(np.ceil(target_fpr * neg.size))
    if k <= 0:
        return 0.0
    thresh = np.sort(neg)[-k]
    return float(np.mean(pos > thresh))


def cell_of(relpath):
    """(group, variant) from a path relative to root. group is 'clean'/'desc'/'atk/<tag>'."""
    parts = relpath.split(os.sep)
    if parts[0] == "atk":
        return f"atk/{parts[1]}", parts[2]
    return parts[0], parts[1]


def episode_score(f, metric):
    # allow_pickle=True: these NPZ are produced by our own eval pipeline
    # and carry object arrays (e.g. task_description strings);
    # trusted local artifacts, same as aggregate_compression_results.py.
    d = np.load(f, allow_pickle=True)
    w = np.asarray(d["wmf_scores"], dtype=float)
    s = 0.0 if w.size == 0 else (float(w.sum()) if metric == "sum" else float(w.mean()))
    return s, bool(d["success"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/vla/eval_out/lingbot_rt_fix")
    ap.add_argument("--metric", choices=["sum", "mean"], default="sum")
    ap.add_argument("--out", default="/workspace/vla/eval_out/lingbot_rt_fix/RESULTS_snr05.md")
    a = ap.parse_args()

    data = {}  # group -> {"wm": [(score,succ)], "plain": [...]}
    for f in glob.glob(os.path.join(a.root, "**", "*.npz"), recursive=True):
        rel = os.path.relpath(f, a.root)
        if rel.startswith("_claims"):
            continue
        try:
            grp, var = cell_of(rel)
        except Exception:
            continue
        if var not in ("wm", "plain"):
            continue
        try:
            s, ok = episode_score(f, a.metric)
        except Exception:
            continue
        data.setdefault(grp, {"wm": [], "plain": []})[var].append((s, ok))

    atk_groups = sorted(g for g in data if g.startswith("atk/"))
    if atk_groups:
        pooled = {"wm": [], "plain": []}
        for g in atk_groups:
            for v in ("wm", "plain"):
                pooled[v] += data[g][v]
        data["atk(ALL)"] = pooled

    order = (["clean"] if "clean" in data else []) + atk_groups \
            + (["atk(ALL)"] if atk_groups else []) + (["desc"] if "desc" in data else [])

    hdr = "| cell | AUC | TPR@1% | TPR@10% | n_wm | n_plain | wm_mean | plain_mean | succ_wm | succ_plain |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        f"# lingbot RoboTwin SNR=0.05 re-run — detection by cell",
        f"_score = {a.metric}(wmf_scores) per episode; wm=positive, plain=negative; "
        f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "", hdr, sep,
    ]
    def fmt(x, p=3):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
    for g in order:
        wm = np.array([s for s, _ in data[g]["wm"]])
        pl = np.array([s for s, _ in data[g]["plain"]])
        sw = sum(ok for _, ok in data[g]["wm"])
        sp = sum(ok for _, ok in data[g]["plain"])
        lines.append(
            f"| {g} | {fmt(roc_auc(wm,pl))} | {fmt(tpr_at_fpr(wm,pl,0.01))} | {fmt(tpr_at_fpr(wm,pl,0.10))} "
            f"| {wm.size} | {pl.size} | {fmt(wm.mean() if wm.size else float('nan'),2)} "
            f"| {fmt(pl.mean() if pl.size else float('nan'),2)} | {sw}/{wm.size} | {sp}/{pl.size} |"
        )
    out = "\n".join(lines) + "\n"
    with open(a.out, "w") as fh:
        fh.write(out)
    print(out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
