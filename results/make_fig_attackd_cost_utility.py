#!/usr/bin/env python3
"""Cost-utility figure for the direct adversarial Attack-D (adaptive fine-tune vs the verifier).

Two rows (datasets: LIBERO-10, RoboTwin-10) x two columns (metrics: per-episode AUC = det@G1,
group DIR@1%FAR). Each panel plots pi0.5 (blue circles) and LingBot (red squares); points are
connected in order of increasing attack strength with little arrowheads. x = watermarked task
success (utility); y = detection. det@G16 is omitted (saturated ~1.0 everywhere).

The story the figure tells: no adaptive fine-tune removes the mark without destroying the model.
pi0.5 (diffuse signal, no per-episode handle) -> detection stays pinned at every attack strength,
task success unmoved; the attack is a no-op on BOTH datasets. LingBot (strong per-episode handle)
-> can drive detection down, but only by spending all task utility (the curve only drops once
success hits 0).

Sources (RAW matched-filter summaries):
  pi0.5/LIBERO   per_episode_scores_work7_global_5050/_attackd_summary.csv  (+ rollout task succ)
  LingBot/LIBERO per_episode_scores_attackd_lingbot/_attackd_lingbot_summary.csv
  pi0.5/RoboTwin per_episode_scores_attackd_robotwin/_attackd_robotwin_summary.csv
  LingBot/RoboTwin per_episode_scores_attackd_lingbot_robotwin/_summary.csv  (optional)

Usage: python make_fig_attackd_cost_utility.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

DATA = Path("/workspace/vla")
SC = DATA / "attack_c_data"
PI_LIBERO = SC / "per_episode_scores_work7_global_5050/_attackd_summary.csv"
LB_LIBERO = SC / "per_episode_scores_attackd_lingbot/_attackd_lingbot_summary.csv"
PI_RT = SC / "per_episode_scores_attackd_robotwin/_attackd_robotwin_summary.csv"
LB_RT = SC / "per_episode_scores_attackd_lingbot_robotwin/_summary.csv"   # optional
OUT = DATA / "paper/figs/fig_attackd_cost_utility.pdf"

PI_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
LB_COLORS = ["#fcae91", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
PI_ORDER = ["lam0", "lam1", "lam5", "lam30"]               # weak -> strong
LB_ORDER = ["lam0", "tw100", "tw10", "tw1", "tw0"]         # weak -> strong
METRICS = [("detG1", "Per-episode AUC"), ("DIR1G16", "DIR @ 1% FAR")]


def plot_strength_path(ax, xs, ys, colors, marker, linestyle, zorder):
    pts = np.column_stack([xs, ys])
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(seg, colors=colors[1:len(xs)], linewidths=1.35,
                                     linestyles=linestyle, zorder=zorder))
    ax.scatter(xs, ys, s=24, marker=marker, color=colors[:len(xs)], edgecolor="none", zorder=zorder + 2)
    for x0, y0, x1, y1, c in zip(xs[:-1], ys[:-1], xs[1:], ys[1:], colors[1:len(xs)]):
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 0.002:
            continue
        ax.annotate("", xy=(x0 + 0.78 * (x1 - x0), y0 + 0.78 * (y1 - y0)),
                    xytext=(x0 + 0.48 * (x1 - x0), y0 + 0.48 * (y1 - y0)),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": c, "mutation_scale": 6.0,
                                "shrinkA": 0, "shrinkB": 0}, zorder=5)


def pi05_libero_task_success():
    succ, root = {}, DATA / "attack_c_data/rollouts/openpi_libero_attackd"
    for lam in PI_ORDER:
        fs = glob.glob(str(root / lam / "**" / "*_watermarked.npz"), recursive=True)
        s = [bool(np.load(x, allow_pickle=True)["success"]) for x in fs
             if "success" in np.load(x, allow_pickle=True).files]
        if s:
            succ[lam] = float(np.mean(s))
    return succ


def series(df, order, succ_col):
    """Return (xs, ys_by_metric) for the rows in `order` that exist in df (indexed by cell)."""
    keys = [k for k in order if k in df.index]
    xs = [float(df.loc[k, succ_col]) for k in keys]
    ys = {col: [float(df.loc[k, col]) for k in keys] for col, _ in METRICS}
    return xs, ys


def main():
    plt.rcParams.update({"font.size": 7.0, "axes.labelsize": 7.3, "axes.titlesize": 7.5,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7.0})

    # --- load each (model, dataset) summary; attach a succ_wm column keyed by cell ---
    pil = pd.read_csv(PI_LIBERO).set_index("lam")
    pil["succ_wm"] = pd.Series(pi05_libero_task_success())
    lbl = pd.read_csv(LB_LIBERO).set_index("cell") if LB_LIBERO.exists() else None
    pirt = pd.read_csv(PI_RT).set_index("cell") if PI_RT.exists() else None
    lbrt = pd.read_csv(LB_RT).set_index("cell") if LB_RT.exists() else None

    rows = [("LIBERO-10", (pil, PI_ORDER), (lbl, LB_ORDER)),
            ("RoboTwin-10", (pirt, PI_ORDER), (lbrt, LB_ORDER))]

    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.25), sharex=True, sharey=True)
    pi_handle = lb_handle = None
    for r, (dset, (pidf, piord), (lbdf, lbord)) in enumerate(rows):
        for c, (col, ylab) in enumerate(METRICS):
            ax = axes[r, c]
            ax.axvspan(-0.05, 0.05, color="gray", alpha=0.10, lw=0)
            if pidf is not None:
                xs, ys = series(pidf, piord, "succ_wm")
                if xs:
                    plot_strength_path(ax, xs, ys[col], PI_COLORS, "o", "solid", zorder=3)
                    pi_handle = Line2D([0], [0], color=PI_COLORS[-1], marker="o", lw=1.35, markersize=4.4)
            if lbdf is not None:
                xs, ys = series(lbdf, lbord, "succ_wm")
                if xs:
                    plot_strength_path(ax, xs, ys[col], LB_COLORS, "s", "dashed", zorder=4)
                    lb_handle = Line2D([0], [0], color=LB_COLORS[-1], marker="s", ls="--", lw=1.35, markersize=4.4)
            ax.set_ylim(-0.05, 1.12); ax.set_xlim(-0.10, 1.08)
            ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
            ax.grid(alpha=0.25, lw=0.45); ax.axhline(0.5, color="gray", ls=":", lw=0.6)
            if r == 0:
                ax.set_title(ylab, pad=2.0)
        axes[r, 0].set_ylabel(f"{dset}\nmetric value", fontsize=6.8)

    fig.text(0.55, 0.015, "Task success", ha="center", va="center", fontsize=7.0)
    handles = [h for h in (pi_handle, lb_handle) if h is not None]
    labels = [l for h, l in zip((pi_handle, lb_handle), ("$\\pi_{0.5}$", "LingBot")) if h is not None]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95), h_pad=0.8, w_pad=0.85)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    note = "" if lbrt is not None else "  (LingBot/RoboTwin row pending)"
    print(f"wrote {OUT}{note}")


if __name__ == "__main__":
    main()
