#!/usr/bin/env python3
"""Render fig_aggregation_mode in the original curve style.

The figure is a compact single-column 2x2 grid. Each panel is one model-suite
cell. Solid curves are the default cross-task grouping; dashed curves constrain
each group to one task. The pi0.5/LIBERO data use the current work-7 + global
lag-search score directory, matching the other polished paper figures.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402

ROOT = HERE.parent / "attack_c_data"
RAW = ROOT / "per_episode_scores_raw"
WORK7 = ROOT / "per_episode_scores_work7_global_5050"
PAPER = HERE.parent / "paper"

FAMILY_ORDER = [
    "lingbot/libero_10",
    "lingbot/robotwin10",
    "pi0.5/libero_10",
    "pi0.5/robotwin10",
]
FAMILY_LABEL = {
    "lingbot/libero_10": "LingBot / LIBERO-10",
    "lingbot/robotwin10": "LingBot / RoboTwin",
    "pi0.5/libero_10": r"$\pi_{0.5}$ / LIBERO-10",
    "pi0.5/robotwin10": r"$\pi_{0.5}$ / RoboTwin",
}
WANTED = [
    ("clean", None, "Clean"),
    ("clip", 1.0, "Clip"),
    ("ema", 0.5, "EMA"),
    ("jitter", 0.01, "Jitter"),
]
G_GRID = [1, 2, 4, 8, 16, 32, 64]


def _strength_value(r0):
    st = r0.get("attack_strength")
    if pd.isna(st) or str(st).strip() in ("", "-"):
        return None
    try:
        return float(st)
    except ValueError:
        return None


def _load_from(score_dir):
    out = {}
    for p in sorted(score_dir.glob("*.csv")):
        if p.name.startswith("utility_") or p.name.startswith("_"):
            continue
        df = pd.read_csv(p)
        if "episode_id" not in df.columns:
            continue
        r0 = df.iloc[0]
        if str(r0["attack"]) not in {a for a, _, _ in WANTED}:
            continue
        fam = f'{r0["model"]}/{r0["dataset"]}'
        key = (fam, str(r0["attack"]), _strength_value(r0))
        out[key] = av.calibrate(df)
    return out


def load_conditions():
    conditions = _load_from(RAW)
    for key, cal in _load_from(WORK7).items():
        if key[0] == "pi0.5/libero_10":
            conditions[key] = cal
    return conditions


def _same_task_tpr(cal, G, seed):
    rng = np.random.default_rng(seed)
    return av.tpr_point_same_task(
        cal.z_h1, cal.task_h1, cal.z_null, cal.task_null_wm,
        G, av.FPR_MAIN, rng, n_trials=av.N_TRIALS
    )


def _cross_task_tpr(cal, G, seed):
    rng = np.random.default_rng(seed)
    return av.tpr_point(cal.z_h1, cal.z_null, G, av.FPR_MAIN, rng, n_trials=av.N_TRIALS)


def compute_rows(conditions):
    rows = []
    for fam in FAMILY_ORDER:
        for attack, strength, label in WANTED:
            cal = conditions.get((fam, attack, strength))
            if cal is None:
                continue
            n_tasks = len(set(cal.task_h1) & set(cal.task_null_wm))
            for G in G_GRID:
                seed = av.RNG_SEED + 1009 * FAMILY_ORDER.index(fam) + 97 * G + 13 * len(rows)
                t_cross = _cross_task_tpr(cal, G, seed)
                t_same = _same_task_tpr(cal, G, seed + 1)
                rows.append(dict(
                    family=fam,
                    attack=attack,
                    strength=strength,
                    label=label,
                    G=G,
                    n_tasks=n_tasks,
                    tpr_cross=t_cross,
                    tpr_same=t_same,
                    same_minus_cross=t_same - t_cross,
                ))
    return pd.DataFrame(rows)


def draw(df, path):
    fig, axes = plt.subplots(
        2, 2, figsize=(3.48, 2.72), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.84,
                        wspace=0.10, hspace=0.42)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10").colors

    for ax, fam in zip(axes, FAMILY_ORDER):
        sub = df[df["family"] == fam]
        for i, (_, _, label) in enumerate(WANTED):
            g = sub[sub["label"] == label].sort_values("G")
            if g.empty:
                continue
            color = colors[i]
            ax.plot(g["G"], g["tpr_same"], marker="o", ms=2.4, lw=1.0,
                    color=color, label=label)
            ax.plot(g["G"], g["tpr_cross"], marker="x", ms=2.6, lw=0.9,
                    ls="--", color=color, alpha=0.85)
        ax.set_title(FAMILY_LABEL[fam], fontsize=6.6, pad=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(G_GRID)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(-0.03, 1.04)
        ax.grid(alpha=0.22, lw=0.45)
        ax.tick_params(labelsize=5.8, length=1.6, pad=1)

    for ax in axes[2:]:
        ax.set_xlabel(r"$|G|$", fontsize=6.4, labelpad=1)
    for ax in axes[::2]:
        ax.set_ylabel("TPR", fontsize=6.4, labelpad=1)

    handles = [
        Line2D([0], [0], color=colors[i], lw=1.1, label=label)
        for i, (_, _, label) in enumerate(WANTED)
    ]
    labels = [label for _, _, label in WANTED]
    handles += [
        Line2D([0], [0], color="0.2", lw=1.0, marker="o", ms=2.4, label="same-task"),
        Line2D([0], [0], color="0.2", lw=0.9, ls="--", marker="x", ms=2.6, label="cross-task"),
    ]
    labels += ["same-task", "cross-task"]
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=5.7,
               frameon=False, bbox_to_anchor=(0.52, 0.985),
               handlelength=1.35, columnspacing=0.9, handletextpad=0.35,
               labelspacing=0.35)
    fig.savefig(path)
    plt.close(fig)


def main():
    conditions = load_conditions()
    df = compute_rows(conditions)
    out_csv = HERE / "aggregation_mode.csv"
    df.to_csv(out_csv, index=False)
    out_pdf = HERE / "fig_aggregation_mode.pdf"
    draw(df, out_pdf)
    shutil.copy(out_pdf, PAPER / "fig_aggregation_mode.pdf")

    meaningful = df[df["family"].str.startswith("lingbot/")]
    summary = meaningful.groupby("family").agg(
        n_points=("attack", "size"),
        n_tasks=("n_tasks", "first"),
        max_abs_delta=("same_minus_cross", lambda x: float(np.max(np.abs(x)))),
        mean_abs_delta=("same_minus_cross", lambda x: float(np.mean(np.abs(x)))),
    )
    print(summary.to_string())
    print(f"wrote {out_pdf}")
    print(f"wrote {PAPER / 'fig_aggregation_mode.pdf'}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
