#!/usr/bin/env python3
"""Render the security-analysis summary figure.

The figure is intentionally compact: the text defines the security quantities,
and the figure shows the two operating facts the section needs.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT / "paper"


def _rows_after(path: Path, marker: str) -> list[list[str]]:
    rows: list[list[str]] = []
    active = False
    with path.open(newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                active = False
                continue
            if line.startswith("#"):
                active = marker in line
                continue
            if active:
                rows.append(next(csv.reader([line])))
    return rows


def load_true_z(path: Path) -> np.ndarray:
    rows = _rows_after(path, "true-key Z")
    return np.asarray([float(r[1]) for r in rows if r and r[0] != "episode_idx"], dtype=float)


def load_operating_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _rows_after(path, "operating points")
    fpr, tau, tpr = [], [], []
    for r in rows:
        if not r or r[0] == "target_fpr":
            continue
        fpr.append(float(r[0]))
        tau.append(float(r[1]))
        tpr.append(float(r[2]))
    return np.asarray(fpr), np.asarray(tau), np.asarray(tpr)


def load_unforge_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    active = False
    with path.open(newline="") as f:
        for raw in f:
            line = raw.strip()
            if line == "# summary":
                active = True
                continue
            if not active or not line.startswith("# "):
                continue
            row = next(csv.reader([line[2:]]))
            if len(row) >= 2:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def main() -> None:
    rng = np.random.default_rng(20260529)
    null_tg = np.load(HERE / "key_collision_TG_null.npy")
    true_z = load_true_z(HERE / "key_collision_analysis.csv")
    fpr, tau, tpr = load_operating_points(HERE / "key_collision_analysis.csv")
    unforge = load_unforge_summary(HERE / "unforgeability_analysis.csv")

    group_size = 16
    true_tg = true_z[rng.integers(0, len(true_z), size=(200_000, group_size))].sum(axis=1)

    plt.rcParams.update(
        {
            "font.size": 6.9,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.9,
        }
    )
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(3.48, 1.62),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )

    bins = np.linspace(-16, 54, 72)
    ax0.hist(null_tg, bins=bins, density=True, color="#9aa6b2", alpha=0.58, label="false key")
    ax0.hist(true_tg, bins=bins, density=True, color="#1f6f8b", alpha=0.62, label="true key")
    for fp, th in zip(fpr, tau):
        if fp not in {1e-3, 1e-6, 1e-9}:
            continue
        ax0.axvline(th, color="#b23a48", lw=1.0, ls="--" if fp > 1e-6 else ":")
    ax0.scatter(
        [unforge.get("hillclimb_group_TG", -1.3)],
        [ax0.get_ylim()[1] * 0.07],
        marker="x",
        s=20,
        color="#4b5563",
        linewidths=1.0,
        label="hill-climb",
        zorder=5,
    )
    ax0.set_yscale("log")
    ax0.set_ylim(3e-5, 0.22)
    ax0.set_xlim(-15, 52)
    ax0.set_xlabel(r"$T_G(k)$", labelpad=0.8)
    ax0.set_ylabel("density", labelpad=0.8)
    ax0.set_title(r"(a) Key separation")
    ax0.legend(loc="upper left", frameon=False, handlelength=1.1,
               borderpad=0.1, labelspacing=0.15)
    ax0.grid(alpha=0.22, which="both")

    budget = 1.0 / fpr
    ax1.plot(tau, budget, "o-", color="#305cde", lw=1.35, ms=3.5)
    for fp, th, b, _q in zip(fpr, tau, budget, tpr):
        exp = int(round(math.log10(fp)))
        ax1.text(th, b * 1.45, rf"$10^{{{exp}}}$", ha="center", va="bottom", fontsize=5.9)
    ax1.set_yscale("log")
    ax1.set_ylim(5e2, 3e10)
    ax1.set_xlim(min(tau) - 1.2, max(tau) + 1.2)
    ax1.set_xlabel(r"threshold $\tau_{\mathrm{dec}}$", labelpad=0.8)
    ax1.set_ylabel(r"guesses $1/p_{\mathrm{coll}}$", labelpad=0.8)
    ax1.set_title("(b) Forgery budget")
    ax1.grid(alpha=0.25, which="both")

    for ax in (ax0, ax1):
        ax.tick_params(axis="both", which="major", pad=1.2)
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.25, top=0.84, wspace=0.34)
    for out in [
        HERE / "fig_security_summary.pdf",
        PAPER / "fig_security_summary.pdf",
        HERE / "fig_security_summary_preview.png",
    ]:
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
