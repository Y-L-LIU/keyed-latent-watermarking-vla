#!/usr/bin/env python3
"""Flat 4x4 OPEN-SET identification-robustness figure (DIR@FAR).

The open-set counterpart of make_fig_identification_combined.py and the direct
ROC analog of make_fig_attack_combined.py: same axes (false-alarm rate on a log
x, rate on y), same flat layout. Each panel shows the detection-and-identification
rate DIR (rank-1 correct AND above an impostor-calibrated threshold) vs.\ the
impostor false-alarm rate, for the clean policy (black) and the attack swept over
strength (plasma), at group budget |G|=G. The dot marks the 1% FAR operating
point. As FAR->1 the curve relaxes to the closed-set rank-1 ceiling, so each
panel also shows the closed-set result at its right edge.

Impostors are no-key (plain) rollouts. Where an attack cell has no plain pool
(most LingBot/LIBERO attack CSVs), we borrow the family's clean-cell plain pool:
the open-set null is attack-invariant (a no-key rollout stays null under output
attacks), verified to match the own-pool DIR within MC noise where both exist.

pi0.5/LIBERO uses work-7+global; other families raw all-32 -- same as the
verification figure. Run: python3 make_fig_identification_openset.py [--preview PNG] [--no-paper]
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_identification as ai      # noqa: E402
import analyze_verification as av        # noqa: E402
import make_fig_attack_combined as mf    # noqa: E402
import make_fig_identification_combined as mfi  # noqa: E402  (build_cache reuse)

FAMILIES = mf.FAMILIES
FAM_YLABEL = mf.FAM_YLABEL
ATTACKS = mf.ATTACKS
ATTACK_TITLE = mf.ATTACK_TITLE
SLAB = mf.SLAB
FIG_W, FIG_H = mf.FIG_W, mf.FIG_H
G = mf.G
FPR_OP = mf.FPR_OP                       # 1% operating point
DATA = mf.DATA
N_TRIALS = 60_000
FAR_GRID = np.logspace(np.log10(8e-4), 0.0, 60)


def _impostor(cache, fam, cal):
    """The cell's own plain pool if present, else the family's clean-cell pool."""
    if len(cal.zt_pl) > 0:
        return cal
    return cache.get((fam, "clean", None))


def dir_curve(gen, imp, G, fars, rng, n_trials=N_TRIALS):
    """DIR (rank-1 correct AND above tau) over a sweep of impostor FARs. Returns
    a DIR array aligned to `fars`. NaN array if no impostor pool is available."""
    if imp is None or len(imp.zt_pl) == 0:
        return np.full(len(fars), np.nan)
    Tt_g, Td_g = ai._group_scores(gen.zt_wm, gen.zd_wm, G, rng, n_trials)
    rank1 = Tt_g >= Td_g.max(axis=1)
    Tt_i, Td_i = ai._group_scores(imp.zt_pl, imp.zd_pl, G, rng, n_trials)
    smax = np.maximum(Tt_i, Td_i.max(axis=1))
    taus = np.quantile(smax, 1.0 - np.asarray(fars))
    return np.array([float((rank1 & (Tt_g >= t)).mean()) for t in taus])


def draw(M, util, cache, path):
    rng = np.random.default_rng(ai.RNG_SEED)
    fams = [f for f in FAMILIES if f in set(M["family"])]
    nR, nC = len(fams), len(ATTACKS)
    fig, axes = plt.subplots(nR, nC, figsize=(FIG_W, FIG_H), squeeze=False,
                             constrained_layout=True)

    for i, fam in enumerate(fams):
        fr = M[M["family"] == fam]
        model, ds = fr.iloc[0]["model"], fr.iloc[0]["dataset"]
        for j, attack in enumerate(ATTACKS):
            ax = axes[i][j]
            ax.set_xscale("log")
            ax.set_xlim(8e-4, 1.06)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
            ax.set_ylim(-0.02, 1.04)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(alpha=0.22, lw=0.5, which="major")
            ax.tick_params(labelsize=7, length=2.5, pad=1.5)
            ax.axvline(FPR_OP, color="0.55", ls=":", lw=0.9, zorder=1)

            if i == 0:
                ax.set_title(ATTACK_TITLE[attack], fontsize=12, weight="bold", pad=4)
            if j == 0:
                ax.set_ylabel(FAM_YLABEL[fam], fontsize=9.5, weight="bold",
                              labelpad=4, linespacing=1.0)
            if i == nR - 1:
                ax.set_xlabel("impostor false-alarm rate", fontsize=8.5, labelpad=1)
            ax.text(0.035, 0.93, rf"$|G|{{=}}{G}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.0, color="0.45")

            sub = M[(M["family"] == fam) & (M["attack"] == attack)
                    & M["strength"].notna()].sort_values("strength")
            if len(sub) == 0:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=11, color="0.6")
                continue

            op_dirs = []
            ck = (fam, "clean", None)
            if ck in cache:
                d = dir_curve(cache[ck], _impostor(cache, fam, cache[ck]), G, FAR_GRID, rng)
                srs = mf.sr_str(util, model, ds, "clean", None)
                lab = "clean" + (f" ({srs})" if srs else "")
                ax.plot(FAR_GRID, d, color="black", lw=2.2, zorder=6, label=lab)
                op = float(np.interp(FPR_OP, FAR_GRID, d))
                ax.plot(FPR_OP, op, "o", color="black", ms=4.5, zorder=7)
                op_dirs.append(op)

            strengths = sub["strength"].values
            colors = plt.cm.plasma(np.linspace(0.12, 0.72, len(strengths)))
            for k, s in enumerate(strengths):
                key = (fam, attack, s)
                if key not in cache:
                    continue
                d = dir_curve(cache[key], _impostor(cache, fam, cache[key]), G, FAR_GRID, rng)
                srs = mf.sr_str(util, model, ds, attack, s)
                lab = SLAB[attack](s) + (f" ({srs})" if srs else "")
                ax.plot(FAR_GRID, d, color=colors[k], lw=1.6, zorder=4, label=lab)
                op = float(np.interp(FPR_OP, FAR_GRID, d))
                ax.plot(FPR_OP, op, "o", color=colors[k], ms=3.5, zorder=5)
                op_dirs.append(op)

            if op_dirs and min(op_dirs) >= 0.99:
                ax.text(0.5, 0.58, "DIR@1% = 1.00", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9.5, color="0.30",
                        weight="medium")

            leg = ax.legend(loc="lower right", fontsize=5.6, frameon=True,
                            framealpha=0.72, handlelength=1.0, handletextpad=0.4,
                            labelspacing=0.14, borderaxespad=0.25)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("none")

    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="")
    ap.add_argument("--no-paper", action="store_true")
    args = ap.parse_args()
    M, util, cache = mfi.build_cache()
    draw(M, util, cache, av.OUT_DIR / "fig_identification_openset.pdf")
    if not args.no_paper:
        draw(M, util, cache, DATA / "paper" / "fig_identification_openset.pdf")
    if args.preview:
        draw(M, util, cache, args.preview)


if __name__ == "__main__":
    main()
