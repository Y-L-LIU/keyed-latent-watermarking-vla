#!/usr/bin/env python3
"""Flat 4x4 identification-robustness figure (closed-set CMC).

The identification counterpart of make_fig_attack_combined.py, in the same flat
style: rows = model x dataset families, cols = output attacks. Each panel shows
the closed-set CMC over the 33-key gallery -- P(rank of the true key <= r) vs
rank r -- for the clean policy (black) and the attack swept over strength
(plasma, light->dark = weaker->stronger), at the group budget |G|=G. The dot at
rank 1 is the headline rank-1 identification rate. Saturated panels (rank-1 >=
0.99 for clean + every strength) carry a 'rank-1 = 1.00' annotation instead of
flat curves. Legend labels carry the fingerprinted policy's success rate (the
utility the attacker pays), matching the verification figure.

pi0.5/LIBERO is scored with work-7+global (its canonical per-episode scores);
the other three families use the raw all-32 scores. Identical data sourcing to
make_fig_attack_combined. Closed-set CMC needs only watermarked rows, so it is
available for every cell (unlike open-set DIR@FAR, whose impostor pool is absent
for most LingBot/LIBERO attack cells).

Run: python3 make_fig_identification_combined.py [--preview PNG] [--no-paper]
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_identification as ai      # noqa: E402  CMC math + loader
import analyze_verification as av        # noqa: E402  utility (SR) loader
import make_fig_attack_combined as mf    # noqa: E402  shared layout constants

# Same data sources + geometry as the verification figure.
RAW = mf.RAW
WORK7 = mf.WORK7
WORK7_FAMILY = mf.WORK7_FAMILY
DATA = mf.DATA
FAMILIES = mf.FAMILIES
FAM_YLABEL = mf.FAM_YLABEL
ATTACKS = mf.ATTACKS
ATTACK_TITLE = mf.ATTACK_TITLE
SLAB = mf.SLAB
FIG_W, FIG_H = mf.FIG_W, mf.FIG_H
G = mf.G                       # group budget (uniform; 32 to match the verif fig)
GALLERY = ai.GALLERY           # 33 = true key + 32 decoys
av.SCORE_DIR = RAW             # utility CSVs live in the raw dir (detector-independent)


def discover_mixed():
    """Conditions from RAW for every family except pi0.5/LIBERO, which is read
    from the work-7+global score dir (same CSV schema)."""
    conds = []
    for p in sorted(RAW.glob("*_partial_map_*.csv")):
        if p.name.startswith("pi05_libero_10_partial_map_"):
            continue
        conds.append(ai._load_csv(p, "main"))
    for p in sorted(WORK7.glob("pi05_libero_10_partial_map_*.csv")):
        conds.append(ai._load_csv(p, "main"))
    return conds


def build_cache():
    conds = discover_mixed()
    util = av.load_utility()
    mf.inject_delay_sr(util)
    mf.inject_lingbot_libero_sr(util)
    mf.inject_sr_from_conds(util)
    rows, cache = [], {}
    for c in conds:
        cache[(c["family"], c["attack"], c["strength"])] = ai.calibrate(c["df"])
        if not mf.in_display_grid(c["attack"], c["strength"]):
            continue  # off-grid strength: keep in cache, hide from the figure
        rows.append(dict(family=c["family"], model=c["model"], dataset=c["dataset"],
                         attack=c["attack"], strength=c["strength"]))
    return pd.DataFrame(rows), util, cache


def draw(M, util, cache, path):
    rng = np.random.default_rng(ai.RNG_SEED)
    fams = [f for f in FAMILIES if f in set(M["family"])]
    nR, nC = len(fams), len(ATTACKS)
    fig, axes = plt.subplots(nR, nC, figsize=(FIG_W, FIG_H), squeeze=False,
                             constrained_layout=True)
    ranks = np.arange(1, GALLERY + 1)

    for i, fam in enumerate(fams):
        fr = M[M["family"] == fam]
        model, ds = fr.iloc[0]["model"], fr.iloc[0]["dataset"]
        for j, attack in enumerate(ATTACKS):
            ax = axes[i][j]
            ax.set_xlim(1, GALLERY)
            ax.set_xticks([1, 11, 22, 33])
            ax.set_ylim(-0.02, 1.04)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(alpha=0.22, lw=0.5, which="major")
            ax.tick_params(labelsize=7, length=2.5, pad=1.5)

            if i == 0:
                ax.set_title(ATTACK_TITLE[attack], fontsize=12, weight="bold", pad=4)
            if j == 0:
                ax.set_ylabel(FAM_YLABEL[fam], fontsize=9.5, weight="bold",
                              labelpad=4, linespacing=1.0)
            if i == nR - 1:
                ax.set_xlabel(r"rank $r$", fontsize=8.5, labelpad=1)
            ax.text(0.035, 0.93, rf"$|G|{{=}}{G}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.0, color="0.45")

            sub = M[(M["family"] == fam) & (M["attack"] == attack)
                    & M["strength"].notna()].sort_values("strength")
            if len(sub) == 0:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=11, color="0.6")
                continue

            r1s = []
            ck = (fam, "clean", None)
            if ck in cache:
                cmc = ai.cmc_curve(cache[ck], G, rng)[0]
                r1s.append(cmc[0])
                srs = mf.sr_str(util, model, ds, "clean", None)
                lab = "clean" + (f" ({srs})" if srs else "")
                ax.plot(ranks, cmc, color="black", lw=2.2, zorder=6, label=lab)
                ax.plot(1, cmc[0], "o", color="black", ms=4.5, zorder=7)

            strengths = sub["strength"].values
            colors = plt.cm.plasma(np.linspace(0.12, 0.72, len(strengths)))
            for k, s in enumerate(strengths):
                key = (fam, attack, s)
                if key not in cache:
                    continue
                cmc = ai.cmc_curve(cache[key], G, rng)[0]
                r1s.append(cmc[0])
                srs = mf.sr_str(util, model, ds, attack, s)
                lab = SLAB[attack](s) + (f" ({srs})" if srs else "")
                ax.plot(ranks, cmc, color=colors[k], lw=1.6, zorder=4, label=lab)
                ax.plot(1, cmc[0], "o", color=colors[k], ms=3.5, zorder=5)

            # Saturated panels: rank-1 already perfect for clean + every strength,
            # so the CMC curves are flat at 1.0 -- state the headline instead.
            if r1s and min(r1s) >= 0.99:
                ax.text(0.5, 0.58, "rank-1 = 1.00", transform=ax.transAxes,
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
    M, util, cache = build_cache()
    draw(M, util, cache, av.OUT_DIR / "fig_identification_combined.pdf")
    if not args.no_paper:
        draw(M, util, cache, DATA / "paper" / "fig_identification_combined.pdf")
    if args.preview:
        draw(M, util, cache, args.preview)


if __name__ == "__main__":
    main()
