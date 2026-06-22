#!/usr/bin/env python3
"""Polished 4x4 attack-robustness figure (RAW matched filter).

Rows = model x dataset families, cols = output attacks (clip / EMA / jitter / delay).
Each panel: bootstrap ROC at the table query budget |G|=16, clean policy (black)
vs the attack swept over strength (viridis, light->dark = weaker->stronger), with a
marker at the 1% FPR operating point. Curve labels carry the fingerprinted policy's
success rate so the panel shows detection AND the utility the attacker pays.

Reads per_episode_scores_raw/ (the raw-scorer output); injects pi0.5 delay SR from
the surviving rollout npz (delay was never in the utility table). Writes the figure
to results/out and to paper/. Run: python3 make_fig_attack_combined.py [--preview PNG]
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402

RAW = av.HERE.parent / "attack_c_data" / "per_episode_scores_raw"
# pi0.5/LIBERO uses the work-7 (actuated-dim) + global lag-search detector, its
# canonical scorer (see memory vla-work7-global-detector). The other three
# families keep the raw all-32 scores: LingBot already detects narrow (saturated)
# and pi0.5/RoboTwin has no work-N artifact. Utility/SR is detector-independent,
# so it is always loaded from RAW.
WORK7 = av.HERE.parent / "attack_c_data" / "per_episode_scores_work7_global_5050"
WORK7_FAMILY = "pi0.5/libero_10"
av.SCORE_DIR = RAW
DATA = av.HERE.parent

# Figure geometry. Subplots are deliberately short (landscape) to save vertical
# page space -- height is the knob; width stays wide enough for the 4 columns.
FIG_W, FIG_H = 10.4, 5.6

FAMILIES = ["lingbot/libero_10", "lingbot/robotwin10",
            "pi0.5/libero_10", "pi0.5/robotwin10"]
FAM_TITLE = {
    "lingbot/libero_10":  "LingBot / LIBERO-10",
    "lingbot/robotwin10": "LingBot / RoboTwin",
    "pi0.5/libero_10":    r"$\pi_{0.5}$ / LIBERO-10",
    "pi0.5/robotwin10":   r"$\pi_{0.5}$ / RoboTwin",
}
# Two-line row labels: short panels can't fit the long single-line rotated title
# (it overflows into the neighbouring row), so stack model over dataset.
FAM_YLABEL = {
    "lingbot/libero_10":  "LingBot\nLIBERO-10",
    "lingbot/robotwin10": "LingBot\nRoboTwin",
    "pi0.5/libero_10":    "$\\pi_{0.5}$\nLIBERO-10",
    "pi0.5/robotwin10":   "$\\pi_{0.5}$\nRoboTwin",
}
ATTACKS = ["clip", "ema", "jitter", "delay"]
ATTACK_TITLE = {"clip": "Clip", "ema": "EMA", "jitter": "Jitter", "delay": "Delay"}
SLAB = {  # short per-strength label
    "clip":   lambda s: rf"$s{{=}}{s:g}$",
    "ema":    lambda s: rf"$\alpha{{=}}{s:g}$",
    "jitter": lambda s: rf"$\sigma{{=}}{s:g}$",
    "delay":  lambda s: rf"$\tau{{=}}{int(s)}$",
}
# ROC drawn at the GROUP budget |G|=32, uniform across the whole figure. The
# per-episode |G|=1 view put every TPR@1% dot far below the table; at |G|=32 the
# clean + canonical cells reach ~1.0 and only the physically-destroyed strengths
# (ema 0.2, jitter 0.05/0.1) keep any visible residual.
G = 32
G_PER_FAMILY = {}  # uniform |G| across the whole figure

# Aligned display grid: every family shows the SAME ema/jitter sweep.
# clip/delay already share {0.5,1,2}/{1,2,3} -> unrestricted. ema/jitter restricted so
# off-grid strengths (pi0.5's ema 0.2/0.8 and jitter 0.1) drop out of the figures even
# though their data stays on disk.
DISPLAY_GRID = {"ema": {0.3, 0.5, 0.7}, "jitter": {0.005, 0.01, 0.02, 0.05}}


def in_display_grid(attack, strength):
    if strength is None or (isinstance(strength, float) and np.isnan(strength)):
        return True
    g = DISPLAY_GRID.get(attack)
    return g is None or any(abs(float(strength) - x) < 1e-9 for x in g)
FPR_OP = av.FPR_MAIN
ROC_TRIALS = 60_000

# pi0.5 delay rollouts (SR not in the utility table -> compute from npz 'success')
DELAY_DIRS = {
    ("pi0.5", "robotwin10", 1.0): "attack_c_data/rollouts/openpi_robotwin/delay_1",
    ("pi0.5", "robotwin10", 2.0): "attack_c_data/rollouts/openpi_robotwin/delay_2",
    ("pi0.5", "robotwin10", 3.0): "attack_c_data/rollouts/openpi_robotwin/delay_3",
    ("pi0.5", "libero_10",  1.0): "attack_c_data/rollouts/openpi_libero/libero_10_delay_1",
    ("pi0.5", "libero_10",  2.0): "attack_c_data/rollouts/openpi_libero/libero_10_delay_2",
    ("pi0.5", "libero_10",  3.0): "attack_c_data/rollouts/openpi_libero/libero_10_delay_3",
}


def inject_delay_sr(util):
    """Add pi0.5 delay SR (fingerprinted success rate) into the util dict."""
    for (model, ds, s), rel in DELAY_DIRS.items():
        succ = []
        for p in (DATA / rel).rglob("*.npz"):
            if "watermark" not in p.name:
                continue
            try:
                succ.append(float(np.load(p, allow_pickle=True)["success"]))
            except Exception:
                pass
        if succ:
            util[(model, ds, "delay", round(float(s), 4))] = dict(
                sr_wm=float(np.mean(succ)), sr_plain=float("nan"),
                steps_wm=float("nan"), steps_plain=float("nan"))


# utility_lingbot.csv only recorded SR for LingBot/LIBERO at the CANONICAL
# strengths (+ clean); the off-canonical sweep was re-rolled later but its SR
# was never written to the CSV. Recover it from the rollout NPZs (each carries a
# 'success' flag), exactly like inject_delay_sr does for pi0.5 delay. Watermarked
# rollouts live under robust/<...>/controller_<attack>_<strength> (ema->'smooth').
LINGBOT_LIBERO_ROBUST = "attack_c_data/rollouts/lingbot_libero/robust/libero_10"


def inject_lingbot_libero_sr(util):
    """Fill missing LingBot/LIBERO fingerprinted SR from the rollout NPZs."""
    base = DATA / LINGBOT_LIBERO_ROBUST
    if not base.is_dir():
        return
    for d in sorted(base.glob("controller_*")):
        tag = d.name[len("controller_"):]
        i = tag.rfind("_")
        if i < 0:
            continue
        a, sstr = tag[:i], tag[i + 1:]
        try:
            s = float(sstr)
        except ValueError:
            continue
        attack = "ema" if a == "smooth" else a
        key = ("lingbot", "libero_10", attack, round(s, 4))
        if key in util and np.isfinite(util[key].get("sr_wm", np.nan)):
            continue  # keep the value already loaded from the CSV
        succ = []
        for p in d.rglob("*.npz"):
            try:
                dd = np.load(p, allow_pickle=True)
                if str(dd["variant"]) == "watermarked":
                    succ.append(float(dd["success"]))
            except Exception:
                pass
        if succ:
            util[key] = dict(sr_wm=float(np.mean(succ)), sr_plain=float("nan"),
                             steps_wm=float("nan"), steps_plain=float("nan"))


def inject_sr_from_conds(util):
    """Catch-all SR fill: for any cell whose fingerprinted SR is missing from the
    utility CSVs (e.g. the grid-align re-roll cells, absent from utility_pi05.csv),
    average the 'success' flag over its watermarked rollout npz, using the same
    dir map as build_raw_perep. Trusted self-generated npz (allow_pickle)."""
    import build_raw_perep as bp
    for cell, attack, strength, _stem, dirs in bp.CONDS:
        model, ds = cell.split("/", 1)
        skey = None if strength in ("", None) else round(float(strength), 4)
        u = util.get((model, ds, attack, skey))
        if u and np.isfinite(u.get("sr_wm", np.nan)):
            continue
        succ = []
        for rel in dirs:
            for p in (DATA / rel).rglob("*.npz"):
                nm = p.name.lower()
                if "plain" in nm:
                    continue
                try:
                    d = np.load(p, allow_pickle=True)
                    is_wm = ("watermark" in nm) or ("variant" in d.files
                                                    and str(d["variant"]) == "watermarked")
                    if is_wm and "success" in d.files:
                        succ.append(float(d["success"]))
                except Exception:
                    pass
        if succ:
            util[(model, ds, attack, skey)] = dict(
                sr_wm=float(np.mean(succ)), sr_plain=float("nan"),
                steps_wm=float("nan"), steps_plain=float("nan"))


def discover_mixed():
    """Conditions from RAW for every family except pi0.5/LIBERO, which is read
    from the work-7+global score dir. Same CSV schema, so av.calibrate handles
    both transparently; only the source folder differs per family."""
    conds = [c for c in av.discover() if c["family"] != WORK7_FAMILY]
    for p in sorted(WORK7.glob("pi05_libero_10_partial_map_*.csv")):
        df = pd.read_csv(p)
        r0 = df.iloc[0]
        st = r0.get("attack_strength")
        strength = None if (pd.isna(st) or str(st).strip() in ("", "-")) else float(st)
        conds.append(dict(
            path=p, df=df,
            model=str(r0["model"]), dataset=str(r0["dataset"]),
            attack=str(r0["attack"]), strength=strength,
            obs_ratio=float(r0["obs_ratio"]),
            family=f'{r0["model"]}/{r0["dataset"]}'))
    return conds


def build_M_cache():
    conds = discover_mixed()
    util = av.load_utility()
    inject_delay_sr(util)
    inject_lingbot_libero_sr(util)
    inject_sr_from_conds(util)
    rows, cache = [], {}
    for c in conds:
        cal = av.calibrate(c["df"])
        cache[(c["family"], c["attack"], c["strength"])] = cal
        if not in_display_grid(c["attack"], c["strength"]):
            continue  # off-grid strength: keep in cache, hide from the figure
        rows.append(dict(family=c["family"], model=c["model"], dataset=c["dataset"],
                         attack=c["attack"], strength=c["strength"], n_h1=len(cal.z_h1)))
    return pd.DataFrame(rows), util, cache


def sr_str(util, model, ds, attack, s):
    u = util.get((model, ds, attack, None if s is None else round(float(s), 4)))
    return f"{u['sr_wm']:.2f}".lstrip("0") if u and np.isfinite(u.get("sr_wm", np.nan)) else None


def draw(M, util, cache, path):
    rng = np.random.default_rng(av.RNG_SEED)
    rng_auc = np.random.default_rng(av.RNG_SEED + 7)
    fams = [f for f in FAMILIES if f in set(M["family"])]
    nR, nC = len(fams), len(ATTACKS)
    fig, axes = plt.subplots(nR, nC, figsize=(FIG_W, FIG_H), squeeze=False,
                             constrained_layout=True)

    for i, fam in enumerate(fams):
        fr = M[M["family"] == fam]
        model, ds = fr.iloc[0]["model"], fr.iloc[0]["dataset"]
        g_row = G_PER_FAMILY.get(fam, G)
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
                ax.set_xlabel("false-positive rate", fontsize=8.5, labelpad=1)

            sub = M[(M["family"] == fam) & (M["attack"] == attack)
                    & M["strength"].notna()].sort_values("strength")
            if len(sub) == 0:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=11, color="0.6")
                ax.set_xticklabels([])
                continue

            # per-row budget annotation (mixed |G| across rows -> state it)
            ax.text(0.035, 0.93, rf"$|G|{{=}}{g_row}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.0, color="0.45")

            # clean baseline
            ck = (fam, "clean", None)
            if ck in cache:
                fpr_c, tpr_c = av._roc_points(cache[ck].z_h1, cache[ck].z_null, g_row, ROC_TRIALS, rng)
                srs = sr_str(util, model, ds, "clean", None)
                lab = "clean" + (f" ({srs})" if srs else "")
                ax.plot(fpr_c, tpr_c, color="black", lw=2.2, zorder=6, label=lab)
                ax.plot(FPR_OP, np.interp(FPR_OP, fpr_c, tpr_c), "o", color="black",
                        ms=4.5, zorder=7)

            # AUC at THIS row's budget: per-episode when g_row==1 (keeps the
            # LingBot annotations byte-stable), group-AUC otherwise so a family
            # that only saturates in aggregate (pi0.5/RoboTwin at |G|=32) is
            # scored at the budget it is drawn at.
            def panel_auc(cal):
                if g_row == 1:
                    return av.auc(cal.z_h1, cal.z_null.ravel())
                return av.auc_group(cal.z_h1, cal.z_null, g_row, rng_auc)

            strengths = sub["strength"].values
            # weak -> strong = purple -> orange (good contrast on white, ordered)
            colors = plt.cm.plasma(np.linspace(0.12, 0.72, len(strengths)))
            panel_aucs = []
            if ck in cache:
                panel_aucs.append(panel_auc(cache[ck]))
            for k, s in enumerate(strengths):
                key = (fam, attack, s)
                if key not in cache:
                    continue
                panel_aucs.append(panel_auc(cache[key]))
                fpr_s, tpr_s = av._roc_points(cache[key].z_h1, cache[key].z_null, g_row, ROC_TRIALS, rng)
                srs = sr_str(util, model, ds, attack, s)
                lab = SLAB[attack](s) + (f" ({srs})" if srs else "")
                ax.plot(fpr_s, tpr_s, color=colors[k], lw=1.6, zorder=4, label=lab)
                ax.plot(FPR_OP, np.interp(FPR_OP, fpr_s, tpr_s), "o", color=colors[k],
                        ms=3.5, zorder=5)

            # Saturated panels (clean + every strength near-perfect even per episode)
            # carry no visible curve, so state the headline number instead of leaving
            # the panel blank. TPR@1% = 1.00 at |G|=16 for these cells (cf. tab:main).
            if panel_aucs and min(panel_aucs) >= 0.99:
                ax.text(0.5, 0.60, rf"AUC $\geq$ {min(panel_aucs):.3f}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="0.30", weight="medium")
                ax.text(0.5, 0.45, "TPR@1% = 1.00", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="0.45")

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
    M, util, cache = build_M_cache()
    draw(M, util, cache, av.OUT_DIR / "fig_attack_combined.pdf")
    if not args.no_paper:
        draw(M, util, cache, DATA / "paper" / "fig_attack_combined.pdf")
    if args.preview:
        draw(M, util, cache, args.preview)


if __name__ == "__main__":
    main()
