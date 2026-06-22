#!/usr/bin/env python3
"""Distillation-survival figure (two panels), data-driven from preserved artifacts.

(a) pi0.5/LIBERO-10 entropy sweep: detection AUC (survival, left axis) and task
    success SR (utility, right axis) versus the DC key cardinality N_KEYS, all marks
    constant (non-zero-mean) and injected at the seed site. The deployed key
    (high-entropy AND zero-mean) is plotted at its matched cardinality (~160) to show
    that PERSISTENCE is the switch: same entropy, AUC drops 0.75 -> 0.50 the moment the
    mark loses its non-zero time average. Entropy is the secondary modulator.
(b) LingBot-VA (world-action): behavior-cloning students trained on
    10-task teacher relabels inherit an observation-bucketed DC seed offset, with
    modest attenuation as the key cardinality grows over the same N={20,80,160}
    grid used for pi0.5.

Reads: distill/VERDICT_latentdc_n{20,80,160}.txt and VERDICT.txt (AUCs),
       distill/eval/.../*_plain.npz (SR), and LingBot DC-detection NPZs.
Writes results/fig_distillation.pdf, paper/fig_distillation.pdf, and a PNG preview.
Run: python3 make_fig_distillation.py
"""
from __future__ import annotations
import re, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

D = Path("/workspace/vla/distill")
LB_OUT = Path("/workspace/vla/lingbot-va/outputs/pathb_det")


def auc_from(verdict, pat):
    t = (D / verdict).read_text()
    m = re.search(pat + r"\s*=\s*([0-9.]+)", t)
    if not m:
        raise SystemExit(f"AUC not found in {verdict} via /{pat}/")
    return float(m.group(1))


def sr_of(tag):
    fs = sorted(glob.glob(str(D / f"eval/libero_goal_{tag}/rollouts/task_rollout/*_plain.npz")))
    if not fs:
        return None
    return float(np.mean([bool(np.load(f)["success"]) for f in fs]))


# ---- panel (a) data: DC entropy sweep (all non-zero-mean, seed site) ----
N = [20, 80, 160]
auc_dc = [
    auc_from("VERDICT_latentdc_n20.txt",  r"AUC \(latentdc-N20 vs clean\)"),
    auc_from("VERDICT_latentdc_n80.txt",  r"AUC \(latentdc-N80 vs clean\)"),
    auc_from("VERDICT_latentdc_n160.txt", r"AUC \(latentdc-N160 vs clean\)"),
]
sr_dc = [sr_of("latentdc_n20_student"), sr_of("latentdc_n80_student"), sr_of("latentdc_n160_student")]
auc_deployed = auc_from("VERDICT.txt", r"Cross-student AUC \(obstied-plain vs clean-plain, true key\)")  # zero-mean, ~160

# ---- panel (b) data: LingBot DC distillation ----
sys.path.insert(0, str(D / "lingbot"))
import score_pathb  # noqa: E402
import score_pathb_gaussian  # noqa: E402


def lingbot_summary(name, n_keys):
    proj = (0, 1, 2)
    zw, rw, sw = score_pathb.run(LB_OUT / name, 42, 32, n_keys, 0.08, proj)
    zc, rc, sc = score_pathb.run(LB_OUT / "clean10x10", 42, 32, n_keys, 0.08, proj)
    auc = sum(x > y for x in zw for y in zc) / max(len(zw) * len(zc), 1)
    return {
        "auc": float(auc),
        "retention": float(rw.mean()),
        "sr": float(sw.mean()),
        "n_wm": int(len(zw)),
        "clean_retention": float(rc.mean()),
    }


lb20 = lingbot_summary("n20_10x10", 20)
lb80 = lingbot_summary("n80_10x10", 80)
lb160 = lingbot_summary("n160_10x10", 160)
proj = (0, 1, 2)
lb_zmean_z, lb_zmean_sr = score_pathb_gaussian.run(LB_OUT / "zmean160_10x10", 42, 32, 160, 0.08, proj)
lb_clean_z, lb_clean_sr = score_pathb_gaussian.run(LB_OUT / "clean10x10", 42, 32, 160, 0.08, proj)
lb_zmean_auc = sum(x > y for x in lb_zmean_z for y in lb_clean_z) / max(len(lb_zmean_z) * len(lb_clean_z), 1)
lb_zmean_summary = {
    "auc": float(lb_zmean_auc),
    "sr": float(lb_zmean_sr.mean()),
    "n_wm": int(len(lb_zmean_z)),
    "z_mean": float(lb_zmean_z.mean()),
    "clean_z_mean": float(lb_clean_z.mean()),
}

print("panel(a) N=", N, "AUC_dc=", [round(a, 3) for a in auc_dc], "SR_dc=", sr_dc, "deployed AUC=", round(auc_deployed, 3))
print("panel(b) LingBot N20=", lb20, "N80=", lb80, "N160=", lb160, "zmean160=", lb_zmean_summary)

# ===================== figure =====================
plt.rcParams.update({
    "font.size": 6.2,
    "axes.titlesize": 6.8,
    "axes.labelsize": 6.4,
    "legend.fontsize": 5.4,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "lines.markeredgewidth": 0.0,
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.45, 1.35), sharey=True)

# ---------- (a) entropy sweep ----------
C_SURV, C_UTIL, C_DEP = "#1f4e9c", "#d9820a", "#c0202a"
axA.set_xscale("log")
axA.xaxis.set_minor_formatter(mticker.NullFormatter())
axA.plot(N, auc_dc, "o-", color=C_SURV, lw=1.2, ms=3.5, label="AUC (DC)", zorder=4)
# deployed zero-mean point at matched cardinality
axA.plot([160], [auc_deployed], "X", color=C_DEP, ms=6.2, mew=0, label="AUC (deployed)", zorder=5)
axA.plot(N, sr_dc, "^--", color=C_UTIL, lw=1.0, ms=3.8, label="SR (DC)", zorder=3)
# the persistence flip: vertical gap at x=160
axA.annotate("", xy=(160, auc_dc[2] - 0.012), xytext=(160, auc_deployed + 0.012),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.7, shrinkA=0, shrinkB=0))
axA.text(171, (auc_dc[2] + auc_deployed) / 2, "zero-mean\nfalls to chance", fontsize=4.8,
         color="0.25", va="center", ha="left")
axA.axhline(0.5, color="0.6", ls=":", lw=1.0)
axA.text(209, 0.512, "chance", fontsize=4.8, color="0.5", va="bottom", ha="right")
axA.set_ylim(0.42, 1.0)
axA.set_xlim(16, 220)
axA.set_xticks(N); axA.set_xticklabels(["20", "80", "160"])
axA.set_xlabel("$N_{\\mathrm{KEYS}}$")
axA.set_ylabel("AUC / SR")
axA.set_title("(a) $\\pi_{0.5}$")
axA.legend(loc="lower left", framealpha=0.90, handlelength=1.2, borderpad=0.25,
           labelspacing=0.25, borderaxespad=0.25)

# ---------- (b) LingBot DC distillation ----------
N_lb = [20, 80, 160]
auc_lb = [lb20["auc"], lb80["auc"], lb160["auc"]]
sr_lb = [lb20["sr"], lb80["sr"], lb160["sr"]]
axB.set_xscale("log")
axB.xaxis.set_minor_formatter(mticker.NullFormatter())
axB.plot(N_lb, auc_lb, "o-", color=C_SURV, lw=1.2, ms=3.5, label="AUC (DC)", zorder=4)
axB.plot(N_lb, sr_lb, "^--", color=C_UTIL, lw=1.0, ms=3.8, label="SR (DC)", zorder=3)
axB.plot([160], [lb_zmean_auc], "X", color=C_DEP, ms=6.2, mew=0, zorder=5)
axB.axhline(0.5, color="0.6", ls=":", lw=1.0)
axB.text(209, 0.512, "chance", fontsize=4.8, color="0.5", va="bottom", ha="right")
axB.text(116, 0.548, "zero-mean", fontsize=4.8, color="0.25", va="bottom", ha="left")
for x_, auc in zip(N_lb, auc_lb):
    axB.text(x_, auc + 0.014, f"{auc:.2f}", ha="center", va="bottom",
             fontsize=5.2, color=C_SURV, fontweight="bold")
axB.set_ylim(0.42, 1.0)
axB.set_xlim(16, 220)
axB.set_xticks(N_lb); axB.set_xticklabels(["20", "80", "160"])
axB.set_xlabel("$N_{\\mathrm{KEYS}}$")
axB.set_title("(b) LingBot-VA")
for x_, sr in zip(N_lb, sr_lb):
    axB.text(x_, sr + 0.014, f"{sr:.2f}", ha="center", va="bottom",
             fontsize=5.2, color=C_UTIL, fontweight="bold")

for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)

fig.tight_layout(w_pad=0.7, pad=0.25)
for out in ["/workspace/vla/results/fig_distillation.pdf", "/workspace/vla/paper/fig_distillation.pdf",
            "/workspace/vla/results/fig_distillation_preview.png"]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
