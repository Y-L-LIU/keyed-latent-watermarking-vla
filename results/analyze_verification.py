#!/usr/bin/env python3
"""
Verification + robustness + utility results from raw per-episode WMF scores.

Inputs (../attack_c_data/per_episode_scores/):
  <model>_<dataset>_partial_map_<attack>[_<strength>].csv
      one row per rollout episode; columns:
      episode_id, variant, model, dataset, obs, obs_ratio, recovery, attack,
      attack_strength, m, s_true, s_false_1..32
  utility_pi05.csv     (wide:  sr_plain/sr_wm, mean_steps_plain/wm, ... per condition)
  utility_lingbot.csv  (long:  one row per variant; sr, mean_steps)

We do ALL calibration/statistics here -- the drop is raw `s` only.
  Z_e(k) = (s_e(k)-mu^-_e)/(sigma^-_e+eps)   [eq:zscore]   H1 = s_true on watermarked rows
  T_G(k) = sum_{e in G} Z_e(k)               [eq:aggregation]   H0 = leave-one-out false keys
  TPR@FPR: threshold T_G at the (1-FPR) quantile of the H0 group distribution.
  |G| swept by resampling row groups -- nothing is re-run.

Discovery is glob-based and reads metadata from the CSV columns, so it does NOT
depend on manifest.json (which lags the data drop).

Outputs (this dir):
  verification_metrics.csv   per-condition AUC / TPR@1% at |G|=1 and |G|=16
  tab_main.tex               headline table: utility + clean + robust-avg detection
  tab_utility.tex            full SR / steps table (plain vs fingerprinted), all conditions
  fig_tpr_vs_G.pdf           detection power vs query budget |G| (clean + canonical attacks)
  fig_robustness.pdf         TPR@1% vs task-success drop (the attacker's tradeoff)
  fig_attack_strength.pdf    TPR@1% vs attack strength, grid of family x attack

Not yet in the drop (left as gaps): delay attack, lingbot clip / 3rd jitter point,
Exp-A full-obs/ODE cells, episode top-up to >=200-300.
"""
import glob
import math
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent          # vla/results
SCORE_DIR = HERE.parent / "attack_c_data" / "per_episode_scores"
OUT_DIR = HERE
EPS = 1e-8
RNG_SEED = 0
N_FALSE = 32
FALSE_COLS = [f"s_false_{i}" for i in range(1, N_FALSE + 1)]

G_GRID = [1, 2, 4, 8, 16, 32, 64]  # budgets swept in the |G| figure (<=n_wm; cells re-rolled to ~130)
G_TABLE = 16                       # fixed budget for table TPR@1%
FPR_MAIN = 0.01
N_TRIALS = 20000                   # MC groups for point estimates
N_BOOT = 200                       # episode-pool resamples (CI band, figure only)

# canonical strength per attack -> used for the robust-average column & |G| figure
CANONICAL = {"clip": 1.0, "ema": 0.5, "jitter": 0.01}
DS_PRETTY = {"libero_10": "LIBERO-10", "robotwin10": "RoboTwin"}
MODEL_PRETTY = {"lingbot": "LingBot"}  # display name; data key stays "lingbot"
ATTACK_ORDER = ["clean", "clip", "ema", "jitter", "delay"]


# --------------------------------------------------------------------------- #
# Calibration & stats
# --------------------------------------------------------------------------- #
Calibrated = namedtuple("Calibrated", [
    "z_h1", "z_null", "z_h0_plain", "task_h1", "task_null_wm", "task_h0_plain"])


def _extract_task(eid: str) -> str:
    """Parse a task identifier out of the per-episode CSV episode_id string.
    Handles three schemas: lingbot libero/robotwin (taskNN_epNN), pi05 libero
    (task_NNN_episode_NNN), pi05 robotwin (episode_NNN, single task)."""
    last = eid.split("|")[-1]
    m = re.match(r"(.+?_task\d+)_ep\d+", last)
    if m:
        return m.group(1)
    m = re.match(r"(.+?task_\d+)_episode_\d+", last)
    if m:
        return m.group(1)
    if re.match(r"episode_\d+", last):
        return "single_task"
    return last


def calibrate(df):
    """Return a Calibrated namedtuple with:
      z_h1         : true-key Z on watermarked rows (H1)
      z_null       : (n_rows, 32) leave-one-out Z of false keys (calibrated H0)
      z_h0_plain   : true-key Z on plain (no-injection) rows (literal H0)
      task_h1/null_wm/h0_plain : per-row task identifiers for cross/same-task
                                 aggregation; task_null_wm is the per-row task
                                 id on watermarked rows used by tpr_point's
                                 z_null indexing (watermarked rows only, to
                                 mirror the H1 sampling pool)."""
    s_false = df[FALSE_COLS].to_numpy(float)
    s_true = df["s_true"].to_numpy(float)
    is_wm = df["variant"].to_numpy() == "watermarked"
    tasks = df["episode_id"].map(_extract_task).to_numpy()

    mu = s_false.mean(axis=1)
    sd = s_false.std(axis=1, ddof=1)
    z_true = (s_true - mu) / (sd + EPS)
    z_h1 = z_true[is_wm]
    z_h0_plain = z_true[~is_wm]

    csum = s_false.sum(axis=1, keepdims=True)
    loo_mu = (csum - s_false) / (N_FALSE - 1)
    sq = (s_false ** 2).sum(axis=1, keepdims=True)
    loo_var = ((sq - s_false ** 2) - (N_FALSE - 1) * loo_mu ** 2) / (N_FALSE - 2)
    z_null_all = (s_false - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + EPS)
    # The original z_null kept ALL rows. tpr_point indexes it freely, so the
    # group-distribution null mixes wm and plain calibration rows. Keep that
    # behaviour for backward compat (existing fig_budget / verification table).
    z_null = z_null_all
    return Calibrated(
        z_h1=z_h1,
        z_null=z_null,
        z_h0_plain=z_h0_plain,
        task_h1=tasks[is_wm],
        task_null_wm=tasks,           # row-wise tasks for the full pool
        task_h0_plain=tasks[~is_wm],
    )


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([neg, pos])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[len(neg):].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def tpr_point(z_h1, z_null, G, fpr, rng, n_trials=N_TRIALS):
    n_wm, n_rows = len(z_h1), z_null.shape[0]
    ep = rng.integers(0, n_rows, size=(n_trials, G))
    key = rng.integers(0, N_FALSE, size=(n_trials, G))
    t0 = z_null[ep, key].sum(axis=1)
    t1 = z_h1[rng.integers(0, n_wm, size=(n_trials, G))].sum(axis=1)
    return float((t1 >= np.quantile(t0, 1 - fpr)).mean())


def auc_group(z_h1, z_null, G, rng, n_trials=N_TRIALS):
    """Threshold-free separability at group budget G: AUC of the H1 group-sum
    distribution vs the calibrated-null group-sum distribution. Same sampling as
    tpr_point, so it is the |G|-aggregated companion to the per-episode AUC."""
    n_wm, n_rows = len(z_h1), z_null.shape[0]
    t1 = z_h1[rng.integers(0, n_wm, size=(n_trials, G))].sum(axis=1)
    ep = rng.integers(0, n_rows, size=(n_trials, G))
    key = rng.integers(0, N_FALSE, size=(n_trials, G))
    t0 = z_null[ep, key].sum(axis=1)
    return auc(t1, t0)


def tpr_ci(z_h1, z_null, G, fpr, rng):
    n_wm, n_rows = len(z_h1), z_null.shape[0]
    vals = []
    for _ in range(N_BOOT):
        bh = z_h1[rng.integers(0, n_wm, size=n_wm)]
        bn = z_null[rng.integers(0, n_rows, size=n_rows)]
        vals.append(tpr_point(bh, bn, G, fpr, rng, n_trials=N_TRIALS // 4))
    v = np.array(vals)
    return v.mean(), np.quantile(v, 0.05), np.quantile(v, 0.95)


def phi_predict_tpr(z_h1, z_null, G, fpr):
    """Prop-4 Gaussian prediction: TPR_pred = Phi(sqrt(G) * mu / sqrt((s0^2+s1^2)/2)
    - Phi^{-1}(1-fpr)). Uses sample mean/std of the calibrated z scores."""
    mu = float(np.mean(z_h1))
    s1 = float(np.std(z_h1, ddof=1))
    s0 = float(np.std(z_null.ravel(), ddof=1))
    pooled = math.sqrt((s0 ** 2 + s1 ** 2) / 2.0)
    if pooled == 0.0:
        return float("nan"), mu, s0, s1
    d_prime = math.sqrt(G) * mu / pooled
    tpr = float(norm.cdf(d_prime - norm.ppf(1.0 - fpr)))
    return tpr, mu, s0, s1


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def discover():
    """Return list of condition dicts from globbed score CSVs."""
    conds = []
    for p in sorted(SCORE_DIR.glob("*_partial_map_*.csv")):
        df = pd.read_csv(p)
        r0 = df.iloc[0]
        st = r0.get("attack_strength")
        strength = None if (pd.isna(st) or str(st).strip() in ("", "-")) else float(st)
        conds.append(dict(
            path=p, df=df,
            model=str(r0["model"]), dataset=str(r0["dataset"]),
            attack=str(r0["attack"]), strength=strength,
            obs_ratio=float(r0["obs_ratio"]),
            family=f'{r0["model"]}/{r0["dataset"]}',
        ))
    return conds


def load_utility():
    """Uniform dict: (model, dataset, attack, strength|None) ->
    {sr_plain, sr_wm, steps_plain, steps_wm}."""
    util = {}

    def skey(attack, s):
        if attack == "clean" or s is None or (isinstance(s, float) and math.isnan(s)):
            return None
        return round(float(s), 4)

    fp = SCORE_DIR / "utility_pi05.csv"
    if fp.exists():
        for _, r in pd.read_csv(fp).iterrows():
            s = None if str(r["attack_strength"]).strip() in ("", "-", "nan") \
                else float(r["attack_strength"])
            util[(r["model"], r["dataset"], r["attack"], skey(r["attack"], s))] = dict(
                sr_plain=float(r["sr_plain"]), sr_wm=float(r["sr_wm"]),
                steps_plain=float(r["mean_steps_plain"]), steps_wm=float(r["mean_steps_wm"]))

    fl = SCORE_DIR / "utility_lingbot.csv"
    if fl.exists():
        df = pd.read_csv(fl)
        for (m, d, a, st), g in df.groupby(
                ["model", "dataset", "attack", "attack_strength"], dropna=False):
            s = None if (pd.isna(st) or str(st).strip() in ("", "-")) else float(st)
            row = {v: gg.iloc[0] for v, gg in g.groupby("variant")}
            if "watermarked" in row:
                pl = row.get("plain")
                util[(m, d, a, skey(a, s))] = dict(
                    sr_plain=float(pl["sr"]) if pl is not None else float("nan"),
                    sr_wm=float(row["watermarked"]["sr"]),
                    steps_plain=float(pl["mean_steps"]) if pl is not None else float("nan"),
                    steps_wm=float(row["watermarked"]["mean_steps"]))
    return util


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(RNG_SEED)
    rng_auc = np.random.default_rng(RNG_SEED + 1)  # independent stream so adding the
    # group-AUC column does not perturb the existing tpr_point Monte-Carlo draws
    conds = discover()
    util = load_utility()
    if not conds:
        print("No score CSVs found.")
        return

    # ---- per-condition metrics -------------------------------------------- #
    rows = []
    cache = {}  # (family, attack, strength) -> Calibrated namedtuple
    for c in conds:
        cal = calibrate(c["df"])
        cache[(c["family"], c["attack"], c["strength"])] = cal
        rows.append(dict(
            family=c["family"], model=c["model"], dataset=c["dataset"],
            attack=c["attack"], strength=c["strength"], obs_ratio=c["obs_ratio"],
            n_h1=len(cal.z_h1),
            auc=auc(cal.z_h1, cal.z_null.ravel()),
            tpr1_g1=tpr_point(cal.z_h1, cal.z_null, 1, FPR_MAIN, rng),
            tpr1_gT=tpr_point(cal.z_h1, cal.z_null, G_TABLE, FPR_MAIN, rng),
            auc_gT=auc_group(cal.z_h1, cal.z_null, G_TABLE, rng_auc),
        ))
    M = pd.DataFrame(rows)
    M.to_csv(OUT_DIR / "verification_metrics.csv", index=False)
    print(M.to_string(index=False))

    write_main_table(M, util, OUT_DIR / "tab_main.tex")
    write_utility_table(util, OUT_DIR / "tab_utility.tex")
    val_rows = fig_budget(cache, rng, OUT_DIR / "fig_tpr_vs_G.pdf")
    write_rate_validation(val_rows, OUT_DIR / "rate_validation.csv")
    fig_rate_calibration(val_rows, OUT_DIR / "fig_rate_calibration.pdf")
    fig_neg_control_h0(cache, rng, OUT_DIR / "fig_neg_control_h0.pdf")
    write_neg_control_h0_csv(cache, rng, OUT_DIR / "neg_control_h0.csv")
    agg_rows = fig_aggregation_mode(cache, rng, OUT_DIR / "fig_aggregation_mode.pdf")
    write_aggregation_mode_csv(agg_rows, OUT_DIR / "aggregation_mode.csv")
    fig_robustness(M, util, OUT_DIR / "fig_robustness.pdf")
    fig_attack_strength(M, OUT_DIR / "fig_attack_strength.pdf")
    # NOTE: paper/fig_attack_combined.pdf is now produced by make_fig_attack_combined.py
    # (polished per-episode ROC grid, raw scorer, incl. pi0.5 delay). Keep a reference
    # copy in OUT_DIR only so this pipeline never clobbers the paper figure.
    fig_attack_combined(M, util, cache, rng, OUT_DIR / "fig_attack_combined_legacy.pdf")
    print(f"\nWrote tables + figures to {OUT_DIR}")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _families_in_order(M):
    seen = []
    for _, r in M.iterrows():
        if r["family"] not in seen:
            seen.append(r["family"])
    return seen


def write_main_table(M, util, path):
    """Headline: utility (clean) + clean detection + robust-avg detection, |G| fixed."""
    L = [
        r"% auto-generated -- do not hand-edit",
        r"\begin{table*}[t]\centering",
        r"\caption{Overall verification results under partial observation (MAP recovery). "
        r"Utility is task success rate (SR) and mean episode steps, plain vs.\ fingerprinted "
        r"(no attack). "
        rf"AUC is per-episode ($|G|{{=}}1$); AUC$_{{{G_TABLE}}}$ and TPR at FPR$=1\%$ "
        rf"use the group budget $|G|{{=}}{G_TABLE}$. "
        r"Detection columns cover the clean policy and the average over the removal "
        r"attacks at canonical strength. "
        r"The \texttt{delay} attack and the weight-level edits are reported per condition "
        r"in \Cref{tab:robust-detection}; \texttt{delay} is kept out of this average so "
        r"the robust column stays comparable across cells.}",
        r"\label{tab:main}\small",
        r"\begin{tabular}{ll cc c ccc ccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Utility (clean)} "
        r"& \multicolumn{3}{c}{Clean} & \multicolumn{3}{c}{Robust (atk.\ avg)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        rf"Model & Dataset & SR$_{{\mathrm{{pl}}{{\to}}\mathrm{{fp}}}}$ & $\Delta$SR "
        rf"& Steps$_{{\mathrm{{pl}}{{\to}}\mathrm{{fp}}}}$ "
        rf"& AUC & AUC$_{{{G_TABLE}}}$ & TPR@1\% "
        rf"& AUC & AUC$_{{{G_TABLE}}}$ & TPR@1\% \\",
        r"\midrule",
    ]
    for fam in _families_in_order(M):
        sub = M[M["family"] == fam]
        r0 = sub.iloc[0]
        model, ds = r0["model"], r0["dataset"]
        # utility from clean
        u = util.get((model, ds, "clean", None))
        if u:
            sr = f"{u['sr_plain']:.2f}$\\to${u['sr_wm']:.2f}"
            dsr = f"{u['sr_wm'] - u['sr_plain']:+.2f}"
            steps = f"{u['steps_plain']:.0f}$\\to${u['steps_wm']:.0f}"
        else:
            sr = dsr = steps = "--"
        cl = sub[sub["attack"] == "clean"]
        cl_auc = f"{cl['auc'].iloc[0]:.3f}" if len(cl) else "--"
        cl_auc_g = f"{cl['auc_gT'].iloc[0]:.3f}" if len(cl) else "--"
        cl_tpr = f"{cl['tpr1_gT'].iloc[0]:.3f}" if len(cl) else "--"
        # robust avg over canonical attack strengths
        mask = [(r["attack"] in CANONICAL and r["strength"] == CANONICAL[r["attack"]])
                for _, r in sub.iterrows()]
        rb = sub[mask]
        rb_auc = f"{rb['auc'].mean():.3f}" if len(rb) else "--"
        rb_auc_g = f"{rb['auc_gT'].mean():.3f}" if len(rb) else "--"
        rb_tpr = f"{rb['tpr1_gT'].mean():.3f}" if len(rb) else "--"
        L.append(f"{MODEL_PRETTY.get(model, model)} & {DS_PRETTY.get(ds, ds)} & {sr} & {dsr} & {steps} "
                 f"& {cl_auc} & {cl_auc_g} & {cl_tpr} & {rb_auc} & {rb_auc_g} & {rb_tpr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(L) + "\n")


def write_utility_table(util, path):
    """Full SR + steps, plain vs fingerprinted, every delivered condition."""
    L = [
        r"% auto-generated -- do not hand-edit",
        r"\begin{table}[t]\centering",
        r"\caption{Task utility (success rate and mean steps) of the plain vs.\ "
        r"fingerprinted policy across attacks and strengths.}",
        r"\label{tab:utility}\small",
        r"\begin{tabular}{lll cc cc}",
        r"\toprule",
        r"Model & Dataset & Attack($s$) & SR$_{\mathrm{pl}}$ & SR$_{\mathrm{fp}}$ "
        r"& Steps$_{\mathrm{pl}}$ & Steps$_{\mathrm{fp}}$ \\",
        r"\midrule",
    ]
    def _fmt(x, prec):
        import math
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "--"
        return f"{x:.{prec}f}"

    for (m, d, a, s), u in sorted(util.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        atk = a if s is None else f"{a}({s:g})"
        L.append(f"{MODEL_PRETTY.get(m, m)} & {DS_PRETTY.get(d, d)} & {atk} & {_fmt(u['sr_plain'],2)} & "
                 f"{_fmt(u['sr_wm'],2)} & {_fmt(u['steps_plain'],0)} & "
                 f"{_fmt(u['steps_wm'],0)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_budget(cache, rng, path):
    """TPR@1% vs |G|; panel per family; lines = clean + canonical attacks.
    Returns a list of (family, attack, strength, G, measured, lo, hi, predicted,
    gap, mu, s0, s1) tuples for fig_rate_calibration + write_rate_validation."""
    fams = sorted({k[0] for k in cache})
    fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 3.4), squeeze=False)
    validation_rows = []
    for ax, fam in zip(axes[0], fams):
        wanted = [("clean", None)] + [(a, s) for a, s in CANONICAL.items()]
        for attack, strength in wanted:
            key = (fam, attack, strength)
            if key not in cache:
                continue
            cal = cache[key]
            z_h1, z_null = cal.z_h1, cal.z_null
            ys, lo, hi = [], [], []
            for G in G_GRID:
                m, l, h = tpr_ci(z_h1, z_null, G, FPR_MAIN, rng)
                p, mu, s0, s1 = phi_predict_tpr(z_h1, z_null, G, FPR_MAIN)
                ys.append(m); lo.append(l); hi.append(h)
                validation_rows.append(dict(
                    family=fam, attack=attack, strength=strength, G=G,
                    measured=m, ci_lo=l, ci_hi=h, predicted=p, gap=p - m,
                    mu=mu, sigma0=s0, sigma1=s1, n_h1=len(z_h1)))
            lbl = attack if strength is None else f"{attack} {strength:g}"
            line, = ax.plot(G_GRID, ys, marker="o", label=lbl)
            ax.fill_between(G_GRID, lo, hi, alpha=0.15, color=line.get_color())
        ax.set_title(fam); ax.set_xlabel(r"query budget $|G|$")
        ax.set_ylabel(r"TPR @ FPR$=1\%$")
        ax.set_xscale("log", base=2); ax.set_xticks(G_GRID)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return validation_rows


def fig_rate_calibration(rows, path):
    """Predicted-vs-measured TPR scatter, one dot per (family, attack, |G|).
    Diagonal y=x; color by family; marker shape by |G|. Highlights how Prop 4
    tracks experiment."""
    df = pd.DataFrame(rows)
    fams = sorted(df["family"].unique())
    g_vals = sorted(df["G"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    markers = ["o", "s", "^", "D", "v", "P"][:len(g_vals)]
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    # diagonal + +/-0.05 envelope
    ax.plot([0, 1], [0, 1], "k-", lw=1, alpha=0.6, zorder=1)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="gray", alpha=0.12,
                    zorder=0, label=r"$\pm 0.05$ TPR")
    for fc, fam in zip(colors, fams):
        sub_f = df[df["family"] == fam]
        for mk, G in zip(markers, g_vals):
            sub = sub_f[sub_f["G"] == G]
            if not len(sub):
                continue
            ax.scatter(sub["measured"], sub["predicted"], color=fc, marker=mk,
                       s=42, alpha=0.85, edgecolor="white", linewidth=0.4,
                       zorder=3)
    # one legend handle per family + one per |G|
    from matplotlib.lines import Line2D
    fam_handles = [Line2D([0], [0], marker="o", color="w", label=fam.replace("/", "/"),
                          markerfacecolor=c, markersize=8)
                   for c, fam in zip(colors, fams)]
    g_handles = [Line2D([0], [0], marker=mk, color="gray", label=f"|G|={G}",
                        linestyle="none", markersize=7)
                 for mk, G in zip(markers, g_vals)]
    leg1 = ax.legend(handles=fam_handles, loc="lower right", fontsize=8,
                     title="family", title_fontsize=8, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=g_handles, loc="upper left", fontsize=8, ncol=2,
              frameon=True)
    ax.set_xlabel("measured TPR (bootstrap)")
    ax.set_ylabel(r"predicted TPR ($\Phi$ from \Cref{prop:rate})".replace(
        r"\Cref{prop:rate}", "Prop. 4.2"))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    n_in = int(((df["predicted"] >= df["ci_lo"]) & (df["predicted"] <= df["ci_hi"])).sum() |
               (df["gap"].abs() < 0.05).sum())  # rough display only
    in_band = (df["predicted"] >= df["ci_lo"]) & (df["predicted"] <= df["ci_hi"])
    small_gap = df["gap"].abs() < 0.05
    n_pass = int((in_band | small_gap).sum())
    ax.set_title(f"Prop. 4.2 calibration: {n_pass}/{len(df)} cells within band or $\\pm$0.05", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_budget_calibration(cache, rng, path):
    """Merged figure: LEFT = square predicted-vs-measured calibration scatter;
    RIGHT = 2x2 grid of TPR@1% vs |G|, one panel per family. Returns the
    validation_rows (same as fig_budget) for write_rate_validation."""
    from matplotlib.lines import Line2D
    fams = sorted({k[0] for k in cache})
    fig = plt.figure(figsize=(13.0, 4.7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.16,
                             top=0.88, bottom=0.16)
    ax_cal = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(2, 2, wspace=0.12, hspace=0.38)
    ax_fams = [fig.add_subplot(inner[r, c]) for r in (0, 1) for c in (0, 1)]

    # --- RIGHT: TPR vs |G| per family (2x2); collect validation rows ---
    validation_rows = []
    line_handles = {}
    # delay-1/2 are shown here for the rate-law view, but kept OUT of CANONICAL so they stay
    # out of tab:main's removal-attack average; the full delay-1/2/3 sweep is in fig_budget_curve.
    fig_attacks = [("clean", None)] + [(a, s) for a, s in CANONICAL.items()] + [("delay", 1.0), ("delay", 2.0)]
    for i, (ax, fam) in enumerate(zip(ax_fams, fams)):
        for attack, strength in fig_attacks:
            key = (fam, attack, strength)
            if key not in cache:
                continue
            cal = cache[key]
            z_h1, z_null = cal.z_h1, cal.z_null
            ys, lo, hi = [], [], []
            for G in G_GRID:
                m, l, h = tpr_ci(z_h1, z_null, G, FPR_MAIN, rng)
                p, mu, s0, s1 = phi_predict_tpr(z_h1, z_null, G, FPR_MAIN)
                ys.append(m); lo.append(l); hi.append(h)
                validation_rows.append(dict(
                    family=fam, attack=attack, strength=strength, G=G,
                    measured=m, ci_lo=l, ci_hi=h, predicted=p, gap=p - m,
                    mu=mu, sigma0=s0, sigma1=s1, n_h1=len(z_h1)))
            lbl = attack if strength is None else f"{attack} {strength:g}"
            line, = ax.plot(G_GRID, ys, marker="o", ms=4, label=lbl)
            ax.fill_between(G_GRID, lo, hi, alpha=0.15, color=line.get_color())
            line_handles.setdefault(lbl, line)
        ax.set_title(FAM_TITLE.get(fam, fam), fontsize=11)
        ax.set_xscale("log", base=2); ax.set_xticks(G_GRID)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
        if i in (0, 2):
            ax.set_ylabel(r"TPR @ FPR$=1\%$", fontsize=10)
        else:
            ax.set_yticklabels([])
        if i in (2, 3):
            ax.tick_params(axis="x", pad=1.0)
    # one shared legend for the attack lines, centered above the 2x2 block
    fig.text(0.74, 0.075, r"query budget $|G|$", ha="center", va="center",
             fontsize=10)
    fig.legend(list(line_handles.values()), list(line_handles.keys()),
               loc="upper center", bbox_to_anchor=(0.74, 1.035),
               ncol=6, fontsize=8.5, frameon=False, handlelength=1.25,
               columnspacing=0.75, handletextpad=0.35)

    # --- LEFT: calibration scatter (square) ---
    df = pd.DataFrame(validation_rows)
    g_vals = sorted(df["G"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    markers = ["o", "s", "^", "D", "v", "P"][:len(g_vals)]
    ax_cal.plot([0, 1], [0, 1], "k-", lw=1, alpha=0.6, zorder=1)
    ax_cal.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="gray", alpha=0.12, zorder=0)
    for fc, fam in zip(colors, fams):
        sub_f = df[df["family"] == fam]
        for mk, G in zip(markers, g_vals):
            sub = sub_f[sub_f["G"] == G]
            if len(sub):
                ax_cal.scatter(sub["measured"], sub["predicted"], color=fc, marker=mk,
                               s=42, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    fam_handles = [Line2D([0], [0], marker="o", color="w", label=FAM_TITLE.get(fam, fam),
                          markerfacecolor=c, markersize=9) for c, fam in zip(colors, fams)]
    g_handles = [Line2D([0], [0], marker=mk, color="gray", label=f"$|G|{{=}}{G}$",
                        linestyle="none", markersize=8) for mk, G in zip(markers, g_vals)]
    leg1 = ax_cal.legend(handles=fam_handles, loc="lower right", fontsize=9,
                         title="family", title_fontsize=9, frameon=True)
    ax_cal.add_artist(leg1)
    ax_cal.legend(handles=g_handles, loc="upper left", fontsize=9, ncol=2, frameon=True)
    in_band = (df["predicted"] >= df["ci_lo"]) & (df["predicted"] <= df["ci_hi"])
    n_pass = int((in_band | (df["gap"].abs() < 0.05)).sum())
    ax_cal.set_xlabel("measured TPR (bootstrap)", fontsize=11)
    ax_cal.set_ylabel("predicted TPR (Prop. 4.2)", fontsize=11)
    ax_cal.set_xlim(-0.02, 1.02); ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_aspect("equal"); ax_cal.grid(alpha=0.3); ax_cal.tick_params(labelsize=9)
    ax_cal.set_title(f"Rate-law calibration ({n_pass}/{len(df)} within band/$\\pm$0.05)",
                     fontsize=11)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return validation_rows


def write_rate_validation(rows, path):
    df = pd.DataFrame(rows).sort_values(["family", "attack", "strength", "G"])
    df.to_csv(path, index=False)
    # quick pass/fail summary printed to stdout
    df["in_band"] = (df["predicted"] >= df["ci_lo"]) & (df["predicted"] <= df["ci_hi"])
    df["small_gap"] = df["gap"].abs() < 0.05
    df["pass"] = df["in_band"] | df["small_gap"]
    n_pass = int(df["pass"].sum())
    n_tot = len(df)
    print(f"\nProp-4 validation: {n_pass}/{n_tot} (family,attack,|G|) cells pass "
          f"(predicted in 5-95% bootstrap band OR |gap| < 0.05)")
    fails = df[~df["pass"]][["family", "attack", "strength", "G",
                              "measured", "predicted", "gap", "mu", "sigma1"]]
    if len(fails):
        print("Fails:")
        print(fails.to_string(index=False))


# --------------------------------------------------------------------------- #
# Negative-control H0: plain-rollout true-key Z vs false-key leave-one-out Z
# --------------------------------------------------------------------------- #
def _sum_G(z, G, rng, n_trials):
    n = len(z)
    return z[rng.integers(0, n, size=(n_trials, G))].sum(axis=1)


def _sum_G_null_2d(z_null, G, rng, n_trials):
    n_rows = z_null.shape[0]
    ep = rng.integers(0, n_rows, size=(n_trials, G))
    key = rng.integers(0, N_FALSE, size=(n_trials, G))
    return z_null[ep, key].sum(axis=1)


def neg_control_compare(cal, G, rng, n_trials=N_TRIALS // 2):
    """Return (t_plain, t_false) — T_G samples under the literal H0 (plain
    rollouts, no injection) vs under the false-key null."""
    t_plain = _sum_G(cal.z_h0_plain, G, rng, n_trials)
    t_false = _sum_G_null_2d(cal.z_null, G, rng, n_trials)
    return t_plain, t_false


def neg_control_operating_fpr(cal, G, rng, n_trials=500_000, batch=50_000):
    """Realized plain-rollout FPR at the nominal decoy-key FPR_MAIN operating
    point. Use exact enumeration for |G|=1; for larger groups, use a larger
    chunked Monte Carlo estimate than the curve/CI defaults so displayed values
    are stable."""
    if G == 1:
        tau = float(np.quantile(cal.z_null.ravel(), 1 - FPR_MAIN))
        fpr_plain = float((cal.z_h0_plain >= tau).mean())
        return tau, fpr_plain

    plain_chunks, false_chunks = [], []
    remaining = int(n_trials)
    while remaining > 0:
        n = min(batch, remaining)
        t_plain, t_false = neg_control_compare(cal, G, rng, n_trials=n)
        plain_chunks.append(t_plain)
        false_chunks.append(t_false)
        remaining -= n
    t_plain = np.concatenate(plain_chunks)
    t_false = np.concatenate(false_chunks)
    tau = float(np.quantile(t_false, 1 - FPR_MAIN))
    fpr_plain = float((t_plain >= tau).mean())
    return tau, fpr_plain


# Formal family labels for figure titles (proper pi notation, upper-case datasets).
FAM_TITLE = {
    "lingbot/libero_10":   r"LingBot / LIBERO-10",
    "lingbot/robotwin10":  r"LingBot / RoboTwin",
    "pi0.5/libero_10":     r"$\pi_{0.5}$ / LIBERO-10",
    "pi0.5/robotwin10":    r"$\pi_{0.5}$ / RoboTwin",
}


def fig_neg_control_h0(cache, rng, path):
    """Per-family negative-control operating point: realized plain-rollout FPR
    when the threshold is calibrated to nominal 1% FPR on the decoy-key null.
    A point on/below the 1% line means the decoy-key null is conservative for
    the literal no-injection H0."""
    fams = sorted({k[0] for k in cache})
    tick_labels = {
        "lingbot/libero_10":   "LingBot\nLIBERO-10",
        "lingbot/robotwin10":  "LingBot\nRoboTwin",
        "pi0.5/libero_10":     "$\\pi_{0.5}$\nLIBERO-10",
        "pi0.5/robotwin10":    "$\\pi_{0.5}$\nRoboTwin",
    }
    fig, ax = plt.subplots(figsize=(4.8, 2.35), constrained_layout=True)
    x = np.arange(len(fams))
    styles = {
        1:  dict(offset=-0.13, marker="o", label="|G|=1"),
        16: dict(offset=0.13, marker="s", label="|G|=16"),
    }
    rows = []
    for i, fam in enumerate(fams):
        key = (fam, "clean", None)
        if key not in cache or len(cache[key].z_h0_plain) == 0:
            continue
        for G, st in styles.items():
            _, fpr_plain = neg_control_operating_fpr(cache[key], G, rng)
            fpr_plain *= 100.0
            rows.append((i + st["offset"], fpr_plain, G))

    for G, st in styles.items():
        sub = [(xx, yy) for xx, yy, gg in rows if gg == G]
        if not sub:
            continue
        xx, yy = zip(*sub)
        ax.scatter(xx, yy, marker=st["marker"], s=46, label=st["label"],
                   edgecolor="white", linewidth=0.55, zorder=4)
        for px, py in sub:
            ax.text(px, py + 0.13, f"{py:.1f}", ha="center", va="bottom",
                    fontsize=7.0, color="0.30", clip_on=False)

    ax.axhline(FPR_MAIN * 100.0, color="0.20", ls="--", lw=1.0, alpha=0.75,
               zorder=1, label="1% target")
    ax.set_xticks(x)
    ax.set_xticklabels([tick_labels.get(fam, fam) for fam in fams], fontsize=8.2)
    ax.set_ylabel("realized plain FPR (%)", fontsize=9.5)
    ax.set_ylim(-0.2, 3.8)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.tick_params(axis="y", labelsize=8.2, length=2.5, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.0, ncol=3, frameon=False,
              handlelength=1.15, columnspacing=1.25, handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_neg_control_h0_csv(cache, rng, path):
    """Per (family, attack, strength, G), report:
      fpr_target  : the calibrated-H0 (false-key) FPR target (1%)
      threshold   : the (1-fpr_target) quantile of T_G_false
      fpr_plain   : empirical FPR under plain-rollout H0 at that threshold
      n_plain     : number of plain rows
      ks_TG       : KS distance between empirical T_G_plain and T_G_false CDFs.
    Pass = fpr_plain within +/- 0.01 of fpr_target (or 2x for fpr_target=0.01).
    """
    out = []
    for (fam, attack, strength), cal in sorted(cache.items()):
        if len(cal.z_h0_plain) == 0:
            continue
        for G in (1, 4, 16):
            t_plain, t_false = neg_control_compare(cal, G, rng)
            tau = float(np.quantile(t_false, 1 - FPR_MAIN))
            fpr_plain = float((t_plain >= tau).mean())
            # KS via empirical CDF on common grid
            grid = np.linspace(
                min(t_plain.min(), t_false.min()),
                max(t_plain.max(), t_false.max()), 200)
            cdf_p = np.searchsorted(np.sort(t_plain), grid) / len(t_plain)
            cdf_f = np.searchsorted(np.sort(t_false), grid) / len(t_false)
            ks = float(np.max(np.abs(cdf_p - cdf_f)))
            out.append(dict(
                family=fam, attack=attack, strength=strength, G=G,
                fpr_target=FPR_MAIN, threshold=tau,
                fpr_plain=fpr_plain, fpr_plain_minus_target=fpr_plain - FPR_MAIN,
                ks_TG=ks, n_plain=len(cal.z_h0_plain),
                mean_plain=float(t_plain.mean()),
                mean_false=float(t_false.mean()),
                std_plain=float(t_plain.std(ddof=1)),
                std_false=float(t_false.std(ddof=1)),
            ))
    df = pd.DataFrame(out).sort_values(["family", "attack", "strength", "G"])
    df.to_csv(path, index=False)
    df["close"] = df["fpr_plain_minus_target"].abs() < max(FPR_MAIN, 0.01)
    print(f"\nNegative-control H0: {int(df['close'].sum())}/{len(df)} (family,attack,|G|) "
          f"cells have |empirical-plain-FPR - 1%| < 1%  at the false-key 1%-quantile threshold")
    bad = df[~df["close"]][["family", "attack", "strength", "G",
                             "fpr_plain", "ks_TG", "n_plain"]]
    if len(bad):
        print("Notable plain-null deviations from 1% target:")
        print(bad.to_string(index=False))


# --------------------------------------------------------------------------- #
# Cross-task vs same-task aggregation
# --------------------------------------------------------------------------- #
def tpr_point_same_task(z_h1, task_h1, z_null, task_null, G, fpr, rng,
                        n_trials=N_TRIALS):
    """Like tpr_point, but each trial picks ONE task and draws all G rows from
    inside it (for both H1 and the false-key null pool). Returns NaN if there
    is no common task with any rows on both sides."""
    common = sorted(set(task_h1) & set(task_null))
    if not common:
        return float("nan")
    h1_idx = {t: np.where(task_h1 == t)[0] for t in common}
    null_idx = {t: np.where(task_null == t)[0] for t in common}
    common = [t for t in common if len(h1_idx[t]) > 0 and len(null_idx[t]) > 0]
    if not common:
        return float("nan")
    task_choice = rng.integers(0, len(common), size=n_trials)
    t0 = np.empty(n_trials, dtype=np.float64)
    t1 = np.empty(n_trials, dtype=np.float64)
    for i_task, t in enumerate(common):
        mask = task_choice == i_task
        n = int(mask.sum())
        if n == 0:
            continue
        h1_pool = h1_idx[t]
        null_pool = null_idx[t]
        h1_pick = h1_pool[rng.integers(0, len(h1_pool), size=(n, G))]
        ep_pick = null_pool[rng.integers(0, len(null_pool), size=(n, G))]
        key_pick = rng.integers(0, N_FALSE, size=(n, G))
        t1[mask] = z_h1[h1_pick].sum(axis=1)
        t0[mask] = z_null[ep_pick, key_pick].sum(axis=1)
    return float((t1 >= np.quantile(t0, 1 - fpr)).mean())


def fig_aggregation_mode(cache, rng, path):
    """Solid = cross-task (default), dashed = same-task. Panel per family,
    one curve per (attack, strength) in the canonical set."""
    fams = sorted({k[0] for k in cache})
    nrows, ncols = 2, 2
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                                  squeeze=False)
    axes_flat = axes_grid.flatten()
    for ax in axes_flat[len(fams):]:
        ax.axis("off")
    rows = []
    for ax, fam in zip(axes_flat, fams):
        wanted = [("clean", None)] + [(a, s) for a, s in CANONICAL.items()]
        for attack, strength in wanted:
            key = (fam, attack, strength)
            if key not in cache:
                continue
            cal = cache[key]
            n_tasks = len(set(cal.task_h1) & set(cal.task_null_wm))
            tpr_cross = [tpr_point(cal.z_h1, cal.z_null, G, FPR_MAIN, rng)
                         for G in G_GRID]
            tpr_same = [tpr_point_same_task(cal.z_h1, cal.task_h1,
                                            cal.z_null, cal.task_null_wm,
                                            G, FPR_MAIN, rng) for G in G_GRID]
            for G, tc, ts in zip(G_GRID, tpr_cross, tpr_same):
                rows.append(dict(family=fam, attack=attack, strength=strength,
                                 G=G, tpr_cross=tc, tpr_same=ts,
                                 same_minus_cross=ts - tc, n_tasks=n_tasks))
            lbl = attack if strength is None else f"{attack} {strength:g}"
            line, = ax.plot(G_GRID, tpr_cross, marker="o", label=lbl)
            ax.plot(G_GRID, tpr_same, marker="x", linestyle="--",
                    color=line.get_color(), alpha=0.8)
        ax.set_title(fam); ax.set_xlabel(r"query budget $|G|$")
        ax.set_ylabel(r"TPR @ FPR$=1\%$  (solid: cross-task, dashed: same-task)")
        ax.set_xscale("log", base=2); ax.set_xticks(G_GRID)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return rows


def write_aggregation_mode_csv(rows, path):
    df = pd.DataFrame(rows).sort_values(["family", "attack", "strength", "G"])
    df.to_csv(path, index=False)
    # summary: average effect of constraining to same-task per family
    summary = df.groupby("family").agg(
        n_tasks=("n_tasks", "first"),
        mean_diff=("same_minus_cross", "mean"),
        worst_diff=("same_minus_cross", "min"))
    print("\nCross-task vs same-task TPR shift per family (same - cross, "
          "negative = same-task is HARDER due to correlated noise):")
    print(summary.to_string())


def fig_robustness(M, util, path):
    """The attacker's tradeoff: TPR@1% (|G|=G_TABLE) vs SR drop induced on the
    fingerprinted policy. One curve per attack; panel per family."""
    fams = _families_in_order(M)
    fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 3.4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        sub = M[M["family"] == fam]
        model, ds = sub.iloc[0]["model"], sub.iloc[0]["dataset"]
        u_clean = util.get((model, ds, "clean", None))
        sr_ref = u_clean["sr_wm"] if u_clean else None
        cl = sub[sub["attack"] == "clean"]
        if len(cl):
            ax.scatter([0], [cl["tpr1_gT"].iloc[0]], c="k", marker="*", s=90, zorder=5,
                       label="clean")
        for attack in [a for a in ATTACK_ORDER if a != "clean"]:
            pts = sub[sub["attack"] == attack].sort_values("strength")
            if not len(pts) or sr_ref is None:
                continue
            xs, ys = [], []
            for _, r in pts.iterrows():
                u = util.get((model, ds, attack, r["strength"]))
                if not u:
                    continue
                xs.append(max(0.0, sr_ref - u["sr_wm"]))
                ys.append(r["tpr1_gT"])
            if xs:
                idx = np.argsort(xs)
                ax.plot(np.array(xs)[idx], np.array(ys)[idx], marker="o", label=attack)
        ax.set_title(fam); ax.set_xlabel(r"task-success drop $\Delta$SR (attack on fp policy)")
        ax.set_ylabel(rf"TPR @ FPR$=1\%$, $|G|{{=}}{G_TABLE}$")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_attack_strength(M, path):
    """Grid rows=family, cols=attack: TPR@1% vs attack strength."""
    fams = _families_in_order(M)
    attacks = [a for a in ATTACK_ORDER if a != "clean"
               and ((M["attack"] == a) & M["strength"].notna()).any()]
    fig, axes = plt.subplots(len(fams), len(attacks),
                             figsize=(3.0 * len(attacks), 2.6 * len(fams)),
                             squeeze=False)
    for i, fam in enumerate(fams):
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            pts = M[(M["family"] == fam) & (M["attack"] == attack)
                    & M["strength"].notna()].sort_values("strength")
            if len(pts):
                ax.plot(pts["strength"], pts["tpr1_gT"], marker="o")
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(attack)
            if j == 0:
                ax.set_ylabel(f"{fam}\nTPR@1%", fontsize=8)
            if i == len(fams) - 1:
                ax.set_xlabel("strength")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _roc_points(z_h1, z_null, G, n_trials, rng):
    """Bootstrap ROC curve at query budget G.

    Returns (fpr, tpr) arrays sorted by FPR ascending, suitable for ax.plot.
    Uses quantile sweep over the union of T_G distributions for stable spacing.
    """
    z_h1 = np.asarray(z_h1, dtype=np.float64)
    z_null = np.asarray(z_null, dtype=np.float64).ravel()
    h1_idx = rng.integers(0, len(z_h1), size=(n_trials, G))
    null_idx = rng.integers(0, len(z_null), size=(n_trials, G))
    t_h1 = z_h1[h1_idx].sum(axis=1)
    t_null = z_null[null_idx].sum(axis=1)
    # threshold sweep via quantiles of the union for stable spacing
    taus = np.quantile(np.concatenate([t_h1, t_null]), np.linspace(0, 1, 201))
    fpr = np.array([(t_null > tau).mean() for tau in taus])
    tpr = np.array([(t_h1 > tau).mean() for tau in taus])
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def fig_attack_combined(M, util, cache, rng, path):
    """4x4 figure: rows=family, cols=attack (clip/ema/jitter/delay).
    Each panel shows per-strength ROC curves (x=FPR log scale, y=TPR linear).
    Clean baseline in black; swept strengths colored by viridis sequential map.
    Legend per panel: label = 'clean SR=X.XX' / 's=X SR=X.XX'.
    SR drawn from util dict; missing -> 'SR=--'.
    """
    ROC_N_TRIALS = 50_000   # bootstrap groups per ROC curve (16 panels x ~5 = ~80 ROCs)

    FAMILIES_ORDERED = ["lingbot/libero_10", "lingbot/robotwin10",
                        "pi0.5/libero_10", "pi0.5/robotwin10"]
    fams_in_data = _families_in_order(M)
    fams = [f for f in FAMILIES_ORDERED if f in fams_in_data]
    if not fams:
        fams = fams_in_data

    attacks = [a for a in ATTACK_ORDER if a != "clean"]

    n_rows, n_cols = len(fams), len(attacks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 5.5),
                             constrained_layout=True,
                             squeeze=False)

    for i, fam in enumerate(fams):
        fam_rows = M[M["family"] == fam]
        if len(fam_rows) == 0:
            model, ds = fam.split("/", 1)
        else:
            model = fam_rows.iloc[0]["model"]
            ds = fam_rows.iloc[0]["dataset"]

        ds_pretty = DS_PRETTY.get(ds, ds)

        for j, attack in enumerate(attacks):
            ax = axes[i][j]

            # ---------- cosmetics shared by all panels ----------
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
            ax.set_ylim(0, 1.04)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(alpha=0.3, which="both")
            ax.tick_params(labelsize=6)
            ax.axvline(FPR_MAIN, color="gray", linestyle="--", linewidth=0.7)

            if i == 0:
                ax.set_title(attack, fontsize=9)
            if j == 0:
                ax.set_ylabel("TPR", fontsize=8)
                ax.annotate(f"{model}/{ds_pretty}", xy=(-0.45, 0.5),
                            xycoords="axes fraction", fontsize=7,
                            rotation=90, va="center", ha="center")
            if i == n_rows - 1:
                ax.set_xlabel("FPR", fontsize=8)

            # ---------- N/A panel logic ----------
            na_panel = (fam in ("pi0.5/libero_10", "pi0.5/robotwin10")
                        and attack == "delay")
            sub = M[(M["family"] == fam) & (M["attack"] == attack)
                    & M["strength"].notna()].sort_values("strength")

            if na_panel or len(sub) == 0:
                # blank the log-scale x ticks so the panel still looks clean
                ax.set_xticklabels([])
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                continue

            strengths = sub["strength"].values
            n_strengths = len(strengths)
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_strengths))

            # ---------- clean baseline curve ----------
            clean_key = (fam, "clean", None)
            if clean_key in cache:
                cal_clean = cache[clean_key]
                fpr_c, tpr_c = _roc_points(
                    cal_clean.z_h1, cal_clean.z_null,
                    G_TABLE, ROC_N_TRIALS, rng)
                u_clean = util.get((model, ds, "clean", None))
                sr_clean_str = (f"{u_clean['sr_wm']:.2f}"
                                if u_clean else "--")
                ax.plot(fpr_c, tpr_c, color="black", linewidth=1.4,
                        label=f"clean SR={sr_clean_str}")

            # ---------- per-strength curves ----------
            for k, s in enumerate(strengths):
                key = (fam, attack, s)
                if key not in cache:
                    continue
                cal = cache[key]
                fpr_s, tpr_s = _roc_points(
                    cal.z_h1, cal.z_null,
                    G_TABLE, ROC_N_TRIALS, rng)
                u_atk = util.get((model, ds, attack, round(float(s), 4)))
                sr_atk_str = (f"{u_atk['sr_wm']:.2f}"
                              if u_atk else "--")
                ax.plot(fpr_s, tpr_s, color=colors[k], linewidth=1.0,
                        label=f"s={s:g} SR={sr_atk_str}")

            ax.legend(loc="lower right", fontsize=5, frameon=False)

    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
