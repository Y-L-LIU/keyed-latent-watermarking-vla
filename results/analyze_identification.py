#!/usr/bin/env python3
"""
Identification (key attribution) results from the SAME raw per-episode WMF
scores that feed analyze_verification.py / tab_main.

Verification (tab_main) asks a *binary* question per candidate key:
    H0: suspect does not carry k*   vs   H1: suspect carries k*.
Identification reframes the exact same evidence as a *multi-class* decision over
a gallery of candidate keys. The gallery is the true key k* plus the J=32 false
keys already scored in every CSV, so |gallery| = 33 -- no new rollouts, no new
scoring: we reuse s_true / s_false_1..32 verbatim.

Per episode e and candidate key k we use the same false-key-calibrated z-score as
verification (eq:zscore):
    Z_e(k*)        = (s_true   - mean(s_false))      / std(s_false)
    Z_e(k_false_j) = (s_false_j - mean(s_false_{-j})) / std(s_false_{-j})   (leave-one-out)
all on one comparable scale. We aggregate over a probe group G (eq:aggregation)
    T_G(k) = sum_{e in G} Z_e(k)
and rank the 33 gallery keys by T_G.

  * Closed-set Top-k / CMC: the suspect carries some gallery key (here k*); the
    decision is argmax_k T_G(k). CMC(r) = P(rank of k* <= r); Top-1 = CMC(1).
  * Open-set DIR@FAR: the suspect may carry NO gallery key. Genuine probes are
    watermarked groups; impostor probes are plain (no-injection) groups, whose
    gallery scores are all null. With tau set so impostor FAR = alpha,
    DIR@FAR(alpha) = P(genuine probe is rank-1 correct AND max_k T_G(k) >= tau).

Probe groups are bootstrap-resampled with replacement, matching the |G| sweep in
analyze_verification.py. Same CSV discovery, same N_FALSE / FPR / |G| conventions.

Outputs (this dir):
  identification_metrics.csv   per (family, attack, strength, |G|): closed-set
                               rank-1/rank-5 + open-set DIR@1%/DIR@10%
  ../paper/tab_identification.tex   headline closed+open identification table (clean)
  ../paper/fig_identification.pdf   CMC curves + rank-1-vs-|G| (clean)
  fig_identification.pdf            same, mirrored into results/

Reuses the same raw per-episode score CSVs as verification, including descendant
and compression rows when those files are present.
"""
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent          # vla/results
# Paper switched to the RAW matched filter (whitening removed) paper-wide, so the
# identification figure/table read the raw per-episode scores like tab_main does.
SCORE_DIR = HERE.parent / "attack_c_data" / "per_episode_scores_raw"
DESC_DIR = HERE.parent / "attack_c_data" / "per_episode_scores_descendant_raw"
# Owner-side compression variants (prune30 / int8 quant), scored by build_raw_compression.py.
COMP_DIR = HERE.parent / "attack_c_data" / "per_episode_scores_compression_raw"
PAPER_DIR = HERE.parent / "paper"
OUT_DIR = HERE
EPS = 1e-8
RNG_SEED = 0
N_FALSE = 32
FALSE_COLS = [f"s_false_{i}" for i in range(1, N_FALSE + 1)]
GALLERY = N_FALSE + 1               # true key + 32 decoys = 33 candidate keys

G_GRID = [1, 2, 4, 8, 16, 32]       # budgets swept in the figure
G_SINGLE = 1
G_TABLE = 16                        # aggregated budget (matches tab_main)
G_TABLE2 = 64                       # larger honest budget (<=n_wm after the re-roll)
RANKS_TABLE = (1, 5)                # closed-set ranks reported in the table
FAR_TABLE = (0.01, 0.10)            # open-set DIR@FAR operating points
N_TRIALS = 20000                    # MC probe groups per point estimate

CANONICAL = {"clip": 1.0, "ema": 0.5, "jitter": 0.01, "delay": 1}
DS_PRETTY = {"libero_10": "LIBERO-10", "robotwin10": "RoboTwin"}
MODEL_PRETTY = {"lingbot": "LingBot"}  # display name; data key stays "lingbot"
# table/figure order parallel to fig_attack_combined in analyze_verification.py
FAMILY_ORDER = ["lingbot/libero_10", "lingbot/robotwin10",
                "pi0.5/libero_10", "pi0.5/robotwin10"]


# --------------------------------------------------------------------------- #
# Calibration: z-scores split into watermarked (genuine) and plain (impostor)
# --------------------------------------------------------------------------- #
Ident = namedtuple("Ident", [
    "zt_wm", "zd_wm",      # true-key z (n_wm,) and decoy z (n_wm, 32) on wm rows
    "zt_pl", "zd_pl",      # true-key z (n_pl,) and decoy z (n_pl, 32) on plain rows
])


def _decoy_loo_z(s_false):
    """Leave-one-out z-score of each of the 32 false keys against the other 31.
    Same construction as analyze_verification.calibrate's z_null_all."""
    csum = s_false.sum(axis=1, keepdims=True)
    loo_mu = (csum - s_false) / (N_FALSE - 1)
    sq = (s_false ** 2).sum(axis=1, keepdims=True)
    loo_var = ((sq - s_false ** 2) - (N_FALSE - 1) * loo_mu ** 2) / (N_FALSE - 2)
    return (s_false - loo_mu) / (np.sqrt(np.clip(loo_var, 0, None)) + EPS)


def calibrate(df):
    s_false = df[FALSE_COLS].to_numpy(float)
    s_true = df["s_true"].to_numpy(float)
    is_wm = df["variant"].to_numpy() == "watermarked"

    mu = s_false.mean(axis=1)
    sd = s_false.std(axis=1, ddof=1)
    z_true = (s_true - mu) / (sd + EPS)          # k* z, all rows (eq:zscore)
    z_decoy = _decoy_loo_z(s_false)              # 32 decoy z, all rows
    return Ident(
        zt_wm=z_true[is_wm], zd_wm=z_decoy[is_wm],
        zt_pl=z_true[~is_wm], zd_pl=z_decoy[~is_wm],
    )


# --------------------------------------------------------------------------- #
# Closed-set CMC and open-set DIR@FAR
# --------------------------------------------------------------------------- #
def _group_scores(zt, zd, G, rng, n_trials):
    """Bootstrap n_trials probe groups of |G| episodes (with replacement) and
    return aggregated gallery scores: T_true (n_trials,) and T_decoy
    (n_trials, 32)."""
    n = len(zt)
    idx = rng.integers(0, n, size=(n_trials, G))
    T_true = zt[idx].sum(axis=1)                 # (n_trials,)
    T_decoy = zd[idx].sum(axis=1)                # (n_trials, 32)
    return T_true, T_decoy


def cmc_curve(cal, G, rng, n_trials=N_TRIALS):
    """Closed-set rank distribution of the true key over the 33-key gallery on
    watermarked (genuine) probe groups. Returns CMC array of length GALLERY
    (CMC[r-1] = P(rank <= r)) and the per-trial rank-1 indicator pool."""
    T_true, T_decoy = _group_scores(cal.zt_wm, cal.zd_wm, G, rng, n_trials)
    rank_true = 1 + (T_decoy > T_true[:, None]).sum(axis=1)   # 1..33
    cmc = np.array([(rank_true <= r).mean() for r in range(1, GALLERY + 1)])
    return cmc, rank_true, T_true, T_decoy


def dir_at_far(cal, G, fars, rng, n_trials=N_TRIALS):
    """Open-set detection-and-identification rate at each false-alarm rate.
    Genuine = watermarked groups; impostor = plain groups. Returns dict
    far -> DIR (rank-1-correct AND above the impostor-calibrated threshold)."""
    if len(cal.zt_pl) == 0:
        # attack CSVs may hold watermarked episodes only -> no impostor pool.
        return {a: float("nan") for a in fars}
    # genuine
    T_true_g, T_decoy_g = _group_scores(cal.zt_wm, cal.zd_wm, G, rng, n_trials)
    rank1 = T_true_g >= T_decoy_g.max(axis=1)                 # argmax == k*
    # impostor: max over the full 33-key gallery on plain probe groups
    T_true_i, T_decoy_i = _group_scores(cal.zt_pl, cal.zd_pl, G, rng, n_trials)
    score_max_imp = np.maximum(T_true_i, T_decoy_i.max(axis=1))
    out = {}
    for a in fars:
        tau = float(np.quantile(score_max_imp, 1 - a))
        out[a] = float((rank1 & (T_true_g >= tau)).mean())
    return out


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def _load_csv(p, group):
    df = pd.read_csv(p)
    r0 = df.iloc[0]
    st = r0.get("attack_strength")
    strength = None if (pd.isna(st) or str(st).strip() in ("", "-")) else float(st)
    return dict(
        df=df, model=str(r0["model"]), dataset=str(r0["dataset"]),
        attack=str(r0["attack"]), strength=strength, group=group,
        family=f'{r0["model"]}/{r0["dataset"]}',
        n_wm=int((df["variant"] == "watermarked").sum()),
        n_pl=int((df["variant"] == "plain").sum()),
    )


def discover():
    """Main (output-attack) score CSVs plus, if present, weight-level LoRA
    descendant CSVs (attack column == 'descendant', CPU-rescored from saved
    map_z)."""
    conds = [_load_csv(p, "main") for p in sorted(SCORE_DIR.glob("*_partial_map_*.csv"))]
    if DESC_DIR.is_dir():
        conds += [_load_csv(p, "descendant") for p in sorted(DESC_DIR.glob("*.csv"))]
    if COMP_DIR.is_dir():
        conds += [_load_csv(p, "compression") for p in sorted(COMP_DIR.glob("*.csv"))]
    return conds


def _fam_sort_key(fam):
    return FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(RNG_SEED)
    conds = discover()
    if not conds:
        print("No score CSVs found.")
        return

    cache = {}   # (family, attack, strength) -> Ident
    rows = []
    for c in conds:
        cal = calibrate(c["df"])
        cache[(c["family"], c["attack"], c["strength"])] = cal
        for G in sorted(set(G_GRID) | {G_TABLE2}):
            cmc, rank_true, _, _ = cmc_curve(cal, G, rng)
            dirs = dir_at_far(cal, G, FAR_TABLE, rng)
            rows.append(dict(
                family=c["family"], model=c["model"], dataset=c["dataset"],
                attack=c["attack"], strength=c["strength"], group=c["group"],
                n_wm=c["n_wm"], n_pl=c["n_pl"], gallery=GALLERY, G=G,
                rank1=cmc[0], rank5=cmc[4],
                dir_far01=dirs[0.01], dir_far10=dirs[0.10],
            ))
    M = pd.DataFrame(rows)
    M.to_csv(OUT_DIR / "identification_metrics.csv", index=False)
    cols = ["family", "attack", "strength", "G", "n_wm", "n_pl",
            "rank1", "rank5", "dir_far01", "dir_far10"]
    print(M[cols].to_string(index=False))

    write_identification_table(M, PAPER_DIR / "tab_identification.tex")
    write_attack_identification_table(M, PAPER_DIR / "tab_identification_attack.tex")
    fig_identification(cache, rng, PAPER_DIR / "fig_identification.pdf")
    fig_identification(cache, rng, OUT_DIR / "fig_identification.pdf")
    # legacy robustness figure (replaced by tab_identification_attack in the paper);
    # kept in results/ only for cross-checking the table numbers.
    fig_identification_robustness(M, OUT_DIR / "fig_identification_robustness.pdf")
    print(f"\nWrote tab_identification.tex + tab_identification_attack.tex + "
          f"fig_identification.pdf "
          f"(gallery size = {GALLERY} keys = true + {N_FALSE} decoys)")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def write_identification_table(M, path):
    """Closed-set Top-1/Top-5 (|G|=1 and |G|=16) + open-set DIR@1%/10% FAR
    (|G|=16) on the clean policy, one row per family -- parallel to tab_main."""
    fams = sorted(M["family"].unique(), key=_fam_sort_key)
    L = [
        r"% auto-generated by results/analyze_identification.py -- do not hand-edit",
        r"\begin{table*}[t]\centering",
        r"\caption{Key identification over a gallery of "
        rf"{GALLERY} candidate keys (the true key $k^*$ plus the $J{{=}}{N_FALSE}$ "
        r"false keys), reusing the same $T_G(k)$ evidence as \Cref{tab:main} with an "
        r"$\arg\max$ decision rule. \emph{Closed-set} reports rank-1 and rank-5 "
        r"identification (CMC) at single-episode ($|G|{=}1$) and aggregated "
        rf"($|G|{{=}}{G_TABLE}$) budgets. \emph{{Open-set}} reports the "
        r"detection-and-identification rate (rank-1, above an impostor-calibrated "
        r"threshold) at false-alarm rates $1\%$ and $10\%$, with plain "
        rf"(no-injection) rollouts as impostors, at $|G|{{=}}{G_TABLE}$. Clean "
        r"policy, partial observation, MAP recovery; probe groups bootstrap-resampled.}",
        r"\label{tab:identification}\small",
        r"\begin{tabular}{ll cc cc cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Closed-set R1} "
        r"& \multicolumn{2}{c}{Closed-set R5} "
        r"& \multicolumn{2}{c}{Open-set DIR@FAR} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        rf"Model & Dataset & $|G|{{=}}1$ & $|G|{{=}}{G_TABLE}$ "
        rf"& $|G|{{=}}1$ & $|G|{{=}}{G_TABLE}$ & $1\%$ & $10\%$ \\",
        r"\midrule",
    ]
    def _row(sub):
        r1 = sub[sub["G"] == G_SINGLE].iloc[0]
        r16 = sub[sub["G"] == G_TABLE].iloc[0]
        model = MODEL_PRETTY.get(r16["model"], r16["model"])
        ds = DS_PRETTY.get(r16["dataset"], r16["dataset"])
        return (f"{model} & {ds} "
                f"& {r1['rank1']:.2f} & {r16['rank1']:.2f} "
                f"& {r1['rank5']:.2f} & {r16['rank5']:.2f} "
                f"& {r16['dir_far01']:.2f} & {r16['dir_far10']:.2f} \\\\")

    L.append(r"\multicolumn{8}{l}{\emph{Clean policy}} \\")
    for fam in fams:
        sub = M[(M["family"] == fam) & (M["attack"] == "clean") & (M["group"] == "main")]
        if len(sub):
            L.append(_row(sub))

    # Weight-level LoRA descendants from the raw descendant score CSVs.
    desc = M[(M["group"] == "descendant")]
    if len(desc):
        L.append(r"\midrule")
        L.append(r"\multicolumn{8}{l}{\emph{Weight-level: LoRA fine-tuned descendant}} \\")
        for fam in sorted(desc["family"].unique(), key=_fam_sort_key):
            sub = desc[desc["family"] == fam]
            if len(sub):
                L.append(_row(sub))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Table: identification under attack (output attacks by strength + weight edits)
# --------------------------------------------------------------------------- #
ATK_PRETTY = {"clip": r"\texttt{clip}", "ema": r"\texttt{EMA}",
              "jitter": r"\texttt{jitter}", "delay": r"\texttt{delay}"}


def _fmt(v):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def write_attack_identification_table(M, path):
    """Closed-set rank-1 at the single-episode ($|G|{=}1$, R1$_1$) and aggregated
    ($|G|{=}16$, R1$_{16}$) budgets, plus open-set DIR@1% (D, at $|G|{=}16$), under
    each removal attack swept over strength and the owner-side weight variants (LoRA
    fine-tune, prune30, int8 quant). Three columns per family; '--' = condition not
    run for that family."""
    fams = [f for f in FAMILY_ORDER if f in set(M["family"])]
    m1 = M[M["G"] == G_SINGLE]
    m16 = M[M["G"] == G_TABLE]
    m64 = M[M["G"] == G_TABLE2]
    ncol = 1 + 5 * len(fams)

    def _pick(mm, family, attack, strength, group):
        sub = mm[(mm["family"] == family) & (mm["attack"] == attack) & (mm["group"] == group)]
        return sub[sub["strength"].isna()] if strength is None else sub[sub["strength"] == strength]

    def by_attack(attack, strength, group="main"):
        return {f: (_pick(m1, f, attack, strength, group),
                    _pick(m16, f, attack, strength, group),
                    _pick(m64, f, attack, strength, group)) for f in fams}

    def by_group(group, attack=None):
        out = {}
        for f in fams:
            s1 = m1[(m1["family"] == f) & (m1["group"] == group)]
            s16 = m16[(m16["family"] == f) & (m16["group"] == group)]
            s64 = m64[(m64["family"] == f) & (m64["group"] == group)]
            if attack is not None:
                s1 = s1[s1["attack"] == attack]
                s16 = s16[s16["attack"] == attack]
                s64 = s64[s64["attack"] == attack]
            out[f] = (s1, s16, s64)
        return out

    def _v(sub, col):
        return _fmt(sub[col].iloc[0]) if len(sub) else "--"

    def row(label, picks):
        cells = []
        for f in fams:
            s1, s16, s64 = picks[f]
            cells += [_v(s1, "rank1"),
                      _v(s16, "rank1"), _v(s16, "dir_far01"),
                      _v(s64, "rank1"), _v(s64, "dir_far01")]
        return f"{label} & " + " & ".join(cells) + r" \\"

    colspec = "l " + " ".join("c c|c c|c" for _ in fams)
    fam_hdr = " & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{_fam_pretty(f)}}}" for f in fams) + r" \\"
    fam_cmid = "".join(rf"\cmidrule(lr){{{5 * i + 2}-{5 * i + 6}}}" for i in range(len(fams)))
    budget_hdr = " & " + " & ".join(
        r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{$|G|{=}16$} & \multicolumn{2}{c}{$|G|{=}64$}"
        for _ in fams) + r" \\"
    budget_cmid = "".join(
        rf"\cmidrule(lr){{{5 * i + 3}-{5 * i + 4}}}\cmidrule(lr){{{5 * i + 5}-{5 * i + 6}}}"
        for i in range(len(fams)))
    metric_hdr = "Condition & " + " & ".join(r"R1$_1$ & R1 & D & R1 & D" for _ in fams) + r" \\"

    L = [
        r"% auto-generated by results/analyze_identification.py -- do not hand-edit",
        r"\begin{table*}[t]\centering",
        r"\caption{Identification under adversary output-removal attacks and owner-side weight variants, "
        rf"over the {GALLERY}-key gallery. Each family reports closed-set rank-1 at the "
        rf"single-episode (R1$_1$, $|G|{{=}}1$) budget, then rank-1 (R1) and open-set "
        rf"DIR@$1\%$ FAR (D) at the two aggregated budgets $|G|{{=}}{G_TABLE}$ and "
        rf"$|G|{{=}}{G_TABLE2}$ (the vertical rule separates R1 from D within each budget). "
        r"For \texttt{clip}/\texttt{EMA}/\texttt{jitter} each attack is shown at its smallest "
        r"(lo), canonical ($\star$), and largest (hi) swept strength (values in "
        r"\Cref{sec:eval-robust}); \texttt{delay} is per step. "
        r"The owner-side variant rows are the LoRA fine-tuned descendant and the compressed "
        r"(\texttt{prune30}, \texttt{int8}) variants. "
        r"`--' marks a condition absent for that family.}",
        r"\label{tab:identification-attack}\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        fam_hdr,
        fam_cmid,
        budget_hdr,
        budget_cmid,
        metric_hdr,
        r"\midrule",
        row(r"none (clean)", by_attack("clean", None)),
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Output removal attacks}}}} \\",
    ]
    CANON = {"clip": 1.0, "ema": 0.5, "jitter": 0.01}
    def fam_strengths(f, attack):
        return sorted(m16[(m16["family"] == f) & (m16["attack"] == attack)
                          & m16["strength"].notna()]["strength"].unique())
    for attack in ("clip", "ema", "jitter"):
        for tag, sel in ((r"\,lo", 0), (r"\,$\star$", "c"), (r"\,hi", -1)):
            picks = {}
            for f in fams:
                fs = fam_strengths(f, attack)
                if not fs:
                    picks[f] = (m1.iloc[0:0], m16.iloc[0:0], m64.iloc[0:0]); continue
                s = CANON[attack] if sel == "c" else fs[sel]
                picks[f] = (_pick(m1, f, attack, s, "main"), _pick(m16, f, attack, s, "main"),
                            _pick(m64, f, attack, s, "main"))
            L.append(row(ATK_PRETTY[attack] + tag, picks))
    for s in sorted(m16[(m16["attack"] == "delay") & m16["strength"].notna()]["strength"].unique()):
        L.append(row(rf"{ATK_PRETTY['delay']} {s:g}", by_attack("delay", s)))
    L += [
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Owner-side weight variants}}}} \\",
        row(r"LoRA finetune", by_group("descendant")),
        row(r"\texttt{prune30}", by_group("compression", "prune30")),
        row(r"\texttt{int8}", by_group("compression", "quant")),
        r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}",
    ]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Figure: identification robustness to output attacks (rank-1 vs strength)
# --------------------------------------------------------------------------- #
def fig_identification_robustness(M, path):
    """Grid rows=family, cols=attack: closed-set rank-1 at |G|=16 vs attack
    strength. The clean (no-attack) rank-1 is the dashed horizontal reference.
    Parallels analyze_verification.fig_attack_strength."""
    Mm = M[M["group"] == "main"]
    fams = sorted(Mm["family"].unique(), key=_fam_sort_key)
    attacks = [a for a in ("clip", "ema", "jitter", "delay")
               if ((Mm["attack"] == a) & Mm["strength"].notna()).any()]
    fig, axes = plt.subplots(len(fams), len(attacks),
                             figsize=(2.9 * len(attacks), 2.4 * len(fams)),
                             squeeze=False)
    for i, fam in enumerate(fams):
        clean = Mm[(Mm["family"] == fam) & (Mm["attack"] == "clean")
                   & (Mm["G"] == G_TABLE)]
        clean_r1 = clean["rank1"].iloc[0] if len(clean) else None
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            pts = Mm[(Mm["family"] == fam) & (Mm["attack"] == attack)
                     & (Mm["G"] == G_TABLE) & Mm["strength"].notna()
                     ].sort_values("strength")
            if len(pts):
                ax.plot(pts["strength"], pts["rank1"], marker="o")
                if clean_r1 is not None:
                    ax.axhline(clean_r1, color="gray", ls="--", lw=0.8, alpha=0.7)
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
                ax.set_xticks([])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(attack)
            if j == 0:
                ax.set_ylabel(f"{fam}\nrank-1 ($|G|{{=}}{G_TABLE}$)", fontsize=8)
            if i == len(fams) - 1:
                ax.set_xlabel("attack strength")
    fig.suptitle("Closed-set rank-1 identification under output attacks "
                 f"($|G|{{=}}{G_TABLE}$; dashed = clean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure: CMC curves (|G|=1 vs |G|=16) + rank-1 vs |G|
# --------------------------------------------------------------------------- #
def _fam_pretty(fam):
    model, ds = fam.split("/", 1)
    m = {"lingbot": "LingBot", "pi0.5": r"$\pi_{0.5}$"}.get(model, model)
    return f"{m}/{DS_PRETTY.get(ds, ds)}"


def fig_identification(cache, rng, path):
    fams = sorted({k[0] for k in cache}, key=_fam_sort_key)
    colors = {f: c for f, c in zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams))))}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # --- panels 1-2: CMC at |G|=1 and |G|=16 ---
    for ax, G in zip(axes[:2], (G_SINGLE, G_TABLE)):
        for fam in fams:
            key = (fam, "clean", None)
            if key not in cache:
                continue
            cmc, *_ = cmc_curve(cache[key], G, rng)
            ranks = np.arange(1, GALLERY + 1)
            ax.plot(ranks, cmc, marker="o", ms=3, color=colors[fam],
                    label=_fam_pretty(fam))
        ax.set_title(rf"Closed-set CMC ($|G|{{=}}{G}$)", fontsize=13)
        ax.set_xlabel("rank $r$", fontsize=12)
        ax.set_ylabel("identification rate (CMC)", fontsize=12)
        ax.set_xlim(1, GALLERY)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    # --- panel 3: rank-1 (closed) + DIR@1% (open) vs |G| ---
    ax = axes[2]
    for fam in fams:
        key = (fam, "clean", None)
        if key not in cache:
            continue
        cal = cache[key]
        r1 = [cmc_curve(cal, G, rng)[0][0] for G in G_GRID]
        d1 = [dir_at_far(cal, G, (0.01,), rng)[0.01] for G in G_GRID]
        line, = ax.plot(G_GRID, r1, marker="o", color=colors[fam])
        ax.plot(G_GRID, d1, marker="x", ls="--", color=line.get_color(), alpha=0.8)
    ax.set_title("Rank-1 (solid) / open-set DIR@1% (dashed)", fontsize=13)
    ax.set_xlabel(r"query budget $|G|$", fontsize=12)
    ax.set_ylabel("rate", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(G_GRID)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)

    # --- one shared legend, large font, single row beneath all three panels ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23)
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=15, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=2.4, markerscale=1.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
