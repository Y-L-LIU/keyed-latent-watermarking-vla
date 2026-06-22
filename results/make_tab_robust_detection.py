#!/usr/bin/env python3
"""Combined detection-robustness table -- the detection-metric twin of
tab:identification-attack. One table, all attacks: output-removal attacks
(clip/EMA/jitter swept lo/canonical/hi, delay per step) and owner-side weight
edits (LoRA fine-tune, prune30, int8), across the 2x2 family design.

Metrics mirror tab:weight-level / tab:main: per-episode AUC (|G|=1), group
AUC16, and TPR at FPR=1% (both |G|=16). Raw matched filter at lag zero (L=0),
matching the rest of the main evaluation (the synchronization search is the
separate tab:lagsearch ablation).

Sources (all per-episode raw CSVs, no re-rollout):
  output attacks : attack_c_data/per_episode_scores_raw/*_partial_map_*.csv
  LoRA finetune  : attack_c_data/per_episode_scores_descendant_raw/*.csv
  prune30/int8   : attack_c_data/per_episode_scores_compression_raw/*.csv
                   (pi0.5/RoboTwin compression is ACTION-ONLY, marked with a dagger)
Writes paper/tab_robust_detection.tex. Re-runnable; not hand-edited.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av  # noqa: E402

ROOT = av.HERE.parent / "attack_c_data"
RAW = ROOT / "per_episode_scores_raw"
DESC = ROOT / "per_episode_scores_descendant_raw"
COMP = ROOT / "per_episode_scores_compression_raw"
PAPER = av.HERE.parent / "paper"

FAMILY_ORDER = ["lingbot/libero_10", "lingbot/robotwin10",
                "pi0.5/libero_10", "pi0.5/robotwin10"]
CANON = {"clip": 1.0, "ema": 0.5, "jitter": 0.01}
ATK_PRETTY = {"clip": r"\texttt{clip}", "ema": r"\texttt{EMA}",
              "jitter": r"\texttt{jitter}", "delay": r"\texttt{delay}"}
G16 = av.G_TABLE
GBIG = 64    # larger budget shown in parentheses where a cell is not yet saturated at G16
             # (honest ceiling: cells re-rolled to ~130 wm so |G|=64 <= n_wm, no resampling)
DAGGER = r"$^\dagger$"


def _fam_pretty(fam):
    model, ds = fam.split("/", 1)
    m = {"lingbot": "LingBot", "pi0.5": r"$\pi_{0.5}$"}.get(model, model)
    d = {"libero_10": "LIBERO-10", "robotwin10": "RoboTwin"}.get(ds, ds)
    return f"{m}/{d}"


def _strength(r0):
    st = r0.get("attack_strength")
    return None if (pd.isna(st) or str(st).strip() in ("", "-")) else float(st)


def load_all():
    """(family, attack, strength) -> {auc, auc16, tpr16}. attack is the CSV's
    attack column (clean/clip/ema/jitter/delay/descendant/prune30/quant)."""
    rng_t = np.random.default_rng(av.RNG_SEED)
    rng_a = np.random.default_rng(av.RNG_SEED + 1)
    rng_t2 = np.random.default_rng(av.RNG_SEED + 2)
    rng_a2 = np.random.default_rng(av.RNG_SEED + 3)
    met = {}
    paths = (sorted(RAW.glob("*_partial_map_*.csv"))
             + (sorted(DESC.glob("*.csv")) if DESC.is_dir() else [])
             + (sorted(COMP.glob("*.csv")) if COMP.is_dir() else []))
    for p in paths:
        if p.name.endswith("_lagsearch.csv"):      # L=0 main eval excludes the lag search
            continue
        df = pd.read_csv(p)
        r0 = df.iloc[0]
        fam = f'{r0["model"]}/{r0["dataset"]}'
        attack = str(r0["attack"])
        key = (fam, attack, _strength(r0))
        cal = av.calibrate(df)
        met[key] = dict(
            auc=av.auc(cal.z_h1, cal.z_null.ravel()),
            auc16=av.auc_group(cal.z_h1, cal.z_null, G16, rng_a),
            tpr16=av.tpr_point(cal.z_h1, cal.z_null, G16, av.FPR_MAIN, rng_t),
            auc_big=av.auc_group(cal.z_h1, cal.z_null, GBIG, rng_a2),
            tpr_big=av.tpr_point(cal.z_h1, cal.z_null, GBIG, av.FPR_MAIN, rng_t2),
        )
    return met


def _fmt(x):
    return f"{x:.2f}" if (x is not None and np.isfinite(x)) else "--"


def cells(met, fams, attack, strength):
    """5 metric strings per family for one (attack, strength):
    per-episode AUC ($|G|{=}1$), then AUC and TPR at $|G|{=}16$ and at $|G|{=}64$."""
    out = []
    for f in fams:
        m = met.get((f, attack, strength))
        if not m:
            out += ["--"] * 5; continue
        out += [_fmt(m["auc"]), _fmt(m["auc16"]), _fmt(m["tpr16"]),
                _fmt(m["auc_big"]), _fmt(m["tpr_big"])]
    return out


def fam_strengths(met, fam, attack):
    return sorted(s for (f, a, s) in met
                  if f == fam and a == attack and s is not None)


def build(met, path):
    fams = [f for f in FAMILY_ORDER
            if any(k[0] == f for k in met)]
    ncol = 1 + 5 * len(fams)
    # per family: AUC($|G|{=}1$)  then AUC|TPR at |G|=16  then AUC|TPR at |G|=64.
    # The vertical rule splits AUC from TPR inside each budget group.
    colspec = "l " + " ".join("c c|c c|c" for _ in fams)

    def row(label, cs):
        return f"{label} & " + " & ".join(cs) + r" \\"

    fam_hdr = " & " + " & ".join(
        rf"\multicolumn{{5}}{{c}}{{{_fam_pretty(f)}{DAGGER if f == 'pi0.5/robotwin10' else ''}}}"
        for f in fams) + r" \\"
    fam_cmid = "".join(rf"\cmidrule(lr){{{5 * i + 2}-{5 * i + 6}}}" for i in range(len(fams)))
    budget_hdr = " & " + " & ".join(
        r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{$|G|{=}16$} & \multicolumn{2}{c}{$|G|{=}64$}"
        for _ in fams) + r" \\"
    budget_cmid = "".join(
        rf"\cmidrule(lr){{{5 * i + 3}-{5 * i + 4}}}\cmidrule(lr){{{5 * i + 5}-{5 * i + 6}}}"
        for i in range(len(fams)))
    metric_hdr = "Condition & " + " & ".join(
        r"AUC$_1$ & AUC & TPR & AUC & TPR" for _ in fams) + r" \\"

    L = [
        r"% auto-generated by results/make_tab_robust_detection.py -- do not hand-edit",
        r"\begin{table*}[t]\centering",
        r"\caption{Detection robustness across the $2{\times}2$ family design, under "
        r"adversary output-removal attacks and owner-side weight variants. Each cell gives "
        rf"per-episode AUC (AUC$_1$, $|G|{{=}}1$, watermarked vs.\ plain), then AUC and TPR at "
        rf"FPR$=1\%$ at the two group budgets $|G|{{=}}{G16}$ and $|G|{{=}}{GBIG}$ "
        r"(the vertical rule separates AUC from TPR within each budget). "
        r"For \texttt{clip}/\texttt{EMA}/\texttt{jitter} each attack is shown at its smallest "
        r"(lo), canonical ($\star$), and largest (hi) swept strength (values in "
        r"\Cref{sec:eval-robust}); \texttt{delay} is per step. "
        r"Detection uses the verifier's synchronization search, which realigns a constant "
        r"delay before matched filtering (\Cref{tab:lagsearch}); it is inert ($\tau^*{=}0$) on "
        r"every other condition. `--' marks a condition absent for that "
        r"family. $^\dagger$For $\pi_{0.5}$/RoboTwin the compression rows restrict the edit "
        r"to the action expert (${\sim}16\%$ of parameters), since compressing the whole "
        r"model discards the vision--language backbone.}",
        r"\label{tab:robust-detection}\small",
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
        row("none (clean)", cells(met, fams, "clean", None)),
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Output removal attacks}}}} \\",
    ]
    for attack in ("clip", "ema", "jitter"):
        for tag, sel in ((r"\,lo", 0), (r"\,$\star$", "c"), (r"\,hi", -1)):
            cs = []
            for f in fams:
                fs = fam_strengths(met, f, attack)
                if not fs:
                    cs += ["--"] * 5
                    continue
                s = CANON[attack] if sel == "c" else fs[sel]
                cs += cells(met, [f], attack, s)
            L.append(row(ATK_PRETTY[attack] + tag, cs))
    delays = sorted({s for (f, a, s) in met if a == "delay" and s is not None})
    for s in delays:
        L.append(row(rf"{ATK_PRETTY['delay']} {s:g}", cells(met, fams, "delay", s)))
    L += [
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Owner-side weight variants}}}} \\",
        row("LoRA finetune", cells(met, fams, "descendant", None)),
        row(r"\texttt{prune30}", cells(met, fams, "prune30", None)),
        row(r"\texttt{int8}", cells(met, fams, "quant", None)),
        r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}",
    ]
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path}")


def main():
    met = load_all()
    # console preview
    print(f"{'family':22}{'attack':>10}{'str':>7}{'AUC':>7}{'AUC16':>8}{'TPR':>7}")
    for k in sorted(met, key=lambda k: (FAMILY_ORDER.index(k[0]) if k[0] in FAMILY_ORDER else 9,
                                        str(k[1]), k[2] if k[2] is not None else -1)):
        m = met[k]
        print(f"{k[0]:22}{str(k[1]):>10}{(f'{k[2]:g}' if k[2] is not None else '-'):>7}"
              f"{m['auc']:7.2f}{m['auc16']:8.2f}{m['tpr16']:7.2f}")
    build(met, PAPER / "tab_robust_detection.tex")


if __name__ == "__main__":
    main()
