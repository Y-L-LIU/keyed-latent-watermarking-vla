#!/usr/bin/env python3
"""tab:descendant from raw descendant CSVs (all 4 cells re-scoreable).

Per cell: N=n_wm, pairwise = AUC(z_wm vs z_plain, |G|=1), AUC(|G|=16), TPR@1%(|G|=16).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_verification as av

RAW = HERE.parent / "attack_c_data" / "per_episode_scores_descendant_raw"
G, Q = 16, 0.01
# (Family, Dataset, csv stem)  -- order to match committed table
ROWS = [
    (r"$\pi_{0.5}$", "LIBERO-10", "pi05_libero_descendant"),
    (r"$\pi_{0.5}$", "RoboTwin",  "pi05_robotwin_descendant"),
    ("LingBot",      "LIBERO-10", "lingbot_libero_descendant"),
    ("LingBot",      "RoboTwin",  "lingbot_robotwin_descendant"),
]


def metrics(stem):
    df = pd.read_csv(RAW / f"{stem}.csv")
    cal = av.calibrate(df)
    rng = np.random.default_rng(0)
    n = len(cal.z_h1)
    pairwise = av.auc(cal.z_h1, cal.z_h0_plain) if len(cal.z_h0_plain) else float("nan")
    auc16 = av.auc_group(cal.z_h1, cal.z_null, G, rng)
    tpr = av.tpr_point(cal.z_h1, cal.z_null, G, Q, rng)
    return n, pairwise, auc16, tpr


def main():
    body = []
    print(f"{'cell':22s}{'N':>5s}{'pairwise':>10s}{'AUC16':>8s}{'TPR@1%':>8s}")
    for fam, ds, stem in ROWS:
        n, pw, a, t = metrics(stem)
        print(f"{fam+'/'+ds:22s}{n:5d}{pw:10.2f}{a:8.3f}{t:8.2f}")
        body.append(f"{fam} & {ds:14s} & {n:3d} & {pw:.2f} & {a:.3f} & {t:.2f} \\\\")
    tex = r"""% group-aggregated detection on LoRA descendants (|G|=16, FPR=1%), raw matched filter;
% computed by results/make_tab_descendant.py from raw descendant score CSVs.
\begin{table}[t]\centering
\caption{Detection on LoRA-fine-tuned descendants in the $2{\times}2$
model$\times$dataset design. Group-aggregated AUC and TPR at
FPR$=1\%$ use the standard query budget $|G|{=}16$; pairwise is the
per-episode-pair fraction with $s_{\mathrm{wm}}>s_{\mathrm{plain}}$. See
\Cref{sec:eval-robust} for the descendant setup.}
\label{tab:descendant}\small
\begin{tabular}{ll r ccc}
\toprule
Family & Dataset & $N$ & pairwise & AUC ($|G|{=}16$) & TPR@1\% \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (HERE / "_raw_tab_descendant.tex").write_text(tex)
    print(f"\nwrote {HERE/'_raw_tab_descendant.tex'}")


if __name__ == "__main__":
    main()
