#!/usr/bin/env python3
"""TPR@1% vs query budget |G| for the under-saturated pi0.5 cells (global-tau* detector).
Shows weak-but-positive per-episode signals all climb to ~1.0 as |G| grows (the sqrt|G| rate
law) -- graceful attenuation, not removal. Reads the committed global-tau* per-episode CSVs."""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import analyze_verification as av
J=32; rng=np.random.default_rng(0)
RAW=Path("/workspace/vla/attack_c_data/per_episode_scores_raw")
COMP=Path("/workspace/vla/attack_c_data/per_episode_scores_compression_raw")
GS=[16,32,64]   # honest ceiling: <= n_wm (cells re-rolled to ~130); no bootstrap-with-replacement past n
def zcal(p):
    df=pd.read_csv(p); st=df["s_true"].to_numpy(float); sf=df[[f"s_false_{i+1}" for i in range(J)]].to_numpy(float)
    wm=(df["variant"]=="watermarked").to_numpy()
    mu=sf.mean(1);sd=sf.std(1,ddof=1);zt=(st-mu)/(sd+1e-9)
    loo=(sf.sum(1,keepdims=True)-sf)/(J-1);sq=(sf**2).sum(1,keepdims=True);lv=((sq-sf**2)-(J-1)*loo**2)/(J-2)
    zn=(sf-loo)/(np.sqrt(np.clip(lv,0,None))+1e-9)
    return zt[wm], zn[wm].ravel()
def tpr_vs_G(p):
    h1,null=zcal(p); out=[]
    for G in GS:
        H1=np.array([rng.choice(h1,G,replace=True).sum() for _ in range(6000)])
        N=np.array([rng.choice(null,G,replace=True).sum() for _ in range(6000)])
        out.append(float((H1>=np.quantile(N,0.99)).mean()))
    return out
# Two groups: cells that RECOVER with budget (solid/dashed, climb to ~1.0) and the two most
# aggressive output perturbations that PRESERVE task success yet defeat detection even at |G|=64
# (dotted, stay near the floor) -- the honest boundary of output-level robustness.
RECOVER=[("delay-1",RAW/"pi05_libero_10_partial_map_delay_1.csv","o-"),
         ("delay-2",RAW/"pi05_libero_10_partial_map_delay_2.csv","s-"),
         ("delay-3",RAW/"pi05_libero_10_partial_map_delay_3.csv","^-"),
         ("EMA 0.5",RAW/"pi05_libero_10_partial_map_ema_0.5.csv","D--"),
         (r"$\pi_{0.5}$/RT prune30",COMP/"pi05_robotwin10_partial_map_prune30.csv","P--")]
SOFT=[(r"EMA 0.2 (SR kept)",RAW/"pi05_libero_10_partial_map_ema_0.2.csv","v:"),
      (r"jitter 0.1 (SR kept)",RAW/"pi05_libero_10_partial_map_jitter_0.1.csv","X:")]
fig,ax=plt.subplots(figsize=(5.0,3.4))
for lbl,p,sty in RECOVER:
    if not p.exists(): continue
    y=tpr_vs_G(p); ax.plot(GS,y,sty,label=lbl,ms=5,lw=1.6)
    print(f"{lbl:22} TPR@1% vs |G|={GS}: {[round(v,2) for v in y]}")
for lbl,p,sty in SOFT:
    if not p.exists(): continue
    y=tpr_vs_G(p); ax.plot(GS,y,sty,label=lbl,ms=6,lw=1.6,color="0.45")
    print(f"{lbl:22} TPR@1% vs |G|={GS}: {[round(v,2) for v in y]}")
ax.axhline(1.0,color="0.7",ls=":",lw=0.8)
ax.axhline(0.9,color="0.85",ls=":",lw=0.8)
ax.set_xscale("log",base=2); ax.set_xticks(GS); ax.set_xticklabels(GS)
ax.set_xlabel(r"query budget $|G|$"); ax.set_ylabel(r"TPR @ 1\% FPR"); ax.set_ylim(-0.03,1.05)
ax.set_title(r"Budget recovers all but the most aggressive smoothing/jitter",fontsize=9.5)
ax.legend(fontsize=7,ncol=2,loc="center right"); ax.grid(alpha=0.3)
fig.tight_layout()
for out in ["/workspace/vla/results/fig_budget_curve.pdf","/workspace/vla/paper/fig_budget_curve.pdf"]:
    fig.savefig(out); print("wrote",out)
