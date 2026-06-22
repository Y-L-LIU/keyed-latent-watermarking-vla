#!/usr/bin/env python
"""Spectral-signature figure for paper Sec 5.1 (imperceptibility).

Shows, per architecture, the power spectral density (PSD) of the executed
7-DoF action stream for: (i) the plain policy, (ii) the plain policy with the
output-space sine watermark added post-hoc (f in {1,2} Hz, beta=0.05), and
(iii) the latent-fingerprinted policy. The sine watermark appears as sharp
peaks at 1 and 2 Hz that an attacker can locate by FFT and remove with a
band-stop (shaded 0.8-2.2 Hz); the latent fingerprint adds no localized peak --
its energy is dispersed by the generator and submerged in the natural action
spectrum, so there is no band to target. This is the spectral picture behind
tab:sine-baseline (sine matched-filter score collapses ~90% under the shaded
band-stop, while latent detection survives).

Pure post-hoc on saved rollouts (no GPU). Reads executed_actions (T,7) @ 20 Hz.
"""
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

FS = 20.0                      # executed-action sample rate (Hz)
TONES = (1.0, 2.0)             # sine watermark tones
BETA = 0.05                    # sine watermark strength used in tab:sine-baseline
BETA_ILLUS = 0.3               # illustrative strength for the figure: at beta=0.05 the
                               # tone sits below the natural action floor (matched-filter
                               # detectable but not a visible peak), so we scale it up so
                               # the band-localized energy is visible. Confinement to the
                               # 1-2 Hz band is beta-invariant; only the height changes.
BAND = (0.8, 2.2)             # Butterworth band-stop band used in tab:sine-baseline
XMAX = 6.0                     # plot range (Hz); most action energy is < 4 Hz
NPERSEG = 80                   # 4 s window -> 0.25 Hz bins; 1 and 2 Hz land on bins
MAX_EP = 80                    # episodes per pool (plenty for an averaged PSD)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
PI05 = "/workspace/vla/eval_out/base/libero_10/rollouts/none/task_rollout"
LB_WM = "/workspace/vla/eval_out/lingbot_libero10_descendant/libero_10"
LB_PL = "/workspace/vla/eval_out/lingbot_libero10_descendant_plain/libero_10"


def _avg_psd(acts):
    """Mean Welch PSD over a list of (T,7) action arrays and their 7 dims."""
    psds = []
    for a in acts:
        T = a.shape[0]
        if T < NPERSEG:
            continue
        a = a - a.mean(0, keepdims=True)
        f, p = welch(a, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2, axis=0)
        psds.append(p.mean(1))  # average over the 7 action dims
    return f, np.mean(psds, 0)


def _add_sine(a, nonce, beta=BETA):
    """Add the keyed output-space sine watermark to a (T,7) action array."""
    T = a.shape[0]
    t = np.arange(T) / FS
    rng = np.random.default_rng(int(nonce) ^ 12345)
    s = np.zeros_like(a)
    for fz in TONES:
        phi = rng.uniform(0, 2 * np.pi, size=a.shape[1])
        s += (beta / len(TONES)) * np.sin(2 * np.pi * fz * t[:, None] + phi[None, :])
    return a + s


def load_pi05():
    plain, wm = [], []
    for fp in sorted(glob.glob(os.path.join(PI05, "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        if "executed_actions" not in z.files:
            continue
        a = z["executed_actions"].astype(np.float64)
        v = str(z["variant"])
        nonce = z["episode_nonce"]
        (plain if v == "plain" else wm).append((a, nonce))
    return plain[:MAX_EP], wm[:MAX_EP]


def load_lingbot():
    def grab(d):
        out = []
        for fp in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(fp, allow_pickle=True)
            if "executed_actions" not in z.files:
                continue
            out.append((z["executed_actions"].astype(np.float64), z["episode_nonce"]))
        return out[:MAX_EP]
    return grab(LB_PL), grab(LB_WM)


def panel(ax, plain, wm, title):
    f, psd_plain = _avg_psd([a for a, _ in plain])
    _, psd_sine = _avg_psd([_add_sine(a, n, BETA_ILLUS) for a, n in plain])
    _, psd_lat = _avg_psd([a for a, _ in wm])
    ax.axvspan(BAND[0], BAND[1], color="0.85", zorder=0, label="band-stop")
    ax.semilogy(f, psd_plain, color="0.45", lw=2.0, label="plain")
    ax.semilogy(f, psd_lat, color="#1f77b4", lw=2.2, label="latent (ours)")
    ax.semilogy(f, psd_sine, color="#d62728", lw=2.2,
                label=rf"$+$ output sine ($\beta{{=}}{BETA_ILLUS}$)")
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.35)


def main():
    pi_pl, pi_wm = load_pi05()
    lb_pl, lb_wm = load_lingbot()
    print(f"pi0.5: {len(pi_pl)} plain / {len(pi_wm)} wm; lingbot: {len(lb_pl)} / {len(lb_wm)}")
    plt.rcParams.update({
        "font.size": 12.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0), sharey=True)
    panel(axes[0], pi_pl, pi_wm, r"$\pi_{0.5}$ / LIBERO-10")
    panel(axes[1], lb_pl, lb_wm, "LingBot / LIBERO-10")
    axes[0].set_ylabel("action PSD")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, framealpha=0.95, handlelength=1.35, columnspacing=1.0,
               borderpad=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    for out in (PAPER / "fig_spectrum.pdf",
                RESULTS / "fig_spectrum.pdf",
                RESULTS / "fig_spectrum_preview.png"):
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
