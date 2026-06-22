#!/usr/bin/env python
"""Band-stop-sweep figure for paper Sec 5.1 (imperceptibility / removability).

Rather than a raw spectrogram (at beta=0.05 the sine is matched-filter-detectable
but NOT a visible magnitude peak), this shows the operationally meaningful fact:
the output-space sine watermark's *detectability* is confined to a single narrow
band. We sweep the center of a fixed-width (1.4 Hz) 4th-order Butterworth band-stop
across the spectrum, and report the matched-filter detection score (normalized to
no-attack) of the sine watermark. The score survives every notch EXCEPT one placed
on the 1-2 Hz watermark band, where it collapses ~90% -- i.e., a single cheap notch
erases the watermark. The latent fingerprint, dispersed across the spectrum by the
generator, has no such removable band (cf. tab:sine-baseline; pilot latent MF
0.74->0.63 under the same notch).

Pure post-hoc on saved rollouts (no GPU). Reuses the plain pools / sine recipe of
make_fig_spectrum.py.
"""
import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

spec = importlib.util.spec_from_file_location(
    "mfs", os.path.join(os.path.dirname(__file__), "make_fig_spectrum.py"))
mfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfs)

FS = mfs.FS
TONES = mfs.TONES
WIDTH = 1.4                                     # notch width (matches table's 0.8-2.2)
CENTERS = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]   # band-stop center frequencies (Hz)
WM_BAND = (1.0, 2.0)                            # where the sine tones live
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"


def _template(T, phi):
    """Unit-norm keyed matched-filter template, shape (T, 7, n_tones)."""
    t = np.arange(T) / FS
    tm = np.stack([np.sin(2 * np.pi * fz * t[:, None] + phi[None, :]) for fz in TONES], -1)
    tm /= np.linalg.norm(tm, axis=0, keepdims=True) + 1e-9
    return tm


def _mf_score(a, phi):
    """Coherent matched-filter score of action (T,7) against keyed sine template."""
    tm = _template(a.shape[0], phi)
    return float(np.einsum("td,tdk->", a - a.mean(0, keepdims=True), tm))


def _bandstop(a, fc):
    lo, hi = max(fc - WIDTH / 2, 0.05), min(fc + WIDTH / 2, FS / 2 - 0.05)
    b, aa = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="bandstop")
    return filtfilt(b, aa, a, axis=0)


def sweep(pool):
    """Mean matched-filter score of the sine watermark vs band-stop center."""
    centers = ["none"] + CENTERS
    scores = {c: [] for c in centers}
    for a, nonce in pool:
        if a.shape[0] < mfs.NPERSEG:
            continue
        rng = np.random.default_rng(int(nonce) ^ 12345)
        phi = rng.uniform(0, 2 * np.pi, size=a.shape[1])
        wm = mfs._add_sine(a, nonce)
        scores["none"].append(_mf_score(wm, phi))
        for fc in CENTERS:
            scores[fc].append(_mf_score(_bandstop(wm, fc), phi))
    base = np.mean(scores["none"])
    xs = CENTERS
    ys = [np.mean(scores[fc]) / base for fc in CENTERS]
    return xs, ys, 1.0  # normalized so no-attack == 1.0


def panel(ax, pool, title):
    xs, ys, base = sweep(pool)
    ax.axvspan(WM_BAND[0], WM_BAND[1], color="#d62728", alpha=0.13, zorder=0,
               label="watermark band")
    ax.axhline(base, color="0.6", ls="--", lw=1.3, label="no attack")
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=2.2, ms=6, label="sine MF score")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0.5, 8.2)
    ax.set_xlabel("notch center (Hz)")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.35)


def main():
    pi_pl, _ = mfs.load_pi05()
    lb_pl, _ = mfs.load_lingbot()
    print(f"pi0.5 plain {len(pi_pl)}, lingbot plain {len(lb_pl)}")
    plt.rcParams.update({
        "font.size": 12.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
    })
    fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.9), sharey=True)
    panel(ax[0], pi_pl, r"$\pi_{0.5}$ / LIBERO-10")
    panel(ax[1], lb_pl, "LingBot / LIBERO-10")
    ax[0].set_ylabel("MF score\n(norm. to no attack)")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, framealpha=0.95, handlelength=1.35, columnspacing=1.0,
               borderpad=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    for out in (RESULTS / "fig_bandstop_sweep_preview.png",
                PAPER / "fig_bandstop_sweep.pdf",
                RESULTS / "fig_bandstop_sweep.pdf"):
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
