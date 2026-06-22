# Action-Space Output Watermark Baseline Analysis

## Overview

Comparison of two watermark approaches for VLA policy action outputs:
1. **Output sine watermark** (baseline): adds `beta * sin(2πft + φ)` directly to action outputs
2. **Internal latent watermark**: injects structured noise in diffusion latent space, signal passes through nonlinear denoising network

Evaluation on Pi0.5 LIBERO-10, action_dim=7, replan_steps=5, sample_rate=20Hz.

## 1. Online Evaluation Results (beta=0.05, 10 episodes)

| Metric | Sine Watermark |
|--------|---------------|
| AUC (marked vs plain) | 1.0000 |
| Success rate (plain) | 100% |
| Success rate (watermarked) | 100% |
| Matched-filter: plain mean | 0.110 |
| Matched-filter: marked mean | 0.591 |
| Matched-filter: wrong-key mean | 0.107 |

Conclusion: with beta=0.05 and full-episode matched filter (500 steps), detection is perfect and task performance is unaffected.

## 2. Robustness to Random Perturbations

### 2.1 EMA Smoothing Attack

Attack: `y[t] = alpha * x[t] + (1-alpha) * y[t-1]` applied to watermarked output.

| Smooth alpha | AUC | Marked score | Plain score | Gap |
|---|---|---|---|---|
| None | 1.0000 | 0.792 | 0.081 | 0.711 |
| 0.8 | 1.0000 | 0.764 | 0.079 | 0.685 |
| 0.5 | 1.0000 | 0.614 | 0.070 | 0.544 |
| 0.3 | 1.0000 | 0.383 | 0.048 | 0.335 |
| 0.2 | 1.0000 | 0.225 | 0.036 | 0.190 |

Conclusion: AUC remains 1.0 at all smoothing levels. Matched filter is inherently robust to EMA because sine correlation accumulates coherently over 500 steps.

### 2.2 Additive Gaussian Jitter

Attack: add iid N(0, σ²) noise to each action step.

| Jitter σ | σ/beta | AUC |
|---|---|---|
| 0.00 | 0x | 1.0000 |
| 0.01 | 0.2x | 1.0000 |
| 0.02 | 0.4x | 1.0000 |
| 0.05 | 1x | 1.0000 |
| 0.10 | 2x | 1.0000 |
| 0.20 | 4x | 1.0000 |
| 0.50 | 10x | 0.9626 |

Conclusion: white noise needs to be 10x the watermark amplitude before AUC degrades. Matched filter rejects uncorrelated noise by 1/√N factor.

## 3. Vulnerability: Frequency Domain Attack

### 3.1 Spectral Detectability

The sine watermark is trivially visible in the frequency domain:

| Metric | Sine watermark | Latent watermark |
|--------|---------------|-----------------|
| In-band (0.8-2.2Hz) power ratio vs plain | **1.94x** | 1.12x |
| Spectral flatness change | drops (0.57→0.51) | unchanged (0.57→0.57) |
| Max single-bin power spike | **~60x** | no consistent spike |

The sine watermark appears as a clear spectral peak at the injected frequency. An attacker can detect its presence with a single FFT, without knowing the secret key.

The latent watermark's signal, having passed through the diffusion model's nonlinear decoder, is spectrally flat and indistinguishable from natural action variability.

### 3.2 Bandstop Filter Removal

Applying bandstop (Butterworth order-4) filters to watermarked actions:

| Filter band | Sine MF score (from 0.83) | Latent MF score (from 0.74) | Action distortion (RMSE) |
|---|---|---|---|
| None | 0.835 | 0.741 | 0.000 |
| 0.8–2.2 Hz | **0.016** | 0.633 | 0.040 |
| 0.5–3.0 Hz | 0.010 | 0.570 | 0.052 |
| 0.1–5.0 Hz | 0.009 | 0.403 | 0.079 |
| 0.1–8.0 Hz | 0.010 | 0.190 | 0.097 |

Key findings:
- **Sine watermark**: a 1.4Hz-wide bandstop filter completely destroys it (MF: 0.83→0.016) with negligible action distortion (RMSE=0.04).
- **Latent watermark**: even removing 0.1–5.0 Hz (half the Nyquist range) only reduces MF to 0.40, still detectable. Removing 0.1–8.0 Hz (80% of spectrum) is needed to approach destruction, but RMSE=0.097 severely degrades control performance.

## 4. Summary: Strengths and Weaknesses

| Property | Output Sine Watermark | Internal Latent Watermark |
|----------|----------------------|--------------------------|
| Implementation complexity | Trivial (post-hoc addition) | Requires model modification |
| Detection from raw output | Perfect (AUC=1.0) | Perfect (AUC=1.0 via MAP) |
| Robustness to smoothing | Strong (AUC=1.0 at alpha=0.2) | Strong |
| Robustness to jitter | Strong (needs 10x sigma) | Strong |
| Spectral stealth | **None** (visible peak) | **High** (spectrally flat) |
| Robustness to bandstop filter | **None** (trivially removed) | **High** (broadband, can't target) |
| Requires model access to bypass | No (just filter output) | Yes (need to modify inference) |
| Robustness to distillation | None (student won't learn sine) | Partial (depends on capacity) |

## 5. Conclusion

The output sine watermark is a useful **lower bound baseline**: it achieves perfect detection under benign conditions and survives stochastic attacks (noise, smoothing), but is **trivially defeated by any frequency-aware attacker**. A single bandstop filter removes it with essentially zero cost to task performance.

The internal latent watermark's key advantage is that its action-space signature is **broadband** — the diffusion model's nonlinear transform spreads the watermark energy across all frequencies, eliminating any single attackable frequency band. Removing it requires destroying the entire action signal.

---

*Experiment config: Pi0.5, LIBERO-10, checkpoint pi05_libero, beta=0.05, freq=[1.0, 2.0]Hz, 500 steps/episode, 7-dim actions.*
