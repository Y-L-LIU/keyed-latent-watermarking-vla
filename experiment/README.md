# Experiment index — keyed-latent VLA watermark paper

A **map**, not a copy. Every script here is a thin wrapper that `exec`s the real
script in place (under `/workspace/vla/...`); nothing is duplicated. The only thing
the wrappers change is the environment/paths a script needs on *this* node
(system python `/usr/bin/python3.11`, `MUJOCO_GL=osmesa`, the `/workspace/vla
→ /workspace/vla` remap for the campaign launchers). Shared env lives in
[`env.sh`](env.sh).

Paper source lives under `/workspace/vla/paper/`. Section numbers below follow the
`\section` order in the paper.

## Quick start (just enter this folder)
```bash
cd /workspace/vla/experiment
bash run.sh check        # preflight: what's runnable now (inputs + targets)
bash run.sh list         # every section -> wrapper -> paper §
bash run.sh all-analysis # regenerate all CPU-only paper assets from your rollout scores
bash run.sh verification # or one section by name (see `list`)
```
Wrappers are cwd-independent (they resolve their own dir), so any of the above
also works as an absolute path from anywhere.

## Two stages

Every result is produced in two stages:

1. **Rollout (GPU)** — run the policy + verifier, save per-episode `.npz`.
   Wrappers in [`rollout/`](rollout). Run these first; this release ships no
   precomputed rollout data.
2. **Analysis (CPU)** — read per-episode data, emit the paper's `.tex`/`.pdf`.
   Wrappers in [`analysis/`](analysis). This is what you re-run while writing.

The bridge between them is the **score export** (`rollout/export_scores.sh`):
rollout `.npz` → `attack_c_data/per_episode_scores/*_partial_map_*.csv`, which is
what the §6.3 / §6.5 analysis globs.

To regenerate all CPU-only assets once the score CSVs exist (run the rollout
stage first to produce them):
```bash
bash analysis/run_all.sh
```

---

## Section → asset → script map

### §5.1  Why latent, not output injection (imperceptibility)
| Asset | Analysis (CPU) | Rollout (GPU) data source |
|---|---|---|
| `fig_spectrum.pdf` | `analysis/sec5.1_spectrum.sh` → `results/make_fig_spectrum.py` | clean pi05 + lingbot LIBERO-10 rollouts (`eval_out/base/...`, `eval_out/lingbot_libero10_descendant{,_plain}`) |
| `fig_bandstop_sweep.pdf` (sine curve) | `analysis/sec5.1_bandstop.sh` → `results/make_fig_bandstop_sweep.py` | same rollouts + `results/sine_baseline_*.csv` |
| `fig_bandstop_sweep.pdf` (latent curve) | `analysis/sec5.1_latent_bandstop.sh` → `results/make_latent_bandstop_sweep.py` **(GPU)** | re-runs partial+MAP on band-stopped pi05 rollouts |

Sine-baseline rollout CSVs (`results/sine_baseline_per_episode_*.csv`) were
produced ad-hoc; they already exist on disk and feed the sine curve directly.

### §5.2 / §6.2  Why MAP, not ODE inversion (recovery)
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `tab_recovery.tex` *(hand-tabulated from JSON)* | `analysis/sec5.2_6.2_recovery.sh` → `compute_recovery_metrics.py` (+ `openpi_postproc_recovery_cells.py` to fill full+MAP / partial+ODE cells) | `attack_c_data/campaign/scripts/sweep_pi05_libero_recovery.sh` → `eval_..._postprocess_robustness.py --save-all-inversion-modes`. Output `eval_out/base/<suite>/` |
| `tab_pad_ablation.tex` *(hand-tabulated)* | pad-value `V` sweep numbers (typed from the sweep output) | pad-value `V` sweep via the worktree `eval_libero_action_inversion.py` (off-distribution pad → reverse-Euler) |

### §6.3  Verification  (+ Appendix A utility)
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `tab_main.tex`, `tab_utility.tex`, `fig_tpr_vs_G.pdf`, `fig_rate_calibration.pdf`, `fig_neg_control_h0.pdf`, `fig_aggregation_mode.pdf`, `verification_metrics.csv` | `analysis/sec6.3_verification.sh` → `results/analyze_verification.py` | openpi: `rollout/openpi_verification.sh`. lingbot: `rollout/lingbot_verification.sh` |

Inputs to the analysis: `attack_c_data/per_episode_scores/*_partial_map_*.csv`
(+ `utility_pi05.csv`, `utility_lingbot.csv`). Produced from rollout `.npz` by the
**score export** step.

### §6.4  Robustness to removal attacks
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `fig_attack_combined.pdf` (output attacks: clip/ema/jitter/delay) | `analysis/sec6.3_verification.sh` (same script) | openpi delay: `rollout/openpi_attacks_delay.sh`; lingbot all: `rollout/lingbot_attacks.sh [libero\|robotwin]` |
| `tab_descendant.tex` *(hand-tabulated)* (LoRA descendants) | `results/descendant_group_metrics.csv` (post-processed) | **train**: `train/{openpi,lingbot}_lora.sh [libero\|robotwin]` (see Training below); then eval with the base detector via the verification rollout, `--checkpoint-dir` pointed at the descendant ckpt |
| §12.5 compression (prune30/int8) rows | (feeds the same descendant/robustness pipeline) | **build**: `train/build_compressed_ckpt.sh <openpi\|lingbot> <prune\|quant> SRC DST`; **eval**: `rollout/{openpi,lingbot}_compression.sh [libero\|robotwin]`. Aggregate: `aggregate_compression_results.py` → `RESULTS_compression.md` |

### §6.5  Identification
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `tab_identification.tex`, `fig_identification.pdf`, `fig_identification_robustness.pdf`, `identification_metrics.csv` | `analysis/sec6.5_identification.sh` → `results/analyze_identification.py` | **none new** — reuses §6.3 per-episode scores + `per_episode_scores_descendant/` (lingbot descendant rescore via `lingbot-va/wan_va/wm/export_lingbot_per_episode_scores.py`) |

### §7.1  Key uniqueness
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `sec_key_uniqueness.tex` numbers, `key_collision_analysis.csv` | `analysis/sec7.1_uniqueness.sh` → `results/make_key_collision_analysis.py` | reuses clean pi05/LIBERO-10 watermarked rollouts (`eval_out/base/.../task_rollout`) |

### §7.2  Unforgeability
| Asset | Analysis (CPU) | Rollout (GPU) |
|---|---|---|
| `sec_unforgeability.tex` numbers, `unforgeability_analysis.csv` | `analysis/sec7.2_unforgeability.sh` → `results/make_unforgeability_analysis.py` | same pool as §7.1 |

---

## Training (GPU) — LoRA descendants + model surgery

These produce the "descendant/suspect" models for §6.4 (`tab_descendant`) and §12.5
compression. All wrap root launchers in place; checkpoints land under
`openpi-checkpoints/` or `lingbot_out/`; logs under `ft_logs/`.

```bash
# pi0.5 LoRA descendant
bash train/openpi_lora.sh libero      # run_openpi_ft.sh: goal(GPU0-3)+spatial(GPU4-7),
                                       #   configs pi05_libero_{goal,spatial}_lora_from_libero,
                                       #   exp descendant_lora (norm-stats first)
                                       #   -> openpi-checkpoints/<cfg>/descendant_lora/<step>/
bash train/openpi_lora.sh robotwin    # run_robotwin_ft.sh: 8-GPU, pi05_aloha_robotwin_lora_local,
                                       #   exp robotwin_descendant
                                       #   -> openpi-checkpoints/pi05_aloha_robotwin_lora_local/robotwin_descendant/<step>/

# lingbot Wan LoRA descendant (merge-on-save)
bash train/lingbot_lora.sh libero     # libero_lora_train, GPU4-7  -> lingbot_out/libero_lora/checkpoints/checkpoint_step_*/
bash train/lingbot_lora.sh robotwin   # robotwin_lora_train (bbh), GPU0-3 -> lingbot_out/robotwin_lora/checkpoints/checkpoint_step_*/

# §12.5 model surgery: prune30 / int8-quant a descendant ckpt (then eval w/ base detector)
bash train/build_compressed_ckpt.sh openpi  prune  <SRC-step-dir>     <DST-step-dir>     [--prune-sparsity 0.3]
bash train/build_compressed_ckpt.sh lingbot quant  <SRC/transformer> <DST/transformer>
```

Eval of a descendant uses the verification rollout with the base detector kept fixed:
`rollout/openpi_verification.sh` / `rollout/lingbot_verification.sh` pointed at the
descendant checkpoint (detector stays the original base — that is the §6.4 test).

## §6.3 score export (rollout `.npz` → per-episode CSV)

The analysis globs `attack_c_data/per_episode_scores/*_partial_map_*.csv`. To
(re)build one from a rollout dir:

**lingbot** (`export_lingbot_per_episode_scores.py`, `--preset libero|robotwin`):
```bash
"$PY" -m wan_va.wm.export_lingbot_per_episode_scores \
  --rollout-dir /workspace/vla/eval_out/lingbot_libero10_descendant/libero_10 \
  --out /workspace/vla/attack_c_data/per_episode_scores/lingbot_libero_clean_partial_map_wm.csv \
  --attack clean --dataset libero_10 --preset libero
# (run from /workspace/vla/lingbot-va with PYTHONPATH=/workspace/vla/lingbot-va)
```

**openpi** exporter:
`openpi/scripts/attacks/export_per_episode_scores.py`.

---

## Notes / caveats
- **No code is copied.** Editing a real script under `results/`, `openpi/`,
  `lingbot-va/`, or `attack_c_data/campaign/` changes behavior here immediately.
- **Hand-authored tables.** `tab_recovery`, `tab_pad_ablation`, `tab_descendant`
  are typed from the analysis JSON/CSV — there is no single `*.py` that writes
  them. The analysis wrapper produces the *numbers*; the `.tex` is filled by hand.
- **Campaign launchers** (`attack_c_data/campaign/scripts/*.sh`) assume the
  canonical `/workspace/vla` root and a project-local `.venv`. The lingbot-attack
  wrappers remap those on the fly (`env.sh:remap_and_run`); verify before a long
  GPU run.
- **Rendering**: a node without an EGL/Vulkan ICD must use `MUJOCO_GL=osmesa` (CPU).
  RoboTwin sapien rollouts need a Vulkan-capable node.
- **Where to copy assets into the paper**: analysis scripts write `.pdf`/`.tex`
  into `results/` (and identification writes directly into `paper/`). Copy the
  rest into `paper/`.
