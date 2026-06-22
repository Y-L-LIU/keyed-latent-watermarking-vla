# Keyed Latent Watermarking for Vision-Language-Action Models

Anonymous code release for the paper submission. It contains everything needed
to reproduce the paper's results: the two modified VLA stacks, the watermark /
detection / attack code, the GPU rollout launchers, and the CPU analysis
scripts that turn per-episode scores into every table and figure in the paper.

All absolute paths in scripts use the canonical root `/workspace/vla`. Either
clone the repo there or symlink it: `ln -s /path/to/clone /workspace/vla`.
A few sweep launchers write large checkpoints to a second volume,
`/workspace/vla_out`; point it anywhere with free disk.

## Layout

```
openpi/        modified pi0.5 stack (fork of Physical-Intelligence/openpi)
               ├─ src/openpi/...                 watermark hooks in the model/policy/training code
               ├─ scripts/eval_libero_*.py       LIBERO rollout + detection evals
               ├─ scripts/eval_robotwin_*.py     RoboTwin rollout + detection evals
               ├─ scripts/attacks/               adaptive-attack fine-tuning (Attack C/D), compression
               │                                 surgery, per-episode score exporters, dispatch scripts
               └─ patches/libero_benchmark.patch one-line LIBERO benchmark registration patch
lingbot-va/    modified LingBot-VA stack (fork of Robbyant/lingbot-va)
               ├─ wan_va/wm/                     watermark injection, MAP latent recovery, detection,
               │                                 robustness evals, per-episode score exporter
               ├─ wan_va/attacks/                compression surgery
               ├─ wan_va/lora.py + configs/      LoRA descendant fine-tuning
               └─ wan_va/tools/extract_latents.py
experiment/    the orchestration layer — START HERE
               ├─ README.md                      paper-section → asset → script map
               ├─ analysis/                      one wrapper per paper section (CPU only)
               ├─ rollout/                       GPU rollout wrappers per model × benchmark
               └─ train/                         LoRA descendant / compressed-checkpoint builders
results/       analysis implementations (scoring, tables, figures)
attack_c_data/ campaign sweep + recovery-metric scripts. Run these to (re)generate
               the rollouts and per-episode score CSVs from your own runs.
launchers/     GPU launchers per model × benchmark (eval, fine-tune, compression)
tools/         dataset converters (LIBERO HDF5 / RoboTwin → LeRobot) and aggregators
distill/       distillation-robustness study (Sec. on removability)
```

## Reproducing tables and figures

Every result is produced in two stages: **rollout** (GPU; run the policy +
verifier, save per-episode `.npz`, export per-episode score CSVs) and
**analysis** (CPU; turn those CSVs into the paper's tables and figures).

This release ships only the code — the per-episode score CSVs from our own
rollouts are **not** included. Regenerate them by running the rollout stage
(see below), which writes the CSVs under `attack_c_data/per_episode_scores*/`.
Once the scores exist, the analysis stage runs on CPU in minutes:

```bash
cd /workspace/vla/experiment
bash run.sh list                        # every section -> wrapper -> paper section
bash analysis/sec6.3_verification.sh    # main results: tab_main, tab_utility, attack figs
bash analysis/sec6.5_identification.sh  # closed-set CMC + open-set DIR@FAR
```

Outputs (`.tex`, `.pdf`, `.csv`) land in `results/` and `paper/`. Each analysis
wrapper reads the per-episode scores (and, for spectrum / recovery / uniqueness
/ unforgeability, the raw rollout `.npz` dumps) produced by the corresponding
`rollout/` wrapper — run that first.

## Full pipeline — GPU rollouts

Every result is produced in two stages: **rollout** (GPU; run the policy +
verifier, save per-episode `.npz`) and **analysis** (CPU; the step above).
Rollout wrappers are in `experiment/rollout/`, e.g.:

```bash
bash rollout/openpi_verification.sh        # pi0.5 / LIBERO verification rollouts
bash rollout/lingbot_attacks.sh robotwin   # LingBot / RoboTwin under output attacks
bash rollout/export_scores.sh              # rollout .npz -> per-episode CSVs
```

Training the watermarked models, LoRA descendants, and adaptive attacks:

```bash
bash train/openpi_lora.sh libero                       # pi0.5 LoRA descendants
bash train/lingbot_lora.sh robotwin                    # LingBot LoRA descendants
bash train/build_compressed_ckpt.sh openpi prune SRC DST   # pruning/quantization surgery
bash /workspace/vla/launchers/run_attack_d_jax_sweep.sh    # adaptive Attack-D sweep (pi0.5)
```

### Environment

The two stacks have separate environments; see `openpi/README.md` and
`lingbot-va/README.md` for their upstream setup, then:

- **openpi**: `PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src`
- **lingbot**: `PYTHONPATH=/workspace/vla/lingbot-va` (plus a LeRobot 0.3.3 checkout)
- LIBERO: clone upstream LIBERO into `openpi/third_party/libero` and apply
  `openpi/patches/libero_benchmark.patch` (registers the held-out task split).
- RoboTwin rollouts need a Vulkan-capable GPU node; LIBERO runs headless with
  `MUJOCO_GL=osmesa` or EGL.
- `launchers/node_env.sh` shows the full set of env vars our campaign nodes used.

## What is not included

Model checkpoints (base, watermarked, descendants), rollout videos, raw
`.npz` rollout dumps, and the per-episode score CSVs from our rollouts are not
included in this repository. Base checkpoints download from the upstream model
zoos (see each stack's README); all fine-tuning configs, rollout launchers, and
score exporters to regenerate the rest are included.

Paths were normalized to the canonical roots `/workspace/vla` (code) and
`/workspace/vla_out` (large outputs). Some launchers hard-code GPU counts/IDs —
adjust to your hardware.

## Licenses

`openpi/` and `lingbot-va/` retain their upstream Apache-2.0 licenses (see
`LICENSE` files inside each). Our additions in this repository are released
under the same terms.
