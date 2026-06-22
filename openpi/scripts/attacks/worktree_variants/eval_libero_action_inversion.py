#!/usr/bin/env python3
"""Evaluate internal watermark presence by inverting full LIBERO action chunks back to sampler noise."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import argparse
import copy
import csv
import dataclasses
import gc
import json
import logging
import pathlib
import sys
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.models import model as model_lib  # noqa: E402
from openpi.models import pi0 as jax_pi0  # noqa: E402
from openpi.models_pytorch import pi0_pytorch  # noqa: E402
from openpi.wm import (  # noqa: E402
    ChannelObservation,
    FMChannelSolver,
    FMChannelSolverConfig,
    FMLatentMAPConfig,
    FMLatentMAPSolver,
    FMLatentPosteriorConfig,
    FMLatentPosteriorSampler,
)
from scripts import eval_libero_internal_watermark as online_eval  # noqa: E402


@dataclasses.dataclass(frozen=True)
class InversionChunkTrace:
    chunk_index: int
    executed_steps: int
    reference: np.ndarray
    recovered_noise: np.ndarray
    injected_noise: np.ndarray
    raw_actions: np.ndarray
    observed_actions: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    selected: bool = True
    prompt: str = ""
    observation_state: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    observation_image: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.uint8)
    )
    observation_wrist_image: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.uint8)
    )
    recovered_noise_by_step: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    map_restart_recovered_noise: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32)
    )
    map_restart_energies: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    map_best_restart_index: int = -1
    posterior_recovered_noise_samples: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32)
    )
    posterior_recovered_noise_mean: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    posterior_recovered_noise_std: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    posterior_restart_energies: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    posterior_best_energy: float = float("nan")
    posterior_best_restart_index: int = -1
    posterior_init_mode: str = ""
    posterior_chain_init: str = ""


@dataclasses.dataclass(frozen=True)
class EpisodeScoreRecord:
    task_id: int
    episode_idx: int
    eval_mode: str
    variant: str
    episode_score: float
    recovery_rms: float
    selected_window_count: int
    chunk_scores: np.ndarray
    episode_score_std: float = 0.0
    episode_score_q05: float = 0.0
    episode_score_q95: float = 0.0
    posterior_sample_count: int = 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=str, default="pi05_libero")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--detector-config-name", type=str, default=None)
    parser.add_argument("--detector-checkpoint-dir", type=str, default=None)
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--num-trials-per-task", type=int, default=2)
    parser.add_argument("--save-rollout-dir", type=pathlib.Path, default=None)
    parser.add_argument("--save-report-dir", type=pathlib.Path, default=None)
    parser.add_argument("--resume-from-rollouts", action="store_true")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--eval-mode", choices=("task_rollout", "probe_verification"), default="task_rollout")
    parser.add_argument("--probe-duration-sec", type=float, default=10.0)
    parser.add_argument("--probe-pattern", choices=("axis_sweep", "circle", "lissajous"), default="axis_sweep")
    parser.add_argument("--probe-amplitude", type=float, default=0.04)
    parser.add_argument("--probe-axis-mode", choices=("x", "xy", "xz", "yz", "xyz"), default="xyz")
    parser.add_argument("--probe-gripper-mode", choices=("hold_open", "hold_closed", "hold_current"), default="hold_open")
    parser.add_argument("--probe-replan-interval", type=int, default=5)
    parser.add_argument("--probe-speed-scale", type=float, default=0.35)
    parser.add_argument(
        "--probe-prompt",
        type=str,
        default="perform a short low-speed verification sweep in free space",
    )
    parser.add_argument("--probe-settle-steps", type=int, default=12)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--secret-key", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--freq-min-hz", type=float, default=1.0)
    parser.add_argument("--freq-max-hz", type=float, default=2.0)
    parser.add_argument("--n-tones", type=int, default=4)
    parser.add_argument("--reference-mode", choices=("bandpass", "gaussian"), default="bandpass")
    parser.add_argument(
        "--chunk-selection-strategy",
        choices=("periodic", "fixed_slots", "stateful_online"),
        default="periodic",
    )
    parser.add_argument("--chunk-selection-period", type=int, default=1)
    parser.add_argument("--chunk-selection-count", type=int, default=1)
    parser.add_argument("--chunk-selection-total-slots", type=int, default=None)
    parser.add_argument("--max-score-windows", type=int, default=None)
    parser.add_argument("--window-aggregator", choices=("sum", "mean"), default="sum")
    parser.add_argument("--score-step-scope", choices=("executed", "full_chunk"), default="executed")
    parser.add_argument("--detector", choices=("cosine", "dot", "mse", "coherence", "wmf", "ace"), default="cosine")
    parser.add_argument("--null-decoy-count", type=int, default=32)
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num-inversion-steps", type=int, default=10)
    parser.add_argument("--save-recovered-noise-cache-steps", type=int, nargs="*", default=[])
    parser.add_argument(
        "--inversion-method",
        choices=("reverse", "reverse_refine"),
        default="reverse",
    )
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--refinement-learning-rate", type=float, default=0.05)
    parser.add_argument("--refinement-latent-l2", type=float, default=1e-4)
    parser.add_argument("--refinement-init-l2", type=float, default=1e-3)
    parser.add_argument("--fm-channel-inverse", action="store_true")
    parser.add_argument("--fm-full-latent-map", action="store_true")
    parser.add_argument("--full-map-no-warm-start", action="store_true")
    parser.add_argument("--fm-latent-map", action="store_true")
    parser.add_argument("--fm-latent-posterior", action="store_true")
    parser.add_argument("--obs-sigma", type=float, default=1e-4)
    parser.add_argument("--fm-guide-scale", type=float, default=0.5)
    parser.add_argument("--fm-guide-schedule", choices=("const", "linear_decay"), default="linear_decay")
    parser.add_argument("--latent-map-iters", type=int, default=100)
    parser.add_argument("--latent-map-lr", type=float, default=1e-1)
    parser.add_argument("--latent-prior-weight", type=float, default=1.0)
    parser.add_argument("--map-num-starts", type=int, default=1)
    parser.add_argument("--map-random-seed", type=int, default=0)
    parser.add_argument("--posterior-step-size", type=float, default=1e-3)
    parser.add_argument("--posterior-burnin", type=int, default=100)
    parser.add_argument("--posterior-thinning", type=int, default=50)
    parser.add_argument("--posterior-num-samples", type=int, default=8)
    parser.add_argument("--posterior-map-tether-weight", type=float, default=1.0)
    parser.add_argument("--posterior-grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--latent-init-from-bridge", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_tasks <= 0:
        raise ValueError("num_tasks must be > 0.")
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be > 0.")
    if args.resume_from_rollouts and args.save_rollout_dir is None:
        raise ValueError("--resume-from-rollouts requires --save-rollout-dir.")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be > 0.")
    if args.max_rollout_steps is not None and args.max_rollout_steps <= 0:
        raise ValueError("max_rollout_steps must be > 0 when provided.")
    if args.resize_size <= 0:
        raise ValueError("resize_size must be > 0.")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be >= 0.")
    if args.probe_duration_sec <= 0:
        raise ValueError("probe_duration_sec must be > 0.")
    if args.probe_replan_interval <= 0:
        raise ValueError("probe_replan_interval must be > 0.")
    if args.probe_settle_steps < 0:
        raise ValueError("probe_settle_steps must be >= 0.")
    if args.probe_amplitude < 0:
        raise ValueError("probe_amplitude must be >= 0.")
    if args.probe_speed_scale <= 0:
        raise ValueError("probe_speed_scale must be > 0.")
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("target_fpr must be in [0, 1).")
    if "://" not in args.checkpoint_dir:
        checkpoint_path = pathlib.Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_path}")
    if args.detector_checkpoint_dir is not None and "://" not in args.detector_checkpoint_dir:
        detector_checkpoint_path = pathlib.Path(args.detector_checkpoint_dir)
        if not detector_checkpoint_path.exists():
            raise FileNotFoundError(f"detector_checkpoint_dir does not exist: {detector_checkpoint_path}")
    if args.num_inversion_steps <= 0:
        raise ValueError("num_inversion_steps must be > 0.")
    if any(int(step) <= 0 for step in args.save_recovered_noise_cache_steps):
        raise ValueError("save_recovered_noise_cache_steps must contain only positive integers.")
    if args.null_decoy_count <= 0:
        raise ValueError("null_decoy_count must be > 0.")
    if args.subspace_rank is not None and args.subspace_rank <= 0:
        raise ValueError("subspace_rank must be > 0 when provided.")
    if args.refinement_steps < 0:
        raise ValueError("refinement_steps must be >= 0.")
    if args.refinement_learning_rate <= 0:
        raise ValueError("refinement_learning_rate must be > 0.")
    if args.refinement_latent_l2 < 0:
        raise ValueError("refinement_latent_l2 must be >= 0.")
    if args.refinement_init_l2 < 0:
        raise ValueError("refinement_init_l2 must be >= 0.")
    if args.obs_sigma <= 0:
        raise ValueError("obs_sigma must be > 0.")
    if args.fm_guide_scale < 0:
        raise ValueError("fm_guide_scale must be >= 0.")
    if args.latent_map_iters <= 0:
        raise ValueError("latent_map_iters must be > 0.")
    if args.latent_map_lr <= 0:
        raise ValueError("latent_map_lr must be > 0.")
    if args.latent_prior_weight < 0:
        raise ValueError("latent_prior_weight must be >= 0.")
    if args.map_num_starts <= 0:
        raise ValueError("map_num_starts must be > 0.")
    if args.posterior_step_size <= 0:
        raise ValueError("posterior_step_size must be > 0.")
    if args.posterior_burnin < 0:
        raise ValueError("posterior_burnin must be >= 0.")
    if args.posterior_thinning <= 0:
        raise ValueError("posterior_thinning must be > 0.")
    if args.posterior_num_samples <= 0:
        raise ValueError("posterior_num_samples must be > 0.")
    if args.posterior_map_tether_weight < 0:
        raise ValueError("posterior_map_tether_weight must be >= 0.")
    if args.posterior_grad_clip_norm < 0:
        raise ValueError("posterior_grad_clip_norm must be >= 0.")
    active_latent_modes = sum(
        int(flag)
        for flag in (
            args.fm_channel_inverse,
            args.fm_full_latent_map,
            args.fm_latent_map,
            args.fm_latent_posterior,
        )
    )
    if active_latent_modes > 1:
        raise ValueError(
            "fm_channel_inverse, fm_full_latent_map, fm_latent_map, and fm_latent_posterior are mutually exclusive."
        )
    if args.chunk_selection_period <= 0:
        raise ValueError("chunk_selection_period must be > 0.")
    if args.chunk_selection_total_slots is not None and args.chunk_selection_total_slots <= 0:
        raise ValueError("chunk_selection_total_slots must be > 0 when provided.")
    if args.chunk_selection_strategy == "periodic":
        if not (0 <= args.chunk_selection_count <= args.chunk_selection_period):
            raise ValueError("chunk_selection_count must be in [0, chunk_selection_period].")
    elif args.chunk_selection_strategy == "fixed_slots":
        if args.chunk_selection_total_slots is None:
            raise ValueError("chunk_selection_total_slots is required when chunk_selection_strategy='fixed_slots'.")
        if not (0 <= args.chunk_selection_count <= args.chunk_selection_total_slots):
            raise ValueError("chunk_selection_count must be in [0, chunk_selection_total_slots].")
    elif args.chunk_selection_count < 0:
        raise ValueError("chunk_selection_count must be >= 0.")
    if args.max_score_windows is not None and args.max_score_windows <= 0:
        raise ValueError("max_score_windows must be > 0 when provided.")


def _probe_total_steps(*, duration_sec: float, sample_rate_hz: float) -> int:
    return max(1, int(round(float(duration_sec) * float(sample_rate_hz))))


def _resolve_task_prompt(task_description: str, *, eval_mode: str, probe_prompt: str) -> str:
    return str(probe_prompt) if eval_mode == "probe_verification" else str(task_description)


def _pairwise_accuracy(marked_scores: np.ndarray, plain_scores: np.ndarray) -> float:
    marked_scores = np.asarray(marked_scores, dtype=np.float32)
    plain_scores = np.asarray(plain_scores, dtype=np.float32)
    count = min(marked_scores.size, plain_scores.size)
    if count == 0:
        return float("nan")
    return float(np.mean(marked_scores[:count] > plain_scores[:count]))


def _tpr_at_fpr(marked_scores: np.ndarray, plain_scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    threshold = online_eval._calibrate_threshold_for_target_fpr(np.asarray(plain_scores, dtype=np.float32), target_fpr)
    return online_eval._binary_metrics(marked_scores, plain_scores, threshold)


def _chunk_scores_for_episode(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    score_step_scope: str,
    max_windows: int | None,
) -> np.ndarray:
    base_detector = "cosine" if detector in {"wmf", "ace"} else detector
    scores = []
    for trace in _selected_score_traces(chunk_traces, max_windows=max_windows):
        score_steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        scores.append(
            _score_chunk_noise_similarity(
                np.asarray(trace.recovered_noise[:score_steps], dtype=np.float32),
                np.asarray(trace.reference[:score_steps], dtype=np.float32),
                detector=base_detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
            )
        )
    return np.asarray(scores, dtype=np.float32)


def _variant_score_vectors(records: list[EpisodeScoreRecord], *, eval_mode: str) -> tuple[np.ndarray, np.ndarray]:
    mode_records = [record for record in records if record.eval_mode == eval_mode]
    plain_scores = np.asarray(
        [record.episode_score for record in mode_records if record.variant == "plain"],
        dtype=np.float32,
    )
    marked_scores = np.asarray(
        [record.episode_score for record in mode_records if record.variant == "watermarked"],
        dtype=np.float32,
    )
    return plain_scores, marked_scores


def _score_variance_stats(records: list[EpisodeScoreRecord], *, variant: str) -> dict[str, float]:
    variant_records = [record for record in records if record.variant == variant]
    episode_scores = np.asarray([record.episode_score for record in variant_records], dtype=np.float32)
    chunk_scores = np.concatenate(
        [record.chunk_scores.astype(np.float32, copy=False) for record in variant_records if record.chunk_scores.size > 0],
        axis=0,
    ) if variant_records else np.zeros((0,), dtype=np.float32)
    task_means = []
    for task_id in sorted({record.task_id for record in variant_records}):
        task_values = [record.episode_score for record in variant_records if record.task_id == task_id]
        task_means.append(float(np.mean(task_values)))
    task_means_np = np.asarray(task_means, dtype=np.float32)
    return {
        f"{variant}_episode_var": float(np.var(chunk_scores if chunk_scores.size else episode_scores)) if variant_records else 0.0,
        f"{variant}_task_offset_var": float(np.var(task_means_np)) if task_means else 0.0,
        f"{variant}_recovery_rms_mean": float(
            np.mean([record.recovery_rms for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
        f"chunk_selected_window_mean_{variant}": float(
            np.mean([record.selected_window_count for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
        f"{variant}_episode_score_std_mean": float(
            np.mean([record.episode_score_std for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
        f"{variant}_episode_score_q05_mean": float(
            np.mean([record.episode_score_q05 for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
        f"{variant}_episode_score_q95_mean": float(
            np.mean([record.episode_score_q95 for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
        f"{variant}_posterior_sample_count_mean": float(
            np.mean([record.posterior_sample_count for record in variant_records], dtype=np.float32)
        ) if variant_records else 0.0,
    }


def _paired_episode_deltas(records: list[EpisodeScoreRecord], *, eval_mode: str) -> np.ndarray:
    paired: dict[tuple[int, int], dict[str, float]] = {}
    for record in records:
        if record.eval_mode != eval_mode:
            continue
        paired.setdefault((record.task_id, record.episode_idx), {})
        paired[(record.task_id, record.episode_idx)][record.variant] = float(record.episode_score)
    deltas = []
    for values in paired.values():
        if "plain" in values and "watermarked" in values:
            deltas.append(values["watermarked"] - values["plain"])
    return np.asarray(deltas, dtype=np.float32)


def _summarize_eval_mode(records: list[EpisodeScoreRecord], *, eval_mode: str) -> dict[str, float | str]:
    plain_scores, marked_scores = _variant_score_vectors(records, eval_mode=eval_mode)
    deltas = _paired_episode_deltas(records, eval_mode=eval_mode)
    tpr_1, _ = _tpr_at_fpr(marked_scores, plain_scores, 0.01)
    tpr_10, _ = _tpr_at_fpr(marked_scores, plain_scores, 0.10)
    summary: dict[str, float | str] = {
        "eval_mode": eval_mode,
        "episode_count_plain": float(plain_scores.size),
        "episode_count_watermarked": float(marked_scores.size),
        "roc_auc": float(online_eval._roc_auc(marked_scores, plain_scores)) if plain_scores.size and marked_scores.size else 0.0,
        "pairwise_wm_gt_plain_accuracy": _pairwise_accuracy(marked_scores, plain_scores),
        "tpr_at_1pct_fpr": float(tpr_1),
        "tpr_at_10pct_fpr": float(tpr_10),
        "plain_score_mean": float(np.mean(plain_scores)) if plain_scores.size else 0.0,
        "watermarked_score_mean": float(np.mean(marked_scores)) if marked_scores.size else 0.0,
        "plain_score_std": float(np.std(plain_scores)) if plain_scores.size else 0.0,
        "watermarked_score_std": float(np.std(marked_scores)) if marked_scores.size else 0.0,
        "wm_minus_plain_mean": float(np.mean(deltas)) if deltas.size else 0.0,
    }
    summary.update(_score_variance_stats([record for record in records if record.eval_mode == eval_mode], variant="plain"))
    summary.update(
        _score_variance_stats([record for record in records if record.eval_mode == eval_mode], variant="watermarked")
    )
    return summary


def _record_to_row(record: EpisodeScoreRecord) -> dict[str, str | int | float]:
    return {
        "task_id": record.task_id,
        "episode_idx": record.episode_idx,
        "eval_mode": record.eval_mode,
        "variant": record.variant,
        "episode_score": record.episode_score,
        "episode_score_std": record.episode_score_std,
        "episode_score_q05": record.episode_score_q05,
        "episode_score_q95": record.episode_score_q95,
        "posterior_sample_count": record.posterior_sample_count,
        "recovery_rms": record.recovery_rms,
        "selected_window_count": record.selected_window_count,
        "chunk_score_mean": float(np.mean(record.chunk_scores)) if record.chunk_scores.size else 0.0,
        "chunk_score_std": float(np.std(record.chunk_scores)) if record.chunk_scores.size else 0.0,
        "chunk_scores": json.dumps(record.chunk_scores.astype(float).tolist()),
    }


def _write_report_artifacts(
    *,
    report_dir: pathlib.Path,
    args: argparse.Namespace,
    records: list[EpisodeScoreRecord],
    summary: dict[str, float | str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"summary_{args.eval_mode}.json"
    episodes_path = report_dir / f"episode_scores_{args.eval_mode}.csv"
    report_path = report_dir / f"report_{args.eval_mode}.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    rows = [_record_to_row(record) for record in records if record.eval_mode == args.eval_mode]
    with episodes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "episode_idx",
                "eval_mode",
                "variant",
                "episode_score",
                "episode_score_std",
                "episode_score_q05",
                "episode_score_q95",
                "posterior_sample_count",
                "recovery_rms",
                "selected_window_count",
                "chunk_score_mean",
                "chunk_score_std",
                "chunk_scores",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        f"# Internal Watermark {args.eval_mode} Report",
        "",
        "## Command Configuration",
        "",
        f"- eval_mode: `{args.eval_mode}`",
        f"- detector: `{args.detector}`",
        f"- reference_mode: `{args.reference_mode}`",
        f"- beta: `{args.beta}`",
        f"- task_suite: `{args.task_suite_name}`",
        "",
        "## Metrics",
        "",
        f"- roc_auc: `{float(summary['roc_auc']):.4f}`",
        f"- pairwise_wm_gt_plain_accuracy: `{float(summary['pairwise_wm_gt_plain_accuracy']):.4f}`",
        f"- tpr_at_1pct_fpr: `{float(summary['tpr_at_1pct_fpr']):.4f}`",
        f"- tpr_at_10pct_fpr: `{float(summary['tpr_at_10pct_fpr']):.4f}`",
        f"- wm_minus_plain_mean: `{float(summary['wm_minus_plain_mean']):.4f}`",
        f"- plain_recovery_rms_mean: `{float(summary['plain_recovery_rms_mean']):.6f}`",
        f"- watermarked_recovery_rms_mean: `{float(summary['watermarked_recovery_rms_mean']):.6f}`",
        "",
        "## Offset Analysis",
        "",
        f"- plain_task_offset_var: `{float(summary['plain_task_offset_var']):.6f}`",
        f"- watermarked_task_offset_var: `{float(summary['watermarked_task_offset_var']):.6f}`",
        f"- plain_episode_var: `{float(summary['plain_episode_var']):.6f}`",
        f"- watermarked_episode_var: `{float(summary['watermarked_episode_var']):.6f}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    other_mode = "probe_verification" if args.eval_mode == "task_rollout" else "task_rollout"
    other_summary_path = report_dir / f"summary_{other_mode}.json"
    if other_summary_path.exists():
        with other_summary_path.open(encoding="utf-8") as f:
            other_summary = json.load(f)
        comparison = {
            "task_rollout": summary if args.eval_mode == "task_rollout" else other_summary,
            "probe_verification": summary if args.eval_mode == "probe_verification" else other_summary,
        }
        with (report_dir / "summary_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, sort_keys=True)
            f.write("\n")
        task_summary = comparison["task_rollout"]
        probe_summary = comparison["probe_verification"]
        comparison_lines = [
            "# Internal Watermark Probe Comparison",
            "",
            "## Detection",
            "",
            f"- task_rollout roc_auc: `{float(task_summary['roc_auc']):.4f}`",
            f"- probe_verification roc_auc: `{float(probe_summary['roc_auc']):.4f}`",
            f"- task_rollout tpr_at_1pct_fpr: `{float(task_summary['tpr_at_1pct_fpr']):.4f}`",
            f"- probe_verification tpr_at_1pct_fpr: `{float(probe_summary['tpr_at_1pct_fpr']):.4f}`",
            "",
            "## Stability",
            "",
            f"- task_rollout plain_task_offset_var: `{float(task_summary['plain_task_offset_var']):.6f}`",
            f"- probe_verification plain_task_offset_var: `{float(probe_summary['plain_task_offset_var']):.6f}`",
            f"- task_rollout plain_recovery_rms_mean: `{float(task_summary['plain_recovery_rms_mean']):.6f}`",
            f"- probe_verification plain_recovery_rms_mean: `{float(probe_summary['plain_recovery_rms_mean']):.6f}`",
        ]
        (report_dir / "comparison_report.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")


def _release_policy(policy) -> None:
    del policy
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        jax.clear_caches()
    except Exception:
        pass


def _prepare_policy_inputs(policy, obs: dict) -> tuple[model_lib.Observation, dict, online_eval.wm.WatermarkContext | None]:
    watermark_context = policy._extract_watermark_context(obs)
    obs = policy._strip_runtime_metadata(obs)
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)
    if policy._is_pytorch_model:
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...], inputs)
    else:
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)
    return observation, inputs, watermark_context


def _sample_raw_actions(
    policy,
    obs: dict,
    *,
    noise: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    observation, inputs, watermark_context = _prepare_policy_inputs(policy, obs)
    prepared_noise = policy._prepare_internal_noise(
        noise,
        batch_size=inputs["state"].shape[0],
        sample_rng_or_pytorch_device=policy._pytorch_device if policy._is_pytorch_model else None,
        noise_rng=None,
        context=watermark_context,
    )
    sample_kwargs = dict(policy._sample_kwargs)
    sample_kwargs["noise"] = prepared_noise
    sample_arg = policy._pytorch_device if policy._is_pytorch_model else jax.random.key(0)
    raw_actions_batch = policy._sample_actions(sample_arg, observation, **sample_kwargs)
    if policy._is_pytorch_model:
        raw_actions = np.asarray(raw_actions_batch[0].detach().cpu(), dtype=np.float32)
        outputs = {
            "state": np.asarray(inputs["state"][0].detach().cpu(), dtype=np.float32),
            "actions": raw_actions.copy(),
        }
        injected_noise = np.asarray(prepared_noise[0].detach().cpu(), dtype=np.float32)
    else:
        raw_actions = np.asarray(raw_actions_batch[0], dtype=np.float32)
        outputs = {
            "state": np.asarray(inputs["state"][0], dtype=np.float32),
            "actions": raw_actions.copy(),
        }
        injected_noise = np.asarray(prepared_noise[0], dtype=np.float32)
    transformed = policy._output_transform(outputs)
    transformed["raw_actions"] = raw_actions
    return transformed, injected_noise


def _prepare_jax_sampling_context(policy, observation: model_lib.Observation):
    model = policy._model
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(preprocessed)
    prefix_attn_mask = jax_pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return preprocessed, prefix_mask, kv_cache


def _prepare_pytorch_sampling_context(policy, observation: model_lib.Observation):
    model = policy._model
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return state, prefix_pad_masks, past_key_values


def _prepare_pytorch_channel_observation_context(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
) -> tuple[tuple[torch.Tensor, torch.Tensor, Any], torch.Tensor, torch.Tensor]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    model_inputs = _prepare_pytorch_sampling_context(policy, observation)
    y_obs_np = _normalize_channel_observation(policy, env_action_chunk)
    y_obs = torch.from_numpy(y_obs_np).to(policy._pytorch_device, dtype=torch.float32)[None, ...]
    time_grid = torch.from_numpy(_make_sample_time_grid(int(policy._sample_kwargs.get("num_steps", 10)))).to(
        policy._pytorch_device,
        dtype=torch.float32,
    )
    return model_inputs, y_obs, time_grid


def _prepare_jax_channel_observation_context(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
) -> tuple[tuple[model_lib.Observation, jax.Array, Any], jax.Array, jax.Array]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    model_inputs = _prepare_jax_sampling_context(policy, observation)
    y_obs = jnp.asarray(_normalize_channel_observation(policy, env_action_chunk), dtype=jnp.float32)[None, ...]
    time_grid = jnp.asarray(_make_sample_time_grid(int(policy._sample_kwargs.get("num_steps", 10))), dtype=jnp.float32)
    return model_inputs, y_obs, time_grid


def _prepare_pytorch_full_action_observation_context(
    policy,
    *,
    obs: dict,
    raw_action_chunk: np.ndarray,
) -> tuple[tuple[torch.Tensor, torch.Tensor, Any], torch.Tensor, torch.Tensor]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    model_inputs = _prepare_pytorch_sampling_context(policy, observation)
    y_obs = torch.from_numpy(np.asarray(raw_action_chunk, dtype=np.float32)).to(policy._pytorch_device, dtype=torch.float32)[
        None, ...
    ]
    time_grid = torch.from_numpy(_make_sample_time_grid(int(policy._sample_kwargs.get("num_steps", 10)))).to(
        policy._pytorch_device,
        dtype=torch.float32,
    )
    return model_inputs, y_obs, time_grid


def _prepare_jax_full_action_observation_context(
    policy,
    *,
    obs: dict,
    raw_action_chunk: np.ndarray,
) -> tuple[tuple[model_lib.Observation, jax.Array, Any], jax.Array, jax.Array]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    model_inputs = _prepare_jax_sampling_context(policy, observation)
    y_obs = jnp.asarray(np.asarray(raw_action_chunk, dtype=np.float32), dtype=jnp.float32)[None, ...]
    time_grid = jnp.asarray(_make_sample_time_grid(int(policy._sample_kwargs.get("num_steps", 10))), dtype=jnp.float32)
    return model_inputs, y_obs, time_grid

def _make_sample_time_grid(num_steps: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, int(num_steps) + 1, dtype=np.float32)


def _channel_latent_recovery_enabled(args: argparse.Namespace) -> bool:
    return bool(args.fm_latent_map or args.fm_latent_posterior)


def _full_action_latent_recovery_enabled(args: argparse.Namespace) -> bool:
    return bool(args.fm_full_latent_map)


def _fm_guide_weight(*, step_idx: int, step_count: int, guide_scale: float, guide_schedule: str) -> float:
    if guide_schedule == "const":
        return float(guide_scale)
    if guide_schedule != "linear_decay":
        raise ValueError(f"Unsupported fm guide schedule: {guide_schedule!r}")
    frac = step_idx / max(step_count - 1, 1)
    return float(guide_scale) * (1.0 - frac)


def _normalize_channel_observation(policy, env_action_chunk: np.ndarray) -> np.ndarray:
    observed = np.asarray(env_action_chunk, dtype=np.float32)
    try:
        from openpi import transforms as transforms_lib
    except ModuleNotFoundError:
        return observed

    transforms = getattr(policy._output_transform, "transforms", ())
    unnormalize = next(
        (transform for transform in transforms if isinstance(transform, transforms_lib.Unnormalize)),
        None,
    )
    if unnormalize is None or unnormalize.norm_stats is None:
        return observed
    normalizer = transforms_lib.Normalize(
        unnormalize.norm_stats,
        use_quantiles=bool(unnormalize.use_quantiles),
    )
    normalized = normalizer({"actions": observed.copy()})
    return np.asarray(normalized["actions"], dtype=np.float32)


def _action_stats_from_transform(transform: Any) -> Any | None:
    norm_stats = getattr(transform, "norm_stats", None)
    if norm_stats is None:
        return None
    if hasattr(norm_stats, "mean") and hasattr(norm_stats, "std"):
        return norm_stats
    if isinstance(norm_stats, dict) and "actions" in norm_stats:
        return norm_stats["actions"]
    return None


def _pad_numpy_stat(value: Any, dim: int, *, fill_value: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] >= dim:
        return array[..., :dim].astype(np.float32)
    pad_shape = (*array.shape[:-1], dim - array.shape[-1])
    pad = np.full(pad_shape, fill_value, dtype=np.float32)
    return np.concatenate((array, pad), axis=-1).astype(np.float32)


def _pad_torch_stat(value: Any, dim: int, *, fill_value: float, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.shape[-1] >= dim:
        return tensor[..., :dim]
    pad_shape = (*tensor.shape[:-1], dim - tensor.shape[-1])
    pad = torch.full(pad_shape, fill_value, dtype=like.dtype, device=like.device)
    return torch.cat((tensor, pad), dim=-1)


def _pad_jax_stat(value: Any, dim: int, *, fill_value: float, like: jax.Array) -> jax.Array:
    tensor = jnp.asarray(value, dtype=like.dtype)
    if tensor.shape[-1] >= dim:
        return tensor[..., :dim]
    pad_shape = (*tensor.shape[:-1], dim - tensor.shape[-1])
    pad = jnp.full(pad_shape, fill_value, dtype=like.dtype)
    return jnp.concatenate((tensor, pad), axis=-1)


def _project_raw_actions_to_env_torch(policy, state: torch.Tensor, raw_actions: torch.Tensor, *, action_dim: int = 7) -> torch.Tensor:
    actions = raw_actions
    for transform in getattr(getattr(policy, "_output_transform", None), "transforms", ()) or ():
        name = type(transform).__name__
        stats = _action_stats_from_transform(transform)
        if stats is not None:
            if bool(getattr(transform, "use_quantiles", False)):
                q01 = _pad_torch_stat(stats.q01, actions.shape[-1], fill_value=0.0, like=actions)
                q99 = _pad_torch_stat(stats.q99, actions.shape[-1], fill_value=1.0, like=actions)
                actions = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            else:
                mean = _pad_torch_stat(stats.mean, actions.shape[-1], fill_value=0.0, like=actions)
                std = _pad_torch_stat(stats.std, actions.shape[-1], fill_value=1.0, like=actions)
                actions = actions * (std + 1e-6) + mean
        elif getattr(transform, "mask", None) is not None or name == "AbsoluteActions":
            mask = getattr(transform, "mask", None)
            if mask is None:
                continue
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=actions.device)
            dims = int(mask_tensor.shape[-1])
            state_prefix = state[..., :dims]
            offset = torch.where(mask_tensor, state_prefix, torch.zeros_like(state_prefix))
            actions = actions.clone()
            actions[..., :dims] = actions[..., :dims] + offset[:, None, :]
        elif name == "LiberoOutputs":
            actions = actions[..., : int(action_dim)]
    return actions[..., : int(action_dim)]


def _project_raw_actions_to_env_jax(policy, state: jax.Array, raw_actions: jax.Array, *, action_dim: int = 7) -> jax.Array:
    actions = raw_actions
    for transform in getattr(getattr(policy, "_output_transform", None), "transforms", ()) or ():
        name = type(transform).__name__
        stats = _action_stats_from_transform(transform)
        if stats is not None:
            if bool(getattr(transform, "use_quantiles", False)):
                q01 = _pad_jax_stat(stats.q01, actions.shape[-1], fill_value=0.0, like=actions)
                q99 = _pad_jax_stat(stats.q99, actions.shape[-1], fill_value=1.0, like=actions)
                actions = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            else:
                mean = _pad_jax_stat(stats.mean, actions.shape[-1], fill_value=0.0, like=actions)
                std = _pad_jax_stat(stats.std, actions.shape[-1], fill_value=1.0, like=actions)
                actions = actions * (std + 1e-6) + mean
        elif getattr(transform, "mask", None) is not None or name == "AbsoluteActions":
            mask = getattr(transform, "mask", None)
            if mask is None:
                continue
            mask_tensor = jnp.asarray(mask, dtype=bool)
            dims = int(mask_tensor.shape[-1])
            state_prefix = state[..., :dims]
            offset = jnp.where(mask_tensor, state_prefix, jnp.zeros_like(state_prefix))
            actions = actions.at[..., :dims].set(actions[..., :dims] + offset[:, None, :])
        elif name == "LiberoOutputs":
            actions = actions[..., : int(action_dim)]
    return actions[..., : int(action_dim)]


def _projection_state_from_model_inputs(policy, model_inputs):
    if policy._is_pytorch_model:
        return model_inputs[0]
    observation = model_inputs[0]
    return observation.state


def _complete_raw_actions_from_channel_observation_jax(
    policy,
    *,
    observation: model_lib.Observation,
    env_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    model_inputs = _prepare_jax_sampling_context(policy, observation)
    y_obs = jnp.asarray(_normalize_channel_observation(policy, env_action_chunk), dtype=jnp.float32)
    time_grid = jnp.asarray(_make_sample_time_grid(int(policy._sample_kwargs.get("num_steps", 10))), dtype=jnp.float32)
    raw_dim = int(getattr(policy._model, "action_dim", 32))
    horizon = int(y_obs.shape[0])
    x_t = jax.random.normal(jax.random.key(0), (1, horizon, raw_dim), dtype=jnp.float32)

    def loss_and_velocity(x_batch: jax.Array, time_value: jax.Array) -> tuple[jax.Array, jax.Array]:
        v_t, a_hat_t = policy._model.predict_velocity_and_endpoint(model_inputs, x_batch, time_value)
        obs_pred = a_hat_t[:, :, : y_obs.shape[-1]]
        obs_pred = jnp.where(jnp.isfinite(obs_pred), obs_pred, y_obs[None, ...])
        obs_loss = 0.5 * jnp.mean(jnp.square((obs_pred - y_obs[None, ...]) / float(args.obs_sigma)))
        return obs_loss, v_t

    value_and_grad = jax.value_and_grad(loss_and_velocity, has_aux=True)
    for step_idx in range(int(time_grid.shape[0]) - 1):
        time_value = time_grid[step_idx]
        next_time = time_grid[step_idx + 1]
        dt = next_time - time_value
        guide_weight = _fm_guide_weight(
            step_idx=step_idx,
            step_count=int(time_grid.shape[0]) - 1,
            guide_scale=float(args.fm_guide_scale),
            guide_schedule=str(args.fm_guide_schedule),
        )
        (_, v_t), grad_x = value_and_grad(x_t, time_value)
        v_t = jnp.nan_to_num(v_t)
        grad_x = jnp.nan_to_num(grad_x)
        step_sign = 1.0 if float(dt) >= 0.0 else -1.0
        v_corr = jnp.nan_to_num(v_t - step_sign * guide_weight * grad_x)
        x_next = x_t + dt * v_corr
        x_t = jnp.where(jnp.isfinite(x_next), x_next, x_t)

    _, a_final = policy._model.predict_velocity_and_endpoint(model_inputs, x_t, time_grid[-1])
    a_final = jnp.where(jnp.isfinite(a_final), a_final, x_t)
    a_final = jnp.nan_to_num(a_final)
    a_final = a_final.at[:, :, : y_obs.shape[-1]].set(y_obs[None, ...])
    return np.asarray(a_final[0], dtype=np.float32)


def _complete_raw_actions_from_channel_observation(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    if not policy._is_pytorch_model:
        return _complete_raw_actions_from_channel_observation_jax(
            policy,
            observation=observation,
            env_action_chunk=env_action_chunk,
            args=args,
        )

    model_inputs, y_obs, time_grid = _prepare_pytorch_channel_observation_context(
        policy,
        obs=obs,
        env_action_chunk=env_action_chunk,
    )
    obs_op = ChannelObservation(obs_sigma=float(args.obs_sigma))
    solver = FMChannelSolver(
        policy._model,
        obs_op,
        FMChannelSolverConfig(
            guide_scale=float(args.fm_guide_scale),
            guide_schedule=str(args.fm_guide_schedule),
            hard_overwrite_final=True,
        ),
    )
    completed = solver.complete(model_inputs=model_inputs, y_obs=y_obs, time_grid=time_grid)
    return np.asarray(completed[0].detach().cpu(), dtype=np.float32)


def _latent_trace_defaults(
    *,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_shape = tuple(int(dim) for dim in np.asarray(raw_action_chunk, dtype=np.float32).shape)
    posterior_sample_count = (
        int(getattr(args, "posterior_num_samples", 0))
        if bool(getattr(args, "fm_latent_posterior", False))
        else 0
    )
    return {
        "map_restart_recovered_noise": np.zeros((0, *raw_shape), dtype=np.float32),
        "map_restart_energies": np.zeros((0,), dtype=np.float32),
        "map_best_restart_index": -1,
        "posterior_recovered_noise_samples": np.zeros(
            (posterior_sample_count, *raw_shape),
            dtype=np.float32,
        ),
        "posterior_recovered_noise_mean": np.zeros(raw_shape, dtype=np.float32),
        "posterior_recovered_noise_std": np.zeros(raw_shape, dtype=np.float32),
        "posterior_restart_energies": np.zeros((0,), dtype=np.float32),
        "posterior_best_energy": float("nan"),
        "posterior_best_restart_index": -1,
        "posterior_init_mode": "",
        "posterior_chain_init": "",
    }


def _latent_bridge_warm_start(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | jax.Array, str]:
    completed_actions = _complete_raw_actions_from_channel_observation(
        policy,
        obs=obs,
        env_action_chunk=env_action_chunk,
        args=args,
    )
    z_init_np = _recover_noise_from_actions(
        policy,
        obs=obs,
        raw_actions=completed_actions,
        args=args,
    )
    if policy._is_pytorch_model:
        z_init = torch.from_numpy(z_init_np).to(policy._pytorch_device, dtype=torch.float32)[None, ...]
    else:
        z_init = jnp.asarray(z_init_np[None, ...], dtype=jnp.float32)
    return z_init, "bridge_old_reverse"


def _posterior_chain_init_mode(init_mode: str) -> str:
    if init_mode == "bridge_old_reverse":
        return "map_from_old_reverse"
    return "map"


def _map_restart_chain_init_mode(init_mode: str, *, best_restart_index: int) -> str:
    if int(best_restart_index) == 0 and init_mode == "bridge_old_reverse":
        return "map_from_old_reverse"
    return "map"


def _map_restart_seed(obs: dict, *, args: argparse.Namespace) -> int:
    seed = (
        np.uint32(getattr(args, "map_random_seed", 0))
        + np.uint32(int(obs.get("episode_nonce", 0)) * 1009)
        + np.uint32(int(obs.get("chunk_index", 0)) * 9176)
    )
    return int(seed)


def _build_map_restart_initial_latents(
    z_seed: np.ndarray,
    *,
    num_starts: int,
    seed: int,
) -> np.ndarray:
    z_seed = np.asarray(z_seed, dtype=np.float32)
    latents = [z_seed]
    if int(num_starts) <= 1:
        return np.asarray(latents, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(num_starts) - 1):
        latents.append(rng.standard_normal(size=z_seed.shape).astype(np.float32))
    return np.asarray(latents, dtype=np.float32)


def _posterior_map_tether_weight(args: argparse.Namespace) -> float:
    return float(getattr(args, "posterior_map_tether_weight", 0.0))


def _posterior_grad_clip_norm(args: argparse.Namespace) -> float:
    return float(getattr(args, "posterior_grad_clip_norm", 0.0))


def _clip_jax_grad_by_global_norm(grad: jax.Array, *, max_norm: float) -> jax.Array:
    if float(max_norm) <= 0.0:
        return grad
    flat = grad.reshape((grad.shape[0], -1))
    norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
    eps = jnp.asarray(jnp.finfo(grad.dtype).eps, dtype=grad.dtype)
    scales = jnp.minimum(1.0, float(max_norm) / jnp.maximum(norms, eps))
    return grad * scales.reshape((grad.shape[0],) + (1,) * (grad.ndim - 1))


def _run_pytorch_channel_latent_map_restarts(
    policy,
    *,
    model_inputs,
    y_obs: torch.Tensor,
    time_grid: torch.Tensor,
    z_seed: torch.Tensor | None,
    args: argparse.Namespace,
    rng_seed: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    obs_op = ChannelObservation(obs_sigma=float(args.obs_sigma))
    solver = FMLatentMAPSolver(
        policy._model,
        obs_op,
        FMLatentMAPConfig(
            num_iters=int(args.latent_map_iters),
            lr=float(args.latent_map_lr),
            obs_sigma=float(args.obs_sigma),
            prior_weight=float(args.latent_prior_weight),
        ),
    )

    def energy_fn(z_map: torch.Tensor) -> torch.Tensor:
        a_pred = policy._model.sample_actions_from_noise(model_inputs, z_map, time_grid)
        pred_obs = obs_op.apply(a_pred)
        obs_loss = 0.5 * torch.mean(torch.square((pred_obs - y_obs) / float(args.obs_sigma)))
        prior_loss = 0.5 * float(args.latent_prior_weight) * torch.mean(torch.square(z_map))
        return obs_loss + prior_loss

    restart_inits: list[torch.Tensor | None]
    if z_seed is not None:
        restart_inits = [
            torch.from_numpy(start[None, ...]).to(policy._pytorch_device, dtype=torch.float32)
            for start in _build_map_restart_initial_latents(
                np.asarray(z_seed[0].detach().cpu(), dtype=np.float32),
                num_starts=int(getattr(args, "map_num_starts", 1)),
                seed=int(rng_seed),
            )
        ]
    else:
        restart_inits = [None]
        if int(getattr(args, "map_num_starts", 1)) > 1:
            rng = np.random.default_rng(int(rng_seed))
            batch_size, horizon, _ = y_obs.shape
            raw_dim = int(getattr(policy._model, "action_dim", 32))
            for _ in range(int(getattr(args, "map_num_starts", 1)) - 1):
                restart_inits.append(
                    torch.from_numpy(rng.standard_normal(size=(batch_size, horizon, raw_dim)).astype(np.float32)).to(
                        policy._pytorch_device,
                        dtype=torch.float32,
                    )
                )

    z_maps: list[torch.Tensor] = []
    restart_energies: list[float] = []
    restart_latents: list[np.ndarray] = []
    for z_init in restart_inits:
        out = solver.solve(model_inputs=model_inputs, y_obs=y_obs, time_grid=time_grid, z_init=z_init)
        z_map = torch.nan_to_num(out["z_map"].detach())
        z_maps.append(z_map)
        restart_latents.append(np.asarray(z_map[0].cpu(), dtype=np.float32))
        restart_energies.append(float(energy_fn(z_map).detach().item()))
    energies = np.asarray(restart_energies, dtype=np.float32)
    best_index = int(np.argmin(energies)) if energies.size else -1
    return z_maps[best_index], np.asarray(restart_latents, dtype=np.float32), energies, best_index


def _run_jax_channel_latent_map_restarts(
    policy,
    *,
    model_inputs,
    y_obs: jax.Array,
    time_grid: jax.Array,
    z_seed: jax.Array,
    args: argparse.Namespace,
    rng_seed: int,
) -> tuple[jax.Array, np.ndarray, np.ndarray, int]:
    def energy_fn(z: jax.Array) -> jax.Array:
        a_pred = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)
        pred_obs = a_pred[:, :, : y_obs.shape[-1]]
        pred_obs = jnp.where(jnp.isfinite(pred_obs), pred_obs, y_obs)
        obs_loss = 0.5 * jnp.mean(jnp.square((pred_obs - y_obs) / float(args.obs_sigma)))
        prior_loss = 0.5 * float(args.latent_prior_weight) * jnp.mean(jnp.square(z))
        return obs_loss + prior_loss

    restart_inits = _build_map_restart_initial_latents(
        np.asarray(z_seed[0], dtype=np.float32),
        num_starts=int(getattr(args, "map_num_starts", 1)),
        seed=int(rng_seed),
    )
    z_maps: list[jax.Array] = []
    restart_latents: list[np.ndarray] = []
    restart_energies: list[float] = []
    for z_init_np in restart_inits:
        z_map = _optimize_latent_with_adam_jax(
            init_noise=jnp.asarray(z_init_np[None, ...], dtype=jnp.float32),
            loss_fn=energy_fn,
            num_steps=int(args.latent_map_iters),
            learning_rate=float(args.latent_map_lr),
        )
        z_map = jnp.nan_to_num(z_map)
        z_maps.append(z_map)
        restart_latents.append(np.asarray(z_map[0], dtype=np.float32))
        restart_energies.append(float(energy_fn(z_map)))
    energies = np.asarray(restart_energies, dtype=np.float32)
    best_index = int(np.argmin(energies)) if energies.size else -1
    return z_maps[best_index], np.asarray(restart_latents, dtype=np.float32), energies, best_index


def _recover_noise_from_full_action_latent(
    policy,
    *,
    obs: dict,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not policy._is_pytorch_model:
        return _recover_noise_from_full_action_latent_jax(
            policy,
            obs=obs,
            raw_action_chunk=raw_action_chunk,
            args=args,
        )

    model_inputs, y_obs, time_grid = _prepare_pytorch_full_action_observation_context(
        policy,
        obs=obs,
        raw_action_chunk=raw_action_chunk,
    )
    obs_op = ChannelObservation(
        channel_idx=tuple(range(int(y_obs.shape[-1]))),
        obs_sigma=float(args.obs_sigma),
    )
    latent_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
    use_random_init = bool(getattr(args, "full_map_no_warm_start", False))
    z_init = None
    init_mode = "random"
    if not use_random_init:
        z_init_np = _recover_noise_from_actions(
            policy,
            obs=obs,
            raw_actions=raw_action_chunk,
            args=args,
        )
        z_init = torch.from_numpy(z_init_np).to(policy._pytorch_device, dtype=torch.float32)[None, ...]
        init_mode = "old_reverse"
    latent_payload["posterior_init_mode"] = init_mode
    latent_payload["posterior_chain_init"] = init_mode

    solver = FMLatentMAPSolver(
        policy._model,
        obs_op,
        FMLatentMAPConfig(
            num_iters=int(args.latent_map_iters),
            lr=float(args.latent_map_lr),
            obs_sigma=float(args.obs_sigma),
            prior_weight=float(args.latent_prior_weight),
        ),
    )
    out = solver.solve(model_inputs=model_inputs, y_obs=y_obs, time_grid=time_grid, z_init=z_init)
    z_map = torch.nan_to_num(out["z_map"].detach())
    latent_payload["recovered_noise"] = np.asarray(z_map[0].cpu(), dtype=np.float32)
    latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_map[0].cpu(), dtype=np.float32)
    latent_payload["posterior_best_energy"] = float(out["final_obs_mse"])
    return latent_payload


def _recover_noise_from_full_action_latent_jax(
    policy,
    *,
    obs: dict,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_inputs, y_obs, time_grid = _prepare_jax_full_action_observation_context(
        policy,
        obs=obs,
        raw_action_chunk=raw_action_chunk,
    )
    latent_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
    raw_dim = int(getattr(policy._model, "action_dim", y_obs.shape[-1]))
    horizon = int(y_obs.shape[1])
    use_random_init = bool(getattr(args, "full_map_no_warm_start", False))
    init_mode = "random"
    if use_random_init:
        z_init = jax.random.normal(jax.random.key(0), (1, horizon, raw_dim), dtype=jnp.float32)
    else:
        z_init_np = _recover_noise_from_actions(
            policy,
            obs=obs,
            raw_actions=raw_action_chunk,
            args=args,
        )
        z_init = jnp.asarray(z_init_np[None, ...], dtype=jnp.float32)
        init_mode = "old_reverse"
    latent_payload["posterior_init_mode"] = init_mode
    latent_payload["posterior_chain_init"] = init_mode

    def energy_fn(z: jax.Array, *, prior_weight: float) -> jax.Array:
        a_pred = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)
        pred_obs = jnp.where(jnp.isfinite(a_pred), a_pred, y_obs)
        obs_loss = 0.5 * jnp.mean(jnp.square((pred_obs - y_obs) / float(args.obs_sigma)))
        prior_loss = 0.5 * float(prior_weight) * jnp.mean(jnp.square(z))
        return obs_loss + prior_loss

    z_map = _optimize_latent_with_adam_jax(
        init_noise=z_init,
        loss_fn=lambda z: energy_fn(z, prior_weight=float(args.latent_prior_weight)),
        num_steps=int(args.latent_map_iters),
        learning_rate=float(args.latent_map_lr),
    )
    z_map = jnp.nan_to_num(z_map)
    latent_payload["recovered_noise"] = np.asarray(z_map[0], dtype=np.float32)
    latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_map[0], dtype=np.float32)
    latent_payload["posterior_best_energy"] = float(energy_fn(z_map, prior_weight=float(args.latent_prior_weight)))
    return latent_payload


def _recover_noise_from_channel_observation_latent(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not policy._is_pytorch_model:
        return _recover_noise_from_channel_observation_latent_jax(
            policy,
            obs=obs,
            env_action_chunk=env_action_chunk,
            raw_action_chunk=raw_action_chunk,
            args=args,
        )

    model_inputs, y_obs, time_grid = _prepare_pytorch_channel_observation_context(
        policy,
        obs=obs,
        env_action_chunk=env_action_chunk,
    )
    obs_op = ChannelObservation(obs_sigma=float(args.obs_sigma))
    latent_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
    z_init = None
    init_mode = "random"
    if args.latent_init_from_bridge:
        z_init, init_mode = _latent_bridge_warm_start(
            policy,
            obs=obs,
            env_action_chunk=env_action_chunk,
            args=args,
        )
    map_seed = _map_restart_seed(obs, args=args)

    if args.fm_latent_map:
        z_map, restart_latents, restart_energies, best_restart_index = _run_pytorch_channel_latent_map_restarts(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs,
            time_grid=time_grid,
            z_seed=z_init,
            args=args,
            rng_seed=map_seed,
        )
        chain_init_mode = _map_restart_chain_init_mode(init_mode, best_restart_index=best_restart_index)
        latent_payload["posterior_init_mode"] = chain_init_mode
        latent_payload["posterior_chain_init"] = chain_init_mode
        latent_payload["map_restart_recovered_noise"] = np.asarray(restart_latents, dtype=np.float32)
        latent_payload["map_restart_energies"] = np.asarray(restart_energies, dtype=np.float32)
        latent_payload["map_best_restart_index"] = int(best_restart_index)
        latent_payload["recovered_noise"] = np.asarray(z_map[0].cpu(), dtype=np.float32)
        latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_map[0].cpu(), dtype=np.float32)
        latent_payload["posterior_best_energy"] = float(restart_energies[best_restart_index])
        return latent_payload

    if args.fm_latent_posterior:
        z_map, restart_latents, restart_energies, best_restart_index = _run_pytorch_channel_latent_map_restarts(
            policy,
            model_inputs=model_inputs,
            y_obs=y_obs,
            time_grid=time_grid,
            z_seed=z_init,
            args=args,
            rng_seed=map_seed,
        )
        chain_init_mode = _map_restart_chain_init_mode(init_mode, best_restart_index=best_restart_index)
        latent_payload["posterior_init_mode"] = chain_init_mode
        latent_payload["posterior_chain_init"] = chain_init_mode
        latent_payload["map_restart_recovered_noise"] = np.asarray(restart_latents, dtype=np.float32)
        latent_payload["map_restart_energies"] = np.asarray(restart_energies, dtype=np.float32)
        latent_payload["map_best_restart_index"] = int(best_restart_index)
        latent_payload["posterior_best_energy"] = float(restart_energies[best_restart_index])
        sampler = FMLatentPosteriorSampler(
            policy._model,
            obs_op,
            FMLatentPosteriorConfig(
                obs_sigma=float(args.obs_sigma),
                step_size=float(args.posterior_step_size),
                burnin_steps=int(args.posterior_burnin),
                thinning=int(args.posterior_thinning),
                num_samples=int(args.posterior_num_samples),
            ),
        )
        out = sampler.sample(model_inputs=model_inputs, y_obs=y_obs, time_grid=time_grid, z_init=z_map)
        z_samples = torch.nan_to_num(out["z_samples"].detach())
        z_mean = z_samples.mean(dim=1)
        z_std = z_samples.std(dim=1, unbiased=False)
        latent_payload["recovered_noise"] = np.asarray(z_mean[0].cpu(), dtype=np.float32)
        latent_payload["posterior_recovered_noise_samples"] = np.asarray(z_samples[0].cpu(), dtype=np.float32)
        latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_mean[0].cpu(), dtype=np.float32)
        latent_payload["posterior_recovered_noise_std"] = np.asarray(z_std[0].cpu(), dtype=np.float32)
        return latent_payload

    raise ValueError("Expected fm_latent_map or fm_latent_posterior to be enabled.")

def _recover_noise_from_channel_observation_latent_jax(
    policy,
    *,
    obs: dict,
    env_action_chunk: np.ndarray,
    raw_action_chunk: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_inputs, y_obs, time_grid = _prepare_jax_channel_observation_context(
        policy,
        obs=obs,
        env_action_chunk=env_action_chunk,
    )
    latent_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
    raw_dim = int(getattr(policy._model, "action_dim", 32))
    horizon = int(y_obs.shape[1])
    z_init = None
    init_mode = "random"
    if args.latent_init_from_bridge:
        z_init, init_mode = _latent_bridge_warm_start(
            policy,
            obs=obs,
            env_action_chunk=env_action_chunk,
            args=args,
        )
    else:
        z_init = jax.random.normal(jax.random.key(0), (1, horizon, raw_dim), dtype=jnp.float32)
    map_seed = _map_restart_seed(obs, args=args)

    if args.fm_latent_map:
        z_map, restart_latents, restart_energies, best_restart_index = _run_jax_channel_latent_map_restarts(
            policy,
            model_inputs=model_inputs,
            y_obs=jnp.asarray(y_obs, dtype=jnp.float32),
            time_grid=jnp.asarray(time_grid, dtype=jnp.float32),
            z_seed=jnp.asarray(z_init, dtype=jnp.float32),
            args=args,
            rng_seed=map_seed,
        )
        chain_init_mode = _map_restart_chain_init_mode(init_mode, best_restart_index=best_restart_index)
        latent_payload["posterior_init_mode"] = chain_init_mode
        latent_payload["posterior_chain_init"] = chain_init_mode
        latent_payload["map_restart_recovered_noise"] = np.asarray(restart_latents, dtype=np.float32)
        latent_payload["map_restart_energies"] = np.asarray(restart_energies, dtype=np.float32)
        latent_payload["map_best_restart_index"] = int(best_restart_index)
        latent_payload["recovered_noise"] = np.asarray(z_map[0], dtype=np.float32)
        latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_map[0], dtype=np.float32)
        latent_payload["posterior_best_energy"] = float(restart_energies[best_restart_index])
        return latent_payload

    if args.fm_latent_posterior:
        def energy_fn(z: jax.Array, *, prior_weight: float) -> jax.Array:
            a_pred = policy._model.sample_actions_from_noise(model_inputs, z, time_grid)
            pred_obs = a_pred[:, :, : y_obs.shape[-1]]
            pred_obs = jnp.where(jnp.isfinite(pred_obs), pred_obs, y_obs)
            obs_loss = 0.5 * jnp.mean(jnp.square((pred_obs - y_obs) / float(args.obs_sigma)))
            prior_loss = 0.5 * float(prior_weight) * jnp.mean(jnp.square(z))
            tether_weight = _posterior_map_tether_weight(args)
            tether_loss = (
                0.5 * tether_weight * jnp.mean(jnp.square(z - z_anchor))
                if tether_weight > 0.0
                else jnp.asarray(0.0, dtype=z.dtype)
            )
            return obs_loss + prior_loss + tether_loss

        z_map, restart_latents, restart_energies, best_restart_index = _run_jax_channel_latent_map_restarts(
            policy,
            model_inputs=model_inputs,
            y_obs=jnp.asarray(y_obs, dtype=jnp.float32),
            time_grid=jnp.asarray(time_grid, dtype=jnp.float32),
            z_seed=jnp.asarray(z_init, dtype=jnp.float32),
            args=args,
            rng_seed=map_seed,
        )
        z_map = jnp.nan_to_num(z_map)
        z_anchor = jnp.asarray(z_map, dtype=jnp.float32)
        chain_init_mode = _map_restart_chain_init_mode(init_mode, best_restart_index=best_restart_index)
        latent_payload["posterior_init_mode"] = chain_init_mode
        latent_payload["posterior_chain_init"] = chain_init_mode
        latent_payload["map_restart_recovered_noise"] = np.asarray(restart_latents, dtype=np.float32)
        latent_payload["map_restart_energies"] = np.asarray(restart_energies, dtype=np.float32)
        latent_payload["map_best_restart_index"] = int(best_restart_index)
        total_steps = int(args.posterior_burnin) + int(args.posterior_num_samples) * int(args.posterior_thinning)
        step_size = float(args.posterior_step_size)
        sqrt_step = jnp.asarray(step_size**0.5, dtype=jnp.float32)

        def ula_step(
            carry: tuple[jax.Array, jax.Array],
            step_idx: jax.Array,  # noqa: ARG001
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            z, key = carry
            grad_z = jax.grad(
                lambda latent: energy_fn(latent, prior_weight=float(args.latent_prior_weight))
            )(z)
            grad_z = jnp.nan_to_num(grad_z)
            grad_z = _clip_jax_grad_by_global_norm(grad_z, max_norm=_posterior_grad_clip_norm(args))
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, z.shape, dtype=jnp.float32)
            z_next = z - 0.5 * step_size * grad_z + sqrt_step * noise
            z_next = jnp.nan_to_num(z_next)
            return (z_next, key), z_next

        (_, _), z_history = jax.lax.scan(
            ula_step,
            (jnp.asarray(z_map, dtype=jnp.float32), jax.random.key(0)),
            jnp.arange(total_steps, dtype=jnp.int32),
        )
        sample_indices = jnp.asarray(
            [
                int(args.posterior_burnin) + sample_idx * int(args.posterior_thinning)
                for sample_idx in range(int(args.posterior_num_samples))
            ],
            dtype=jnp.int32,
        )
        z_samples = jnp.take(z_history, sample_indices, axis=0)
        z_samples = jnp.swapaxes(z_samples, 0, 1)
        z_samples = jnp.nan_to_num(z_samples)
        z_mean = jnp.mean(z_samples, axis=1)
        z_std = jnp.std(z_samples, axis=1)
        sample_energies = jax.vmap(
            lambda z: energy_fn(z[None, ...], prior_weight=float(args.latent_prior_weight))
        )(z_samples[0])
        best_index = int(jnp.argmin(sample_energies))
        latent_payload["recovered_noise"] = np.asarray(z_mean[0], dtype=np.float32)
        latent_payload["posterior_recovered_noise_samples"] = np.asarray(z_samples[0], dtype=np.float32)
        latent_payload["posterior_recovered_noise_mean"] = np.asarray(z_mean[0], dtype=np.float32)
        latent_payload["posterior_recovered_noise_std"] = np.asarray(z_std[0], dtype=np.float32)
        latent_payload["posterior_restart_energies"] = np.asarray(sample_energies, dtype=np.float32)
        latent_payload["posterior_best_energy"] = float(sample_energies[best_index])
        latent_payload["posterior_best_restart_index"] = best_index
        return latent_payload

    raise ValueError("Expected fm_latent_map or fm_latent_posterior to be enabled.")


def _make_jax_velocity_fn(policy, observation: model_lib.Observation) -> Callable[[np.ndarray, float], np.ndarray]:
    model = policy._model
    model_inputs = _prepare_jax_sampling_context(policy, observation)

    def velocity_fn(x_t: np.ndarray, time: float) -> np.ndarray:
        x_t_batch = jnp.asarray(x_t, dtype=jnp.float32)[None, ...]
        v_t, _ = model.predict_velocity_and_endpoint(model_inputs, x_t_batch, jnp.asarray(time, dtype=jnp.float32))
        return np.asarray(v_t[0], dtype=np.float32)

    return velocity_fn


def _make_pytorch_velocity_fn(policy, observation: model_lib.Observation) -> Callable[[np.ndarray, float], np.ndarray]:
    model = policy._model
    model_inputs = _prepare_pytorch_sampling_context(policy, observation)

    def velocity_fn(x_t: np.ndarray, time: float) -> np.ndarray:
        x_t_batch = torch.from_numpy(np.asarray(x_t, dtype=np.float32)).to(policy._pytorch_device)[None, ...]
        timestep = torch.tensor(float(time), dtype=torch.float32, device=policy._pytorch_device)
        v_t, _ = model.predict_velocity_and_endpoint(model_inputs, x_t_batch, timestep)
        return np.asarray(v_t[0].detach().cpu(), dtype=np.float32)

    return velocity_fn


def _make_velocity_fn(policy, obs: dict) -> Callable[[np.ndarray, float], np.ndarray]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    if policy._is_pytorch_model:
        return _make_pytorch_velocity_fn(policy, observation)
    return _make_jax_velocity_fn(policy, observation)


def _integrate_reverse_flow(
    actions: np.ndarray,
    *,
    num_steps: int,
    velocity_fn: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    x_t = np.asarray(actions, dtype=np.float32).copy()
    dt = 1.0 / float(num_steps)
    time = dt
    for _ in range(num_steps):
        v_t = np.asarray(velocity_fn(x_t, time), dtype=np.float32)
        x_t = x_t + dt * v_t
        time += dt
    return x_t.astype(np.float32)


def _integrate_forward_flow_static(
    noise: np.ndarray,
    *,
    num_steps: int,
    velocity_fn: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    x_t = np.asarray(noise, dtype=np.float32).copy()
    dt = -1.0 / float(num_steps)
    time = 1.0
    for _ in range(num_steps):
        v_t = np.asarray(velocity_fn(x_t, time), dtype=np.float32)
        x_t = x_t + dt * v_t
        time += dt
    return x_t.astype(np.float32)


def _refine_latent_via_forward_optimization(
    initial_noise: np.ndarray,
    *,
    target_actions: np.ndarray,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    num_steps: int,
    learning_rate: float,
    latent_l2: float,
    init_l2: float,
    finite_difference_eps: float = 1e-3,
) -> np.ndarray:
    if num_steps <= 0:
        return np.asarray(initial_noise, dtype=np.float32)

    initial_noise64 = np.asarray(initial_noise, dtype=np.float64)
    z = initial_noise64.copy()
    target_actions64 = np.asarray(target_actions, dtype=np.float64)
    def loss_fn(latent: np.ndarray) -> float:
        prediction = np.asarray(forward_fn(latent.astype(np.float32)), dtype=np.float64)
        recon = float(np.mean(np.square(prediction - target_actions64)))
        latent_penalty = float(latent_l2 * np.mean(np.square(latent)))
        init_penalty = float(init_l2 * np.mean(np.square(latent - initial_noise64)))
        return recon + latent_penalty + init_penalty

    for step in range(1, num_steps + 1):
        grad = np.zeros_like(z, dtype=np.float64)
        for index in np.ndindex(z.shape):
            delta = np.zeros_like(z, dtype=np.float64)
            delta[index] = finite_difference_eps
            grad[index] = (loss_fn(z + delta) - loss_fn(z - delta)) / (2.0 * finite_difference_eps)
        z = z - learning_rate * grad
    return z.astype(np.float32)


def _make_jax_differentiable_forward_fn(policy, obs: dict) -> Callable[[jax.Array], jax.Array]:
    observation, _, _ = _prepare_policy_inputs(policy, obs)
    model = policy._model
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = preprocessed.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(preprocessed)
    prefix_attn_mask = jax_pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    num_steps = int(policy._sample_kwargs.get("num_steps", 10))
    dt = -1.0 / float(num_steps)
    times = jnp.asarray([1.0 + dt * step for step in range(num_steps)], dtype=jnp.float32)

    def velocity_fn(x_t: jax.Array, time: jax.Array) -> jax.Array:
        x_t_batch = x_t[None, ...]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            preprocessed,
            x_t_batch,
            jnp.broadcast_to(time, batch_size),
        )
        suffix_attn_mask = jax_pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        return v_t[0]

    def forward_fn(noise: jax.Array) -> jax.Array:
        noise_arr = jnp.asarray(noise, dtype=jnp.float32)

        def step(x_t, time):
            v_t = velocity_fn(x_t, time)
            return x_t + jnp.asarray(dt, dtype=jnp.float32) * v_t, None

        x_final, _ = jax.lax.scan(step, noise_arr, times)
        return x_final

    return forward_fn


def _make_jax_forward_fn(policy, obs: dict) -> Callable[[np.ndarray], np.ndarray]:
    differentiable_forward_fn = _make_jax_differentiable_forward_fn(policy, obs)

    def forward_fn(noise: np.ndarray) -> np.ndarray:
        return np.asarray(differentiable_forward_fn(jnp.asarray(noise, dtype=jnp.float32)), dtype=np.float32)

    return forward_fn


def _optimize_latent_with_adam_jax(
    *,
    init_noise: jax.Array,
    loss_fn: Callable[[jax.Array], jax.Array],
    num_steps: int,
    learning_rate: float,
) -> jax.Array:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    loss_and_grad = jax.value_and_grad(loss_fn)

    def step_fn(step_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        latent, first_moment, second_moment = carry
        _, grad = loss_and_grad(latent)
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(grad)
        step = jnp.asarray(step_idx + 1, dtype=jnp.float32)
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)
        latent = latent - learning_rate * first_hat / (jnp.sqrt(second_hat) + eps)
        return latent, first_moment, second_moment

    init_carry = (
        init_noise,
        jnp.zeros_like(init_noise),
        jnp.zeros_like(init_noise),
    )
    latent, _, _ = jax.lax.fori_loop(0, num_steps, step_fn, init_carry)
    return latent


def _refine_recovered_noise_jax(
    policy,
    *,
    obs: dict,
    raw_actions: np.ndarray,
    initial_noise: np.ndarray,
    num_steps: int,
    learning_rate: float,
    latent_l2: float,
    init_l2: float,
) -> np.ndarray:
    forward_fn = _make_jax_differentiable_forward_fn(policy, obs)
    target_actions = jnp.asarray(raw_actions, dtype=jnp.float32)
    init_noise = jnp.asarray(initial_noise, dtype=jnp.float32)

    def loss_fn(latent: jax.Array) -> jax.Array:
        prediction = forward_fn(latent)
        recon = jnp.mean(jnp.square(prediction - target_actions))
        latent_penalty = latent_l2 * jnp.mean(jnp.square(latent))
        init_penalty = init_l2 * jnp.mean(jnp.square(latent - init_noise))
        return recon + latent_penalty + init_penalty

    optimize_fn = jax.jit(_optimize_latent_with_adam_jax, static_argnames=("loss_fn", "num_steps"))
    z = optimize_fn(
        init_noise=init_noise,
        loss_fn=loss_fn,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    return np.asarray(z, dtype=np.float32)


def _refine_recovered_noise(
    policy,
    *,
    obs: dict,
    raw_actions: np.ndarray,
    initial_noise: np.ndarray,
    num_steps: int,
    learning_rate: float,
    latent_l2: float,
    init_l2: float,
) -> np.ndarray:
    if policy._is_pytorch_model:
        raise NotImplementedError("JAX latent refinement is implemented first; PyTorch refinement is not supported.")
    return _refine_recovered_noise_jax(
        policy,
        obs=obs,
        raw_actions=raw_actions,
        initial_noise=initial_noise,
        num_steps=num_steps,
        learning_rate=learning_rate,
        latent_l2=latent_l2,
        init_l2=init_l2,
    )


def _recover_noise_from_actions(
    policy,
    *,
    obs: dict,
    raw_actions: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    initial_noise = _integrate_reverse_flow(
        raw_actions,
        num_steps=args.num_inversion_steps,
        velocity_fn=_make_velocity_fn(policy, obs),
    )
    if args.inversion_method != "reverse_refine" or args.refinement_steps == 0:
        return initial_noise
    return _refine_recovered_noise(
        policy,
        obs=obs,
        raw_actions=raw_actions,
        initial_noise=initial_noise,
        num_steps=args.refinement_steps,
        learning_rate=args.refinement_learning_rate,
        latent_l2=args.refinement_latent_l2,
        init_l2=args.refinement_init_l2,
    )


def _recover_noise_cache_for_steps(
    policy,
    *,
    obs: dict,
    raw_actions: np.ndarray,
    args: argparse.Namespace,
) -> dict[int, np.ndarray]:
    step_counts = sorted({int(step) for step in getattr(args, "save_recovered_noise_cache_steps", []) if int(step) > 0})
    if not step_counts:
        return {}
    cache = {}
    for step_count in step_counts:
        cached_args = argparse.Namespace(**vars(args))
        cached_args.num_inversion_steps = int(step_count)
        cache[int(step_count)] = np.asarray(
            _recover_noise_from_actions(
                policy,
                obs=obs,
                raw_actions=raw_actions,
                args=cached_args,
            ),
            dtype=np.float32,
        )
    return cache


def _score_inverted_noise(
    recovered_noise: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> float:
    recovered_noise, reference = online_eval._align_reference_to_action_signal(recovered_noise, reference)
    score, _ = online_eval._multichannel_band_coherence_score(
        recovered_noise,
        reference,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
    )
    return float(score)


def _score_chunk_noise_similarity(
    recovered_noise: np.ndarray,
    reference: np.ndarray,
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> float:
    recovered_noise, reference = online_eval._align_reference_to_action_signal(recovered_noise, reference)
    if reference_mode == "bandpass":
        recovered_noise = online_eval.wm._band_limit(
            recovered_noise,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
        )
        reference = online_eval.wm._band_limit(
            reference,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
        )
    recovered_vec = recovered_noise.reshape(-1).astype(np.float32, copy=False)
    reference_vec = reference.reshape(-1).astype(np.float32, copy=False)
    if detector == "cosine":
        denom = float(np.linalg.norm(recovered_vec) * np.linalg.norm(reference_vec))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(recovered_vec, reference_vec) / denom)
    if detector == "dot":
        return float(np.mean(recovered_vec * reference_vec))
    if detector == "mse":
        return float(-np.mean(np.square(recovered_vec - reference_vec)))
    raise ValueError(f"Unsupported detector={detector!r}")


def _flatten_inversion_signal(chunk_traces: list[InversionChunkTrace], attr: str) -> np.ndarray:
    pieces = []
    for trace in chunk_traces:
        values = np.asarray(getattr(trace, attr), dtype=np.float32)
        pieces.append(values[: trace.executed_steps])
    if not pieces:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def _episode_inversion_score(
    chunk_traces: list[InversionChunkTrace],
    *,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> float:
    recovered_noise = _flatten_inversion_signal(chunk_traces, "recovered_noise")
    reference = _flatten_inversion_signal(chunk_traces, "reference")
    if recovered_noise.size == 0 or reference.size == 0:
        return 0.0
    return _score_inverted_noise(
        recovered_noise,
        reference,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
    )


def _selected_score_traces(
    chunk_traces: list[InversionChunkTrace],
    *,
    max_windows: int | None,
) -> list[InversionChunkTrace]:
    selected = [trace for trace in chunk_traces if trace.selected and trace.executed_steps > 0]
    if max_windows is None:
        return selected
    return selected[:max_windows]


def _episode_chunk_noise_score(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
) -> float:
    per_chunk_scores = []
    for trace in _selected_score_traces(chunk_traces, max_windows=max_windows):
        score_steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        per_chunk_scores.append(
            _score_chunk_noise_similarity(
                np.asarray(trace.recovered_noise[:score_steps], dtype=np.float32),
                np.asarray(trace.reference[:score_steps], dtype=np.float32),
                detector=detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
            )
        )
    if not per_chunk_scores:
        return 0.0
    scores = np.asarray(per_chunk_scores, dtype=np.float32)
    if aggregator == "sum":
        return float(np.sum(scores))
    if aggregator == "mean":
        return float(np.mean(scores))
    raise ValueError(f"Unsupported aggregator={aggregator!r}")


def _wrong_key_reference_configs(
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig,
    *,
    count: int,
    key_offset: int = 1,
) -> list[online_eval.wm.InternalNoiseWatermarkConfig]:
    if count <= 0:
        raise ValueError("count must be > 0")
    return [
        dataclasses.replace(reference_config, secret_key=int(reference_config.secret_key) + key_offset + idx)
        for idx in range(count)
    ]


def _window_score_vector(
    chunk_traces: list[InversionChunkTrace],
    *,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    score_step_scope: str,
    max_windows: int | None,
    base_detector: str = "cosine",
) -> np.ndarray:
    scores = []
    for trace in _selected_score_traces(chunk_traces, max_windows=max_windows):
        score_steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        scores.append(
            _score_chunk_noise_similarity(
                np.asarray(trace.recovered_noise[:score_steps], dtype=np.float32),
                np.asarray(trace.reference[:score_steps], dtype=np.float32),
                detector=base_detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
            )
        )
    return np.asarray(scores, dtype=np.float32)


def _select_whitened_subspace(
    centered_feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_feature = np.asarray(centered_feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if centered_feature.ndim != 1:
        raise ValueError(f"centered_feature must be rank 1, got shape={centered_feature.shape}")
    if null_matrix.ndim != 2:
        raise ValueError(f"null_matrix must be rank 2, got shape={null_matrix.shape}")
    if null_matrix.shape[1] != centered_feature.shape[0]:
        raise ValueError("null_matrix width must match centered_feature length")

    dim = centered_feature.shape[0]
    if dim == 0:
        return centered_feature, np.eye(0, dtype=np.float64), np.zeros((0,), dtype=np.float64)

    centered_null = null_matrix - np.mean(null_matrix, axis=0, keepdims=True)
    if null_matrix.shape[0] <= 1:
        cov = np.eye(dim, dtype=np.float64)
    else:
        cov = np.cov(centered_null, rowvar=False, bias=False)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    reg = max(1e-6, 1e-4 * float(np.trace(cov)) / max(dim, 1))
    cov = cov + reg * np.eye(dim, dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if subspace_rank is not None:
        rank = min(subspace_rank, dim)
        eigvals = eigvals[:rank]
        eigvecs = eigvecs[:, :rank]
    projected_feature = eigvecs.T @ centered_feature
    return projected_feature, eigvecs, eigvals


def _wmf_score_from_vectors(
    feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None = None,
) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0:
        return 0.0
    mu = np.mean(null_matrix, axis=0)
    centered_feature = feature - mu
    projected_feature, eigvecs, eigvals = _select_whitened_subspace(
        centered_feature,
        null_matrix,
        subspace_rank=subspace_rank,
    )
    if projected_feature.size == 0:
        return 0.0
    template = np.sum(eigvecs, axis=0)
    whitened = projected_feature / np.sqrt(np.maximum(eigvals, 1e-8))
    template_whitened = template / np.sqrt(np.maximum(eigvals, 1e-8))
    return float(np.dot(template_whitened, whitened))


def _ace_score_from_vectors(
    feature: np.ndarray,
    null_matrix: np.ndarray,
    *,
    subspace_rank: int | None = None,
) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0:
        return 0.0
    mu = np.mean(null_matrix, axis=0)
    centered_feature = feature - mu
    projected_feature, eigvecs, eigvals = _select_whitened_subspace(
        centered_feature,
        null_matrix,
        subspace_rank=subspace_rank,
    )
    if projected_feature.size == 0:
        return 0.0
    template = np.sum(eigvecs, axis=0)
    inv_std = 1.0 / np.sqrt(np.maximum(eigvals, 1e-8))
    whitened_feature = projected_feature * inv_std
    whitened_template = template * inv_std
    denom = float(np.linalg.norm(whitened_feature) * np.linalg.norm(whitened_template))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(whitened_feature, whitened_template) / denom)


def _advanced_episode_score(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig,
    episode_nonce: int,
    sample_rate_hz: float,
    reference_mode: str,
    freq_range: tuple[float, float],
    score_step_scope: str,
    max_windows: int | None,
    null_decoy_count: int,
    subspace_rank: int | None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> float:
    true_vector = _window_score_vector(
        chunk_traces,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
        base_detector="cosine",
    )
    if true_vector.size == 0:
        return 0.0
    configs = (
        null_reference_configs
        if null_reference_configs is not None
        else _wrong_key_reference_configs(reference_config, count=null_decoy_count)
    )
    null_vectors = []
    for config in configs:
        retargeted = _retarget_chunk_references(
            chunk_traces,
            reference_config=config,
            sample_rate_hz=sample_rate_hz,
            episode_nonce=episode_nonce,
        )
        null_vector = _window_score_vector(
            retargeted,
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
            score_step_scope=score_step_scope,
            max_windows=max_windows,
            base_detector="cosine",
        )
        if null_vector.shape == true_vector.shape:
            null_vectors.append(null_vector)
    if not null_vectors:
        return 0.0
    null_matrix = np.asarray(null_vectors, dtype=np.float64)
    if detector == "wmf":
        return _wmf_score_from_vectors(true_vector, null_matrix, subspace_rank=subspace_rank)
    if detector == "ace":
        return _ace_score_from_vectors(true_vector, null_matrix, subspace_rank=subspace_rank)
    raise ValueError(f"Unsupported advanced detector={detector!r}")


def _episode_score(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig | None = None,
    episode_nonce: int | None = None,
    null_decoy_count: int = 32,
    subspace_rank: int | None = None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> float:
    if detector in {"wmf", "ace"}:
        if reference_config is None or episode_nonce is None:
            raise ValueError(f"{detector} requires reference_config and episode_nonce")
        return _advanced_episode_score(
            chunk_traces,
            detector=detector,
            reference_config=reference_config,
            episode_nonce=episode_nonce,
            sample_rate_hz=sample_rate_hz,
            reference_mode=reference_mode,
            freq_range=freq_range,
            score_step_scope=score_step_scope,
            max_windows=max_windows,
            null_decoy_count=null_decoy_count,
            subspace_rank=subspace_rank,
            null_reference_configs=null_reference_configs,
        )
    if detector == "coherence":
        return _episode_inversion_score(
            _selected_score_traces(chunk_traces, max_windows=max_windows),
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
        )
    return _episode_chunk_noise_score(
        chunk_traces,
        detector=detector,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        aggregator=aggregator,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
    )


def _aggregate_chunk_scores(chunk_scores: np.ndarray, *, aggregator: str) -> float:
    chunk_scores = np.asarray(chunk_scores, dtype=np.float32)
    if chunk_scores.size == 0:
        return 0.0
    if aggregator == "sum":
        return float(np.sum(chunk_scores))
    if aggregator == "mean":
        return float(np.mean(chunk_scores))
    raise ValueError(f"Unsupported aggregator={aggregator!r}")


def _score_episode_from_noise_traces(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig | None = None,
    episode_nonce: int | None = None,
    null_decoy_count: int = 32,
    subspace_rank: int | None = None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> tuple[float, np.ndarray]:
    chunk_scores = _chunk_scores_for_episode(
        chunk_traces,
        detector=detector,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
    )
    episode_score = _episode_score(
        chunk_traces,
        detector=detector,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        aggregator=aggregator,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
        reference_config=reference_config,
        episode_nonce=episode_nonce,
        null_decoy_count=null_decoy_count,
        subspace_rank=subspace_rank,
        null_reference_configs=null_reference_configs,
    )
    return float(episode_score), np.asarray(chunk_scores, dtype=np.float32)


def _posterior_noise_samples_for_trace(trace: InversionChunkTrace) -> np.ndarray:
    samples = np.asarray(trace.posterior_recovered_noise_samples, dtype=np.float32)
    if samples.ndim == 3 and samples.shape[0] > 0:
        return samples
    recovered_noise = np.asarray(trace.recovered_noise, dtype=np.float32)
    if recovered_noise.ndim != 2:
        raise ValueError(f"Expected recovered_noise to have shape [T, D], got {recovered_noise.shape}")
    return np.zeros((0, *recovered_noise.shape), dtype=np.float32)


def _posterior_sample_count(
    chunk_traces: list[InversionChunkTrace],
    *,
    max_windows: int | None,
) -> int:
    sample_counts = [
        int(_posterior_noise_samples_for_trace(trace).shape[0])
        for trace in _selected_score_traces(chunk_traces, max_windows=max_windows)
    ]
    if not sample_counts:
        return 0
    if len(set(sample_counts)) != 1:
        raise ValueError(f"Posterior sample count mismatch across selected traces: {sample_counts}")
    return sample_counts[0]


def _posterior_sample_traces(
    chunk_traces: list[InversionChunkTrace],
    *,
    sample_index: int,
) -> list[InversionChunkTrace]:
    sampled_traces = []
    for trace in chunk_traces:
        if trace.selected and trace.executed_steps > 0:
            samples = _posterior_noise_samples_for_trace(trace)
            sampled_traces.append(
                dataclasses.replace(
                    trace,
                    recovered_noise=np.asarray(samples[sample_index], dtype=np.float32),
                )
            )
        else:
            sampled_traces.append(trace)
    return sampled_traces


def _posterior_episode_score_samples(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig | None = None,
    episode_nonce: int | None = None,
    null_decoy_count: int = 32,
    subspace_rank: int | None = None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = _posterior_sample_count(chunk_traces, max_windows=max_windows)
    if sample_count == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    sample_chunk_scores = []
    sample_episode_scores = []
    for sample_index in range(sample_count):
        sample_traces = _posterior_sample_traces(chunk_traces, sample_index=sample_index)
        sample_chunk_scores.append(
            _chunk_scores_for_episode(
                sample_traces,
                detector=detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
                score_step_scope=score_step_scope,
                max_windows=max_windows,
            )
        )
        sample_episode_scores.append(
            _episode_score(
                sample_traces,
                detector=detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
                aggregator=aggregator,
                score_step_scope=score_step_scope,
                max_windows=max_windows,
                reference_config=reference_config,
                episode_nonce=episode_nonce,
                null_decoy_count=null_decoy_count,
                subspace_rank=subspace_rank,
                null_reference_configs=null_reference_configs,
            )
        )

    return np.asarray(sample_episode_scores, dtype=np.float32), np.asarray(sample_chunk_scores, dtype=np.float32)


def _summarize_episode_score_samples(
    sample_scores: np.ndarray,
    *,
    fallback_score: float | None = None,
) -> dict[str, float | int]:
    sample_scores = np.asarray(sample_scores, dtype=np.float32).reshape(-1)
    if sample_scores.size == 0:
        base_score = 0.0 if fallback_score is None else float(fallback_score)
        return {
            "episode_score_mean": float(base_score),
            "episode_score_std": 0.0,
            "episode_score_q05": float(base_score),
            "episode_score_q95": float(base_score),
            "posterior_sample_count": 0,
        }
    q05, q95 = np.quantile(sample_scores, [0.05, 0.95])
    return {
        "episode_score_mean": float(np.mean(sample_scores)),
        "episode_score_std": float(np.std(sample_scores)),
        "episode_score_q05": float(q05),
        "episode_score_q95": float(q95),
        "posterior_sample_count": int(sample_scores.size),
    }


def _score_episode_from_noise_samples_summary(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig | None = None,
    episode_nonce: int | None = None,
    null_decoy_count: int = 32,
    subspace_rank: int | None = None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> tuple[dict[str, float | int], np.ndarray]:
    sample_scores, sample_chunk_scores = _posterior_episode_score_samples(
        chunk_traces,
        detector=detector,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        aggregator=aggregator,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
        reference_config=reference_config,
        episode_nonce=episode_nonce,
        null_decoy_count=null_decoy_count,
        subspace_rank=subspace_rank,
        null_reference_configs=null_reference_configs,
    )
    if sample_scores.size == 0:
        episode_score, chunk_scores = _score_episode_from_noise_traces(
            chunk_traces,
            detector=detector,
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
            aggregator=aggregator,
            score_step_scope=score_step_scope,
            max_windows=max_windows,
            reference_config=reference_config,
            episode_nonce=episode_nonce,
            null_decoy_count=null_decoy_count,
            subspace_rank=subspace_rank,
            null_reference_configs=null_reference_configs,
        )
        return _summarize_episode_score_samples(sample_scores, fallback_score=episode_score), np.asarray(
            chunk_scores,
            dtype=np.float32,
        )
    return _summarize_episode_score_samples(sample_scores), np.asarray(
        np.mean(sample_chunk_scores, axis=0),
        dtype=np.float32,
    )


def _score_episode_from_noise_samples(
    chunk_traces: list[InversionChunkTrace],
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    aggregator: str,
    score_step_scope: str,
    max_windows: int | None = None,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig | None = None,
    episode_nonce: int | None = None,
    null_decoy_count: int = 32,
    subspace_rank: int | None = None,
    null_reference_configs: list[online_eval.wm.InternalNoiseWatermarkConfig] | None = None,
) -> tuple[float, np.ndarray]:
    summary, chunk_scores = _score_episode_from_noise_samples_summary(
        chunk_traces,
        detector=detector,
        reference_mode=reference_mode,
        sample_rate_hz=sample_rate_hz,
        freq_range=freq_range,
        aggregator=aggregator,
        score_step_scope=score_step_scope,
        max_windows=max_windows,
        reference_config=reference_config,
        episode_nonce=episode_nonce,
        null_decoy_count=null_decoy_count,
        subspace_rank=subspace_rank,
        null_reference_configs=null_reference_configs,
    )
    return float(summary["episode_score_mean"]), np.asarray(chunk_scores, dtype=np.float32)


def _empirical_survival_pvalues(observed_scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    observed_scores = np.asarray(observed_scores, dtype=np.float32)
    null_scores = np.asarray(null_scores, dtype=np.float32)
    if observed_scores.ndim != 1:
        raise ValueError(f"observed_scores must be rank 1, got shape={observed_scores.shape}")
    if null_scores.ndim != 1:
        raise ValueError(f"null_scores must be rank 1, got shape={null_scores.shape}")
    if null_scores.size == 0:
        return np.ones_like(observed_scores, dtype=np.float32)
    return np.asarray(
        [(1.0 + float(np.sum(null_scores >= score))) / float(null_scores.size + 1) for score in observed_scores],
        dtype=np.float32,
    )


def _episode_recovery_rms(chunk_traces: list[InversionChunkTrace], *, max_windows: int | None = None) -> float:
    selected_traces = _selected_score_traces(chunk_traces, max_windows=max_windows)
    recovered_noise = _flatten_inversion_signal(selected_traces, "recovered_noise")
    injected_noise = _flatten_inversion_signal(selected_traces, "injected_noise")
    if recovered_noise.size == 0 or injected_noise.size == 0:
        return 0.0
    recovered_noise, injected_noise = online_eval._align_reference_to_action_signal(recovered_noise, injected_noise)
    return float(np.sqrt(np.mean(np.square(recovered_noise - injected_noise))))


def _wrong_key_reference_config(
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig,
) -> online_eval.wm.InternalNoiseWatermarkConfig:
    return dataclasses.replace(reference_config, secret_key=int(reference_config.secret_key) + 1)


def _retarget_chunk_references(
    chunk_traces: list[InversionChunkTrace],
    *,
    reference_config: online_eval.wm.InternalNoiseWatermarkConfig,
    sample_rate_hz: float,
    episode_nonce: int,
) -> list[InversionChunkTrace]:
    retargeted = []
    for trace in chunk_traces:
        if trace.selected and trace.executed_steps > 0:
            context = online_eval.wm.WatermarkContext(chunk_index=trace.chunk_index, episode_nonce=episode_nonce)
            reference = online_eval.wm.generate_keyed_reference(
                length=int(trace.reference.shape[0]),
                action_dim=int(trace.reference.shape[1]),
                sample_rate_hz=sample_rate_hz,
                config=reference_config,
                context=context,
            )
        else:
            reference = np.zeros_like(trace.reference, dtype=np.float32)
        retargeted.append(dataclasses.replace(trace, reference=np.asarray(reference, dtype=np.float32)))
    return retargeted


def _probe_position_dims(axis_mode: str) -> tuple[int, ...]:
    mapping = {
        "x": (0,),
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "xyz": (0, 1, 2),
    }
    return mapping[axis_mode]


def _probe_gripper_value(gripper_mode: str, *, current_value: float) -> float:
    if gripper_mode == "hold_open":
        return -1.0
    if gripper_mode == "hold_closed":
        return 1.0
    return float(current_value)


def _shape_probe_action_chunk(
    action_chunk: np.ndarray,
    *,
    args: argparse.Namespace,
    chunk_index: int,
) -> np.ndarray:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    executed = np.zeros_like(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected rank-2 action chunk, got shape={action_chunk.shape}")

    scaled = action_chunk * float(args.probe_speed_scale)
    pos_dims = [dim for dim in _probe_position_dims(args.probe_axis_mode) if dim < scaled.shape[1]]
    if pos_dims:
        if args.probe_pattern == "axis_sweep":
            active_dims = (pos_dims[chunk_index % len(pos_dims)],)
        elif args.probe_pattern == "circle":
            active_dims = tuple(pos_dims[:2])
        else:
            active_dims = tuple(pos_dims[:3])
        for dim in active_dims:
            executed[:, dim] = np.clip(scaled[:, dim], -args.probe_amplitude, args.probe_amplitude)

    if scaled.shape[1] > 6:
        current_gripper = float(np.median(scaled[:, 6])) if scaled.shape[0] else -1.0
        executed[:, 6] = _probe_gripper_value(args.probe_gripper_mode, current_value=current_gripper)
    return executed.astype(np.float32)


def _probe_settle_action(args: argparse.Namespace, *, action_dim: int) -> np.ndarray:
    action = np.zeros((action_dim,), dtype=np.float32)
    if action_dim > 2:
        action[2] = min(float(args.probe_amplitude), 0.02)
    if action_dim > 6:
        action[6] = _probe_gripper_value(args.probe_gripper_mode, current_value=-1.0)
    return action


def _run_task_rollout_with_inversion(
    policy,
    detector_policy,
    *,
    task,
    initial_state: np.ndarray,
    args: argparse.Namespace,
    runtime_modules: dict[str, Any],
    episode_nonce: int,
) -> tuple[online_eval.RolloutResult, list[InversionChunkTrace]]:
    reference_config = online_eval._make_watermark_config(args, telemetry_dim=int(policy._model.action_dim))
    env, task_description = online_eval._get_libero_env(
        task,
        resolution=args.resize_size,
        seed=args.seed,
        runtime_modules=runtime_modules,
    )
    telemetry: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    execution_segments: list[online_eval.ExecutionSegment] = []
    inversion_traces: list[InversionChunkTrace] = []
    chunk_index = 0
    chunk_size = 0
    done = False
    active_segment: dict[str, int] | None = None
    active_trace: dict[str, Any] | None = None

    try:
        env.reset()
        obs = env.set_init_state(copy.deepcopy(initial_state))
        max_steps = online_eval._suite_max_steps(args.task_suite_name)
        if args.max_rollout_steps is not None:
            max_steps = min(max_steps, int(args.max_rollout_steps))
        t = 0

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(online_eval.LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if not action_plan:
                element = online_eval._prepare_policy_observation(
                    obs,
                    task_description=task_description,
                    resize_size=args.resize_size,
                    image_tools=runtime_modules["image_tools"],
                )
                element["chunk_index"] = chunk_index
                element["episode_nonce"] = episode_nonce
                internal_horizon = int(policy._model.action_horizon)
                internal_dim = int(policy._model.action_dim)
                base_noise = online_eval._make_chunk_base_noise(
                    action_horizon=internal_horizon,
                    action_dim=internal_dim,
                    episode_nonce=episode_nonce,
                    chunk_index=chunk_index,
                    seed=args.seed,
                )
                context = online_eval.wm.WatermarkContext(chunk_index=chunk_index, episode_nonce=episode_nonce)
                selected = online_eval.wm.should_watermark_chunk(reference_config, context)
                reference = online_eval.wm.generate_keyed_reference(
                    length=internal_horizon,
                    action_dim=internal_dim,
                    sample_rate_hz=args.sample_rate_hz,
                    config=reference_config,
                    context=context,
                )
                if not selected:
                    reference = np.zeros_like(reference)
                outputs, injected_noise = _sample_raw_actions(policy, element, noise=base_noise)
                action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
                raw_action_chunk = np.asarray(outputs["raw_actions"], dtype=np.float32)
                if chunk_size == 0:
                    chunk_size = int(action_chunk.shape[0])
                planned_steps = int(min(args.replan_steps, action_chunk.shape[0]))
                observed_action_chunk = np.asarray(action_chunk, dtype=np.float32)
                latent_trace_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
                if selected:
                    if _full_action_latent_recovery_enabled(args):
                        latent_trace_payload = _recover_noise_from_full_action_latent(
                            detector_policy,
                            obs=element,
                            raw_action_chunk=raw_action_chunk,
                            args=args,
                        )
                        recovered_noise = np.asarray(latent_trace_payload["recovered_noise"], dtype=np.float32)
                        recovered_noise_by_step = {}
                    elif _channel_latent_recovery_enabled(args):
                        latent_trace_payload = _recover_noise_from_channel_observation_latent(
                            detector_policy,
                            obs=element,
                            env_action_chunk=observed_action_chunk,
                            raw_action_chunk=raw_action_chunk,
                            args=args,
                        )
                        recovered_noise = np.asarray(latent_trace_payload["recovered_noise"], dtype=np.float32)
                        recovered_noise_by_step = {}
                    else:
                        inversion_action_chunk = raw_action_chunk
                        if args.fm_channel_inverse:
                            inversion_action_chunk = _complete_raw_actions_from_channel_observation(
                                detector_policy,
                                obs=element,
                                env_action_chunk=observed_action_chunk,
                                args=args,
                            )
                        recovered_noise = _recover_noise_from_actions(
                            detector_policy,
                            obs=element,
                            raw_actions=inversion_action_chunk,
                            args=args,
                        )
                        recovered_noise_by_step = _recover_noise_cache_for_steps(
                            detector_policy,
                            obs=element,
                            raw_actions=inversion_action_chunk,
                            args=args,
                        )
                else:
                    recovered_noise = np.zeros_like(reference)
                    recovered_noise_by_step = {
                        int(step): np.zeros_like(reference, dtype=np.float32)
                        for step in getattr(args, "save_recovered_noise_cache_steps", [])
                        if int(step) > 0
                    }
                action_plan.extend(observed_action_chunk[:planned_steps])
                active_segment = {
                    "chunk_index": chunk_index,
                    "start_step": len(telemetry),
                    "planned_steps": planned_steps,
                    "executed_steps": 0,
                }
                active_trace = {
                    "chunk_index": chunk_index,
                    "planned_steps": planned_steps,
                    "reference": np.asarray(reference, dtype=np.float32),
                    "recovered_noise": np.asarray(recovered_noise, dtype=np.float32),
                    "injected_noise": np.asarray(injected_noise, dtype=np.float32),
                    "raw_actions": np.asarray(raw_action_chunk, dtype=np.float32),
                    "observed_actions": np.asarray(observed_action_chunk, dtype=np.float32),
                    "selected": bool(selected),
                    "prompt": str(element["prompt"]),
                    "observation_state": np.asarray(element["observation/state"], dtype=np.float32),
                    "observation_image": np.asarray(element["observation/image"], dtype=np.uint8),
                    "observation_wrist_image": np.asarray(element["observation/wrist_image"], dtype=np.uint8),
                    "recovered_noise_by_step": {
                        int(step): np.asarray(value, dtype=np.float32)
                        for step, value in recovered_noise_by_step.items()
                    },
                    "map_restart_recovered_noise": np.asarray(
                        latent_trace_payload["map_restart_recovered_noise"],
                        dtype=np.float32,
                    ),
                    "map_restart_energies": np.asarray(
                        latent_trace_payload["map_restart_energies"],
                        dtype=np.float32,
                    ),
                    "map_best_restart_index": int(latent_trace_payload["map_best_restart_index"]),
                    "posterior_recovered_noise_samples": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_samples"],
                        dtype=np.float32,
                    ),
                    "posterior_recovered_noise_mean": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_mean"],
                        dtype=np.float32,
                    ),
                    "posterior_recovered_noise_std": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_std"],
                        dtype=np.float32,
                    ),
                    "posterior_restart_energies": np.asarray(
                        latent_trace_payload["posterior_restart_energies"],
                        dtype=np.float32,
                    ),
                    "posterior_best_energy": float(latent_trace_payload["posterior_best_energy"]),
                    "posterior_best_restart_index": int(latent_trace_payload["posterior_best_restart_index"]),
                    "posterior_init_mode": str(latent_trace_payload["posterior_init_mode"]),
                    "posterior_chain_init": str(latent_trace_payload["posterior_chain_init"]),
                }
                chunk_index += 1

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            telemetry.append(online_eval._extract_telemetry(obs))
            executed_actions.append(np.asarray(action, dtype=np.float32))
            if active_segment is None or active_trace is None:
                raise RuntimeError("Expected active segment/trace while executing action plan.")
            active_segment["executed_steps"] += 1
            if active_segment["executed_steps"] == active_segment["planned_steps"] or done:
                executed_steps = int(active_segment["executed_steps"])
                execution_segments.append(
                    online_eval.ExecutionSegment(
                        chunk_index=active_segment["chunk_index"],
                        start_step=active_segment["start_step"],
                        end_step=active_segment["start_step"] + executed_steps,
                        executed_steps=executed_steps,
                    )
                )
                inversion_traces.append(
                    InversionChunkTrace(
                        chunk_index=int(active_trace["chunk_index"]),
                        executed_steps=executed_steps,
                        reference=np.asarray(active_trace["reference"], dtype=np.float32),
                        recovered_noise=np.asarray(active_trace["recovered_noise"], dtype=np.float32),
                        injected_noise=np.asarray(active_trace["injected_noise"], dtype=np.float32),
                        raw_actions=np.asarray(active_trace["raw_actions"], dtype=np.float32),
                        observed_actions=np.asarray(active_trace["observed_actions"], dtype=np.float32),
                        selected=bool(active_trace["selected"]),
                        prompt=str(active_trace["prompt"]),
                        observation_state=np.asarray(active_trace["observation_state"], dtype=np.float32),
                        observation_image=np.asarray(active_trace["observation_image"], dtype=np.uint8),
                        observation_wrist_image=np.asarray(active_trace["observation_wrist_image"], dtype=np.uint8),
                        recovered_noise_by_step={
                            int(step): np.asarray(value, dtype=np.float32)
                            for step, value in active_trace["recovered_noise_by_step"].items()
                        },
                        map_restart_recovered_noise=np.asarray(
                            active_trace["map_restart_recovered_noise"],
                            dtype=np.float32,
                        ),
                        map_restart_energies=np.asarray(
                            active_trace["map_restart_energies"],
                            dtype=np.float32,
                        ),
                        map_best_restart_index=int(active_trace["map_best_restart_index"]),
                        posterior_recovered_noise_samples=np.asarray(
                            active_trace["posterior_recovered_noise_samples"],
                            dtype=np.float32,
                        ),
                        posterior_recovered_noise_mean=np.asarray(
                            active_trace["posterior_recovered_noise_mean"],
                            dtype=np.float32,
                        ),
                        posterior_recovered_noise_std=np.asarray(
                            active_trace["posterior_recovered_noise_std"],
                            dtype=np.float32,
                        ),
                        posterior_restart_energies=np.asarray(
                            active_trace["posterior_restart_energies"],
                            dtype=np.float32,
                        ),
                        posterior_best_energy=float(active_trace["posterior_best_energy"]),
                        posterior_best_restart_index=int(active_trace["posterior_best_restart_index"]),
                        posterior_init_mode=str(active_trace["posterior_init_mode"]),
                        posterior_chain_init=str(active_trace["posterior_chain_init"]),
                    )
                )
                active_segment = None
                active_trace = None
            t += 1
            if done:
                break
    finally:
        env.close()

    result = online_eval.RolloutResult(
        telemetry=np.asarray(telemetry, dtype=np.float32),
        success=bool(done),
        chunk_size=chunk_size,
        task_description=task_description,
        steps=len(telemetry),
        execution_segments=tuple(execution_segments),
        chunk_traces=(),
        executed_actions=np.asarray(executed_actions, dtype=np.float32) if executed_actions else np.zeros((0, 0), dtype=np.float32),
    )
    return result, inversion_traces


def _run_probe_rollout_with_inversion(
    policy,
    detector_policy,
    *,
    task,
    initial_state: np.ndarray,
    args: argparse.Namespace,
    runtime_modules: dict[str, Any],
    episode_nonce: int,
) -> tuple[online_eval.RolloutResult, list[InversionChunkTrace]]:
    reference_config = online_eval._make_watermark_config(args, telemetry_dim=int(policy._model.action_dim))
    env, task_description = online_eval._get_libero_env(
        task,
        resolution=args.resize_size,
        seed=args.seed,
        runtime_modules=runtime_modules,
    )
    telemetry: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    execution_segments: list[online_eval.ExecutionSegment] = []
    inversion_traces: list[InversionChunkTrace] = []
    chunk_index = 0
    chunk_size = 0
    done = False
    active_segment: dict[str, int] | None = None
    active_trace: dict[str, Any] | None = None
    probe_steps_remaining = _probe_total_steps(duration_sec=args.probe_duration_sec, sample_rate_hz=args.sample_rate_hz)

    try:
        env.reset()
        obs = env.set_init_state(copy.deepcopy(initial_state))
        for _ in range(args.num_steps_wait):
            obs, _, done, _ = env.step(online_eval.LIBERO_DUMMY_ACTION)
            if done:
                break
        if not done:
            settle_action = _probe_settle_action(args, action_dim=len(online_eval.LIBERO_DUMMY_ACTION))
            for _ in range(args.probe_settle_steps):
                obs, _, done, _ = env.step(settle_action.tolist())
                if done:
                    break
        probe_anchor_obs = copy.deepcopy(obs)
        probe_prompt = _resolve_task_prompt(
            task_description,
            eval_mode="probe_verification",
            probe_prompt=args.probe_prompt,
        )

        while probe_steps_remaining > 0 and not done:
            if not action_plan:
                element = online_eval._prepare_policy_observation(
                    probe_anchor_obs,
                    task_description=probe_prompt,
                    resize_size=args.resize_size,
                    image_tools=runtime_modules["image_tools"],
                )
                element["chunk_index"] = chunk_index
                element["episode_nonce"] = episode_nonce
                internal_horizon = int(policy._model.action_horizon)
                internal_dim = int(policy._model.action_dim)
                base_noise = online_eval._make_chunk_base_noise(
                    action_horizon=internal_horizon,
                    action_dim=internal_dim,
                    episode_nonce=episode_nonce,
                    chunk_index=chunk_index,
                    seed=args.seed,
                )
                context = online_eval.wm.WatermarkContext(chunk_index=chunk_index, episode_nonce=episode_nonce)
                selected = online_eval.wm.should_watermark_chunk(reference_config, context)
                reference = online_eval.wm.generate_keyed_reference(
                    length=internal_horizon,
                    action_dim=internal_dim,
                    sample_rate_hz=args.sample_rate_hz,
                    config=reference_config,
                    context=context,
                )
                if not selected:
                    reference = np.zeros_like(reference)
                outputs, injected_noise = _sample_raw_actions(policy, element, noise=base_noise)
                action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
                raw_action_chunk = np.asarray(outputs["raw_actions"], dtype=np.float32)
                if chunk_size == 0:
                    chunk_size = int(action_chunk.shape[0])
                planned_steps = int(min(args.probe_replan_interval, action_chunk.shape[0], probe_steps_remaining))
                observed_action_chunk = np.asarray(action_chunk, dtype=np.float32)
                latent_trace_payload = _latent_trace_defaults(raw_action_chunk=raw_action_chunk, args=args)
                if selected:
                    if _full_action_latent_recovery_enabled(args):
                        latent_trace_payload = _recover_noise_from_full_action_latent(
                            detector_policy,
                            obs=element,
                            raw_action_chunk=raw_action_chunk,
                            args=args,
                        )
                        recovered_noise = np.asarray(latent_trace_payload["recovered_noise"], dtype=np.float32)
                        recovered_noise_by_step = {}
                    elif _channel_latent_recovery_enabled(args):
                        latent_trace_payload = _recover_noise_from_channel_observation_latent(
                            detector_policy,
                            obs=element,
                            env_action_chunk=observed_action_chunk,
                            raw_action_chunk=raw_action_chunk,
                            args=args,
                        )
                        recovered_noise = np.asarray(latent_trace_payload["recovered_noise"], dtype=np.float32)
                        recovered_noise_by_step = {}
                    else:
                        inversion_action_chunk = raw_action_chunk
                        if args.fm_channel_inverse:
                            inversion_action_chunk = _complete_raw_actions_from_channel_observation(
                                detector_policy,
                                obs=element,
                                env_action_chunk=observed_action_chunk,
                                args=args,
                            )
                        recovered_noise = _recover_noise_from_actions(
                            detector_policy,
                            obs=element,
                            raw_actions=inversion_action_chunk,
                            args=args,
                        )
                        recovered_noise_by_step = _recover_noise_cache_for_steps(
                            detector_policy,
                            obs=element,
                            raw_actions=inversion_action_chunk,
                            args=args,
                        )
                else:
                    recovered_noise = np.zeros_like(reference)
                    recovered_noise_by_step = {
                        int(step): np.zeros_like(reference, dtype=np.float32)
                        for step in getattr(args, "save_recovered_noise_cache_steps", [])
                        if int(step) > 0
                    }
                shaped_actions = _shape_probe_action_chunk(
                    observed_action_chunk[:planned_steps],
                    args=args,
                    chunk_index=chunk_index,
                )
                action_plan.extend(shaped_actions)
                active_segment = {
                    "chunk_index": chunk_index,
                    "start_step": len(telemetry),
                    "planned_steps": planned_steps,
                    "executed_steps": 0,
                }
                active_trace = {
                    "chunk_index": chunk_index,
                    "reference": np.asarray(reference, dtype=np.float32),
                    "recovered_noise": np.asarray(recovered_noise, dtype=np.float32),
                    "injected_noise": np.asarray(injected_noise, dtype=np.float32),
                    "raw_actions": np.asarray(raw_action_chunk, dtype=np.float32),
                    "observed_actions": np.asarray(observed_action_chunk, dtype=np.float32),
                    "selected": bool(selected),
                    "prompt": str(element["prompt"]),
                    "observation_state": np.asarray(element["observation/state"], dtype=np.float32),
                    "observation_image": np.asarray(element["observation/image"], dtype=np.uint8),
                    "observation_wrist_image": np.asarray(element["observation/wrist_image"], dtype=np.uint8),
                    "recovered_noise_by_step": {
                        int(step): np.asarray(value, dtype=np.float32)
                        for step, value in recovered_noise_by_step.items()
                    },
                    "map_restart_recovered_noise": np.asarray(
                        latent_trace_payload["map_restart_recovered_noise"],
                        dtype=np.float32,
                    ),
                    "map_restart_energies": np.asarray(
                        latent_trace_payload["map_restart_energies"],
                        dtype=np.float32,
                    ),
                    "map_best_restart_index": int(latent_trace_payload["map_best_restart_index"]),
                    "posterior_recovered_noise_samples": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_samples"],
                        dtype=np.float32,
                    ),
                    "posterior_recovered_noise_mean": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_mean"],
                        dtype=np.float32,
                    ),
                    "posterior_recovered_noise_std": np.asarray(
                        latent_trace_payload["posterior_recovered_noise_std"],
                        dtype=np.float32,
                    ),
                    "posterior_restart_energies": np.asarray(
                        latent_trace_payload["posterior_restart_energies"],
                        dtype=np.float32,
                    ),
                    "posterior_best_energy": float(latent_trace_payload["posterior_best_energy"]),
                    "posterior_best_restart_index": int(latent_trace_payload["posterior_best_restart_index"]),
                    "posterior_init_mode": str(latent_trace_payload["posterior_init_mode"]),
                    "posterior_chain_init": str(latent_trace_payload["posterior_chain_init"]),
                }
                chunk_index += 1

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            telemetry.append(online_eval._extract_telemetry(obs))
            executed_actions.append(np.asarray(action, dtype=np.float32))
            probe_steps_remaining -= 1
            if active_segment is None or active_trace is None:
                raise RuntimeError("Expected active segment/trace while executing probe action plan.")
            active_segment["executed_steps"] += 1
            if active_segment["executed_steps"] == active_segment["planned_steps"] or done:
                executed_steps = int(active_segment["executed_steps"])
                execution_segments.append(
                    online_eval.ExecutionSegment(
                        chunk_index=active_segment["chunk_index"],
                        start_step=active_segment["start_step"],
                        end_step=active_segment["start_step"] + executed_steps,
                        executed_steps=executed_steps,
                    )
                )
                inversion_traces.append(
                    InversionChunkTrace(
                        chunk_index=int(active_trace["chunk_index"]),
                        executed_steps=executed_steps,
                        reference=np.asarray(active_trace["reference"], dtype=np.float32),
                        recovered_noise=np.asarray(active_trace["recovered_noise"], dtype=np.float32),
                        injected_noise=np.asarray(active_trace["injected_noise"], dtype=np.float32),
                        raw_actions=np.asarray(active_trace["raw_actions"], dtype=np.float32),
                        observed_actions=np.asarray(active_trace["observed_actions"], dtype=np.float32),
                        selected=bool(active_trace["selected"]),
                        prompt=str(active_trace["prompt"]),
                        observation_state=np.asarray(active_trace["observation_state"], dtype=np.float32),
                        observation_image=np.asarray(active_trace["observation_image"], dtype=np.uint8),
                        observation_wrist_image=np.asarray(active_trace["observation_wrist_image"], dtype=np.uint8),
                        recovered_noise_by_step={
                            int(step): np.asarray(value, dtype=np.float32)
                            for step, value in active_trace["recovered_noise_by_step"].items()
                        },
                        map_restart_recovered_noise=np.asarray(
                            active_trace["map_restart_recovered_noise"],
                            dtype=np.float32,
                        ),
                        map_restart_energies=np.asarray(
                            active_trace["map_restart_energies"],
                            dtype=np.float32,
                        ),
                        map_best_restart_index=int(active_trace["map_best_restart_index"]),
                        posterior_recovered_noise_samples=np.asarray(
                            active_trace["posterior_recovered_noise_samples"],
                            dtype=np.float32,
                        ),
                        posterior_recovered_noise_mean=np.asarray(
                            active_trace["posterior_recovered_noise_mean"],
                            dtype=np.float32,
                        ),
                        posterior_recovered_noise_std=np.asarray(
                            active_trace["posterior_recovered_noise_std"],
                            dtype=np.float32,
                        ),
                        posterior_restart_energies=np.asarray(
                            active_trace["posterior_restart_energies"],
                            dtype=np.float32,
                        ),
                        posterior_best_energy=float(active_trace["posterior_best_energy"]),
                        posterior_best_restart_index=int(active_trace["posterior_best_restart_index"]),
                        posterior_init_mode=str(active_trace["posterior_init_mode"]),
                        posterior_chain_init=str(active_trace["posterior_chain_init"]),
                    )
                )
                active_segment = None
                active_trace = None
    finally:
        env.close()

    result = online_eval.RolloutResult(
        telemetry=np.asarray(telemetry, dtype=np.float32),
        success=bool(len(telemetry) > 0 and not probe_steps_remaining),
        chunk_size=chunk_size,
        task_description=f"{probe_prompt} [probe:{args.probe_pattern}]",
        steps=len(telemetry),
        execution_segments=tuple(execution_segments),
        chunk_traces=(),
        executed_actions=np.asarray(executed_actions, dtype=np.float32) if executed_actions else np.zeros((0, 0), dtype=np.float32),
    )
    return result, inversion_traces


def _run_rollout_with_inversion(
    policy,
    detector_policy=None,
    *,
    task,
    initial_state: np.ndarray,
    args: argparse.Namespace,
    runtime_modules: dict[str, Any],
    episode_nonce: int,
) -> tuple[online_eval.RolloutResult, list[InversionChunkTrace]]:
    detector_policy = detector_policy or policy
    if args.eval_mode == "probe_verification":
        return _run_probe_rollout_with_inversion(
            policy,
            detector_policy,
            task=task,
            initial_state=initial_state,
            args=args,
            runtime_modules=runtime_modules,
            episode_nonce=episode_nonce,
        )
    return _run_task_rollout_with_inversion(
        policy,
        detector_policy,
        task=task,
        initial_state=initial_state,
        args=args,
        runtime_modules=runtime_modules,
        episode_nonce=episode_nonce,
    )


def _pad_trace_sequences(
    traces: list[InversionChunkTrace],
    *,
    attr: str,
    pad_to: int | None = None,
    pad_value: float = np.nan,
) -> np.ndarray:
    if not traces:
        return np.zeros((0, 0, 0), dtype=np.float32)
    sequences = [np.asarray(getattr(trace, attr), dtype=np.float32) for trace in traces]
    widths = {seq.shape[1] for seq in sequences if seq.ndim == 2 and seq.shape[1] > 0}
    if not widths:
        return np.zeros((len(traces), 0, 0), dtype=np.float32)
    if len(widths) != 1:
        raise ValueError(f"Inconsistent sequence widths for {attr}: {sorted(widths)}")
    width = next(iter(widths))
    target_len = max((seq.shape[0] for seq in sequences), default=0) if pad_to is None else int(pad_to)
    padded = np.full((len(traces), target_len, width), pad_value, dtype=np.float32)
    for idx, seq in enumerate(sequences):
        valid = min(seq.shape[0], target_len)
        if valid > 0:
            padded[idx, :valid] = seq[:valid]
    return padded


def _stack_map_restart_recovered_noise(
    traces: list[InversionChunkTrace],
    *,
    map_num_starts: int,
) -> np.ndarray:
    if not traces:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    recovered_shapes = {
        tuple(int(dim) for dim in np.asarray(trace.recovered_noise, dtype=np.float32).shape)
        for trace in traces
    }
    if len(recovered_shapes) != 1:
        raise ValueError(f"Inconsistent recovered_noise shapes for map restart save: {sorted(recovered_shapes)}")
    raw_shape = next(iter(recovered_shapes))
    target_restarts = max(
        int(map_num_starts),
        max(
            (
                int(np.asarray(trace.map_restart_recovered_noise, dtype=np.float32).shape[0])
                if np.asarray(trace.map_restart_recovered_noise, dtype=np.float32).ndim >= 1
                else 0
            )
            for trace in traces
        ),
    )
    padded = np.zeros((len(traces), target_restarts, *raw_shape), dtype=np.float32)
    for idx, trace in enumerate(traces):
        restart_noise = np.asarray(trace.map_restart_recovered_noise, dtype=np.float32)
        if restart_noise.size == 0:
            continue
        expected_ndim = len(raw_shape) + 1
        if restart_noise.ndim != expected_ndim:
            raise ValueError(
                f"Unexpected map_restart_recovered_noise ndim={restart_noise.ndim} "
                f"for trace {trace.chunk_index}; expected {expected_ndim}"
            )
        if tuple(int(dim) for dim in restart_noise.shape[1:]) != raw_shape:
            raise ValueError(
                f"Unexpected map_restart_recovered_noise shape={restart_noise.shape} "
                f"for trace {trace.chunk_index}; expected (*, {raw_shape})"
            )
        valid = min(int(restart_noise.shape[0]), target_restarts)
        padded[idx, :valid] = restart_noise[:valid]
    return padded


def _stack_map_restart_energies(
    traces: list[InversionChunkTrace],
    *,
    map_num_starts: int,
) -> np.ndarray:
    if not traces:
        return np.zeros((0, 0), dtype=np.float32)
    target_restarts = max(
        int(map_num_starts),
        max(int(np.asarray(trace.map_restart_energies, dtype=np.float32).shape[0]) for trace in traces),
    )
    padded = np.full((len(traces), target_restarts), np.nan, dtype=np.float32)
    for idx, trace in enumerate(traces):
        energies = np.asarray(trace.map_restart_energies, dtype=np.float32)
        valid = min(int(energies.shape[0]), target_restarts)
        if valid > 0:
            padded[idx, :valid] = energies[:valid]
    return padded


def _save_inversion_rollout(
    *,
    save_dir: pathlib.Path,
    task_id: int,
    episode_idx: int,
    episode_nonce: int,
    variant: str,
    result: online_eval.RolloutResult,
    inversion_traces: list[InversionChunkTrace],
    args: argparse.Namespace,
) -> pathlib.Path:
    save_dir = save_dir / args.eval_mode
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"task_{task_id:03d}_episode_{episode_idx:03d}_{variant}.npz"
    effective_score_step_scope = args.score_step_scope
    map_restart_recovered_noise = _stack_map_restart_recovered_noise(
        inversion_traces,
        map_num_starts=int(getattr(args, "map_num_starts", 0)),
    )
    map_restart_energies = _stack_map_restart_energies(
        inversion_traces,
        map_num_starts=int(getattr(args, "map_num_starts", 0)),
    )
    cached_step_counts = sorted(
        {
            int(step)
            for trace in inversion_traces
            for step in trace.recovered_noise_by_step
        }
    )
    payload = {
        "telemetry": result.telemetry,
        "success": np.asarray(result.success),
        "chunk_size": np.asarray(result.chunk_size, dtype=np.int32),
        "steps": np.asarray(result.steps, dtype=np.int32),
        "task_id": np.asarray(task_id, dtype=np.int32),
        "episode_idx": np.asarray(episode_idx, dtype=np.int32),
        "episode_nonce": np.asarray(episode_nonce, dtype=np.int64),
        "variant": np.asarray(variant),
        "task_description": np.asarray(result.task_description),
        "task_suite_name": np.asarray(args.task_suite_name),
        "eval_mode": np.asarray(args.eval_mode),
        "max_rollout_steps": np.asarray(-1 if args.max_rollout_steps is None else args.max_rollout_steps, dtype=np.int32),
        "detector": np.asarray(args.detector),
        "reference_mode": np.asarray(args.reference_mode),
        "sample_rate_hz": np.asarray(getattr(args, "sample_rate_hz", 20.0), dtype=np.float32),
        "secret_key": np.asarray(getattr(args, "secret_key", 17), dtype=np.int32),
        "beta": np.asarray(getattr(args, "beta", 0.0), dtype=np.float32),
        "freq_min_hz": np.asarray(getattr(args, "freq_min_hz", 1.0), dtype=np.float32),
        "freq_max_hz": np.asarray(getattr(args, "freq_max_hz", 2.0), dtype=np.float32),
        "n_tones": np.asarray(getattr(args, "n_tones", 4), dtype=np.int32),
        "null_decoy_count": np.asarray(getattr(args, "null_decoy_count", 32), dtype=np.int32),
        "subspace_rank": np.asarray(
            -1 if getattr(args, "subspace_rank", None) is None else getattr(args, "subspace_rank"),
            dtype=np.int32,
        ),
        "probe_duration_sec": np.asarray(args.probe_duration_sec, dtype=np.float32),
        "probe_pattern": np.asarray(args.probe_pattern),
        "probe_amplitude": np.asarray(args.probe_amplitude, dtype=np.float32),
        "probe_axis_mode": np.asarray(args.probe_axis_mode),
        "probe_gripper_mode": np.asarray(args.probe_gripper_mode),
        "probe_replan_interval": np.asarray(args.probe_replan_interval, dtype=np.int32),
        "probe_speed_scale": np.asarray(args.probe_speed_scale, dtype=np.float32),
        "chunk_selection_strategy": np.asarray(args.chunk_selection_strategy),
        "chunk_selection_period": np.asarray(args.chunk_selection_period, dtype=np.int32),
        "chunk_selection_count": np.asarray(args.chunk_selection_count, dtype=np.int32),
        "chunk_selection_total_slots": np.asarray(
            -1 if args.chunk_selection_total_slots is None else args.chunk_selection_total_slots,
            dtype=np.int32,
        ),
        "max_score_windows": np.asarray(-1 if args.max_score_windows is None else args.max_score_windows, dtype=np.int32),
        "window_aggregator": np.asarray(args.window_aggregator),
        "score_step_scope": np.asarray(effective_score_step_scope),
        "requested_score_step_scope": np.asarray(args.score_step_scope),
        "num_inversion_steps": np.asarray(args.num_inversion_steps, dtype=np.int32),
        "inversion_method": np.asarray(args.inversion_method),
        "refinement_steps": np.asarray(args.refinement_steps, dtype=np.int32),
        "refinement_learning_rate": np.asarray(args.refinement_learning_rate, dtype=np.float32),
        "refinement_latent_l2": np.asarray(args.refinement_latent_l2, dtype=np.float32),
        "refinement_init_l2": np.asarray(args.refinement_init_l2, dtype=np.float32),
        "fm_channel_inverse": np.asarray(bool(getattr(args, "fm_channel_inverse", False))),
        "fm_full_latent_map": np.asarray(bool(getattr(args, "fm_full_latent_map", False))),
        "fm_latent_map": np.asarray(bool(getattr(args, "fm_latent_map", False))),
        "fm_latent_posterior": np.asarray(bool(getattr(args, "fm_latent_posterior", False))),
        "obs_sigma": np.asarray(getattr(args, "obs_sigma", 1e-4), dtype=np.float32),
        "fm_guide_scale": np.asarray(getattr(args, "fm_guide_scale", 0.5), dtype=np.float32),
        "fm_guide_schedule": np.asarray(getattr(args, "fm_guide_schedule", "linear_decay")),
        "latent_map_iters": np.asarray(getattr(args, "latent_map_iters", 100), dtype=np.int32),
        "latent_map_lr": np.asarray(getattr(args, "latent_map_lr", 1e-1), dtype=np.float32),
        "latent_prior_weight": np.asarray(getattr(args, "latent_prior_weight", 1.0), dtype=np.float32),
        "map_num_starts": np.asarray(getattr(args, "map_num_starts", 1), dtype=np.int32),
        "map_random_seed": np.asarray(getattr(args, "map_random_seed", 0), dtype=np.int32),
        "posterior_step_size": np.asarray(getattr(args, "posterior_step_size", 1e-3), dtype=np.float32),
        "posterior_burnin": np.asarray(getattr(args, "posterior_burnin", 100), dtype=np.int32),
        "posterior_thinning": np.asarray(getattr(args, "posterior_thinning", 50), dtype=np.int32),
        "posterior_num_samples": np.asarray(getattr(args, "posterior_num_samples", 8), dtype=np.int32),
        "posterior_map_tether_weight": np.asarray(getattr(args, "posterior_map_tether_weight", 1.0), dtype=np.float32),
        "posterior_grad_clip_norm": np.asarray(getattr(args, "posterior_grad_clip_norm", 100.0), dtype=np.float32),
        "latent_init_from_bridge": np.asarray(bool(getattr(args, "latent_init_from_bridge", False))),
        "segment_chunk_index": np.asarray([segment.chunk_index for segment in result.execution_segments], dtype=np.int32),
        "segment_start_step": np.asarray([segment.start_step for segment in result.execution_segments], dtype=np.int32),
        "segment_end_step": np.asarray([segment.end_step for segment in result.execution_segments], dtype=np.int32),
        "segment_executed_steps": np.asarray(
            [segment.executed_steps for segment in result.execution_segments],
            dtype=np.int32,
        ),
    }
    if result.executed_actions.size:
        payload["executed_actions"] = np.asarray(result.executed_actions, dtype=np.float32)
    if inversion_traces:
        chunk_horizon = max((int(np.asarray(trace.raw_actions).shape[0]) for trace in inversion_traces), default=0)
        payload.update(
            {
                "chunk_chunk_index": np.asarray([trace.chunk_index for trace in inversion_traces], dtype=np.int32),
                "chunk_executed_steps": np.asarray([trace.executed_steps for trace in inversion_traces], dtype=np.int32),
                "chunk_selected": np.asarray([trace.selected for trace in inversion_traces], dtype=bool),
                "chunk_prompt": np.asarray([trace.prompt for trace in inversion_traces]),
                "chunk_observation_state": np.asarray(
                    [trace.observation_state for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_observation_image": np.asarray(
                    [trace.observation_image for trace in inversion_traces],
                    dtype=np.uint8,
                ),
                "chunk_observation_wrist_image": np.asarray(
                    [trace.observation_wrist_image for trace in inversion_traces],
                    dtype=np.uint8,
                ),
                "chunk_reference": np.asarray([trace.reference for trace in inversion_traces], dtype=np.float32),
                "chunk_recovered_noise": np.asarray(
                    [trace.recovered_noise for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_map_restart_recovered_noise": map_restart_recovered_noise,
                "chunk_map_restart_energies": map_restart_energies,
                "chunk_map_best_restart_index": np.asarray(
                    [trace.map_best_restart_index for trace in inversion_traces],
                    dtype=np.int32,
                ),
                "chunk_injected_noise": np.asarray(
                    [trace.injected_noise for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_raw_actions": np.asarray([trace.raw_actions for trace in inversion_traces], dtype=np.float32),
                "chunk_observed_actions": np.asarray([trace.observed_actions for trace in inversion_traces], dtype=np.float32),
                "chunk_posterior_recovered_noise_samples": np.asarray(
                    [trace.posterior_recovered_noise_samples for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_posterior_recovered_noise_mean": np.asarray(
                    [trace.posterior_recovered_noise_mean for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_posterior_recovered_noise_std": np.asarray(
                    [trace.posterior_recovered_noise_std for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_posterior_best_energy": np.asarray(
                    [trace.posterior_best_energy for trace in inversion_traces],
                    dtype=np.float32,
                ),
                "chunk_posterior_best_restart_index": np.asarray(
                    [trace.posterior_best_restart_index for trace in inversion_traces],
                    dtype=np.int32,
                ),
                "chunk_posterior_init_mode": np.asarray([trace.posterior_init_mode for trace in inversion_traces]),
                "chunk_posterior_chain_init": np.asarray([trace.posterior_chain_init for trace in inversion_traces]),
            }
        )
        if cached_step_counts:
            payload["chunk_cached_inversion_steps"] = np.asarray(cached_step_counts, dtype=np.int32)
            payload["chunk_recovered_noise_by_step"] = np.asarray(
                [
                    [
                        trace.recovered_noise_by_step.get(step, trace.recovered_noise)
                        for step in cached_step_counts
                    ]
                    for trace in inversion_traces
                ],
                dtype=np.float32,
            )
    np.savez_compressed(out_path, **payload)
    return out_path


def _saved_inversion_rollout_path(
    *,
    save_dir: pathlib.Path,
    eval_mode: str,
    task_id: int,
    episode_idx: int,
    variant: str,
) -> pathlib.Path:
    return save_dir / eval_mode / f"task_{task_id:03d}_episode_{episode_idx:03d}_{variant}.npz"


def _load_saved_inversion_traces(payload: np.lib.npyio.NpzFile) -> list[InversionChunkTrace]:
    if "chunk_chunk_index" not in payload:
        return []
    cached_steps = tuple(int(step) for step in payload["chunk_cached_inversion_steps"]) if "chunk_cached_inversion_steps" in payload else ()
    cached_by_step = payload["chunk_recovered_noise_by_step"] if "chunk_recovered_noise_by_step" in payload else None
    prompts = payload["chunk_prompt"] if "chunk_prompt" in payload else np.asarray([""] * len(payload["chunk_chunk_index"]))
    states = payload["chunk_observation_state"] if "chunk_observation_state" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0), dtype=np.float32)
    images = payload["chunk_observation_image"] if "chunk_observation_image" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0, 0), dtype=np.uint8)
    wrist_images = payload["chunk_observation_wrist_image"] if "chunk_observation_wrist_image" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0, 0), dtype=np.uint8)
    observed_actions = payload["chunk_observed_actions"] if "chunk_observed_actions" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0), dtype=np.float32)
    map_restart_recovered_noise = payload["chunk_map_restart_recovered_noise"] if "chunk_map_restart_recovered_noise" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0, 0), dtype=np.float32)
    map_restart_energies = payload["chunk_map_restart_energies"] if "chunk_map_restart_energies" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0), dtype=np.float32)
    map_best_restart_index = payload["chunk_map_best_restart_index"] if "chunk_map_best_restart_index" in payload else np.full((len(payload["chunk_chunk_index"]),), -1, dtype=np.int32)
    posterior_samples = payload["chunk_posterior_recovered_noise_samples"] if "chunk_posterior_recovered_noise_samples" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0, 0), dtype=np.float32)
    posterior_mean = payload["chunk_posterior_recovered_noise_mean"] if "chunk_posterior_recovered_noise_mean" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0), dtype=np.float32)
    posterior_std = payload["chunk_posterior_recovered_noise_std"] if "chunk_posterior_recovered_noise_std" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0), dtype=np.float32)
    posterior_energies = payload["chunk_posterior_restart_energies"] if "chunk_posterior_restart_energies" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0), dtype=np.float32)
    posterior_best_energy = payload["chunk_posterior_best_energy"] if "chunk_posterior_best_energy" in payload else np.full((len(payload["chunk_chunk_index"]),), np.nan, dtype=np.float32)
    posterior_best_restart_index = payload["chunk_posterior_best_restart_index"] if "chunk_posterior_best_restart_index" in payload else np.full((len(payload["chunk_chunk_index"]),), -1, dtype=np.int32)
    posterior_init_mode = payload["chunk_posterior_init_mode"] if "chunk_posterior_init_mode" in payload else np.asarray([""] * len(payload["chunk_chunk_index"]))
    posterior_chain_init = payload["chunk_posterior_chain_init"] if "chunk_posterior_chain_init" in payload else np.asarray([""] * len(payload["chunk_chunk_index"]))
    traces: list[InversionChunkTrace] = []
    for index, (chunk_index, executed_steps, selected, prompt, state, image, wrist_image, observed_action, reference, recovered_noise, injected_noise, raw_actions) in enumerate(
        zip(
            payload["chunk_chunk_index"],
            payload["chunk_executed_steps"],
            payload["chunk_selected"],
            prompts,
            states,
            images,
            wrist_images,
            observed_actions,
            payload["chunk_reference"],
            payload["chunk_recovered_noise"],
            payload["chunk_injected_noise"],
            payload["chunk_raw_actions"],
            strict=True,
        )
    ):
        recovered_noise_by_step = {}
        if cached_by_step is not None:
            for step_offset, step_count in enumerate(cached_steps):
                recovered_noise_by_step[int(step_count)] = np.asarray(cached_by_step[index, step_offset], dtype=np.float32)
        traces.append(
            InversionChunkTrace(
                chunk_index=int(chunk_index),
                executed_steps=int(executed_steps),
                reference=np.asarray(reference, dtype=np.float32),
                recovered_noise=np.asarray(recovered_noise, dtype=np.float32),
                injected_noise=np.asarray(injected_noise, dtype=np.float32),
                raw_actions=np.asarray(raw_actions, dtype=np.float32),
                observed_actions=np.asarray(observed_action, dtype=np.float32),
                selected=bool(selected),
                prompt=str(prompt),
                observation_state=np.asarray(state, dtype=np.float32),
                observation_image=np.asarray(image, dtype=np.uint8),
                observation_wrist_image=np.asarray(wrist_image, dtype=np.uint8),
                recovered_noise_by_step=recovered_noise_by_step,
                map_restart_recovered_noise=np.asarray(map_restart_recovered_noise[index], dtype=np.float32),
                map_restart_energies=np.asarray(map_restart_energies[index], dtype=np.float32),
                map_best_restart_index=int(map_best_restart_index[index]),
                posterior_recovered_noise_samples=np.asarray(posterior_samples[index], dtype=np.float32),
                posterior_recovered_noise_mean=np.asarray(posterior_mean[index], dtype=np.float32),
                posterior_recovered_noise_std=np.asarray(posterior_std[index], dtype=np.float32),
                posterior_restart_energies=np.asarray(posterior_energies[index], dtype=np.float32),
                posterior_best_energy=float(posterior_best_energy[index]),
                posterior_best_restart_index=int(posterior_best_restart_index[index]),
                posterior_init_mode=str(posterior_init_mode[index]),
                posterior_chain_init=str(posterior_chain_init[index]),
            )
        )
    return traces


def _load_saved_inversion_rollout(
    path: pathlib.Path,
) -> tuple[online_eval.RolloutResult, list[InversionChunkTrace], int]:
    payload = np.load(path)
    result = online_eval.RolloutResult(
        telemetry=np.asarray(payload["telemetry"], dtype=np.float32),
        success=bool(payload["success"].item()),
        chunk_size=int(payload["chunk_size"].item()),
        task_description=str(payload["task_description"].item()),
        steps=int(payload["steps"].item()),
        execution_segments=tuple(
            online_eval.ExecutionSegment(
                chunk_index=int(chunk_index),
                start_step=int(start_step),
                end_step=int(end_step),
                executed_steps=int(executed_steps),
            )
            for chunk_index, start_step, end_step, executed_steps in zip(
                payload["segment_chunk_index"],
                payload["segment_start_step"],
                payload["segment_end_step"],
                payload["segment_executed_steps"],
                strict=True,
            )
        ),
        chunk_traces=(),
        executed_actions=np.asarray(payload["executed_actions"], dtype=np.float32)
        if "executed_actions" in payload
        else np.zeros((0, 0), dtype=np.float32),
        output_reference=np.asarray(payload["output_reference"], dtype=np.float32)
        if "output_reference" in payload
        else np.zeros((0, 0), dtype=np.float32),
        clip_fraction=float(payload["clip_fraction"].item()) if "clip_fraction" in payload else 0.0,
        saturation_fraction=float(payload["saturation_fraction"].item()) if "saturation_fraction" in payload else 0.0,
        mean_action_l2=float(payload["mean_action_l2"].item()) if "mean_action_l2" in payload else 0.0,
    )
    return result, _load_saved_inversion_traces(payload), int(payload["episode_nonce"].item())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    _validate_args(args)

    runtime_modules = online_eval._load_runtime_modules()
    train_config = runtime_modules["training_config"].get_config(args.config_name)
    detector_config_name = args.detector_config_name or args.config_name
    detector_checkpoint_dir = args.detector_checkpoint_dir or args.checkpoint_dir
    detector_train_config = runtime_modules["training_config"].get_config(detector_config_name)
    use_separate_detector_policy = (
        detector_config_name != args.config_name
        or str(detector_checkpoint_dir) != str(args.checkpoint_dir)
    )
    if detector_train_config.model.action_dim != train_config.model.action_dim:
        raise ValueError(
            "Detector model action_dim must match rollout model action_dim: "
            f"rollout={train_config.model.action_dim}, detector={detector_train_config.model.action_dim}"
        )
    if detector_train_config.model.action_horizon != train_config.model.action_horizon:
        raise ValueError(
            "Detector model action_horizon must match rollout model action_horizon: "
            f"rollout={train_config.model.action_horizon}, detector={detector_train_config.model.action_horizon}"
        )

    benchmark_dict = runtime_modules["benchmark"].get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_start = int(args.task_offset)
    task_stop = min(task_suite.n_tasks, task_start + args.num_tasks)
    if task_start >= task_stop:
        raise ValueError(
            f"Requested empty task range: offset={args.task_offset}, num_tasks={args.num_tasks}, n_tasks={task_suite.n_tasks}"
        )

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    wrong_key_scores: list[float] = []
    plain_recovery_rms: list[float] = []
    marked_recovery_rms: list[float] = []
    episode_records: list[EpisodeScoreRecord] = []
    freq_range = (args.freq_min_hz, args.freq_max_hz)
    effective_score_step_scope = args.score_step_scope
    for task_id in range(task_start, task_stop):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        num_trials = min(args.num_trials_per_task, len(initial_states))
        task_plain_episode_cache: dict[int, tuple[online_eval.RolloutResult, list[InversionChunkTrace], int]] = {}
        pending_plain_episode_indices: list[int] = []

        for episode_idx in range(num_trials):
            episode_nonce = (task_id + 1) * 100_000 + episode_idx
            loaded = False
            if args.resume_from_rollouts and args.save_rollout_dir is not None:
                plain_path = _saved_inversion_rollout_path(
                    save_dir=args.save_rollout_dir,
                    eval_mode=args.eval_mode,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    variant="plain",
                )
                if plain_path.exists():
                    plain_result, plain_inversion, saved_nonce = _load_saved_inversion_rollout(plain_path)
                    if saved_nonce == episode_nonce:
                        task_plain_episode_cache[episode_idx] = (plain_result, plain_inversion, saved_nonce)
                        loaded = True
                        logging.info(
                            "Resuming plain rollout from saved file task_id=%s episode_idx=%s path=%s",
                            task_id,
                            episode_idx,
                            plain_path,
                        )
                    else:
                        logging.warning(
                            "Ignoring saved plain rollout with mismatched nonce task_id=%s episode_idx=%s expected=%s found=%s path=%s",
                            task_id,
                            episode_idx,
                            episode_nonce,
                            saved_nonce,
                            plain_path,
                        )
            if not loaded:
                pending_plain_episode_indices.append(episode_idx)

        if pending_plain_episode_indices:
            plain_policy = runtime_modules["policy_config"].create_trained_policy(train_config, args.checkpoint_dir)
            plain_detector_policy = (
                runtime_modules["policy_config"].create_trained_policy(
                    detector_train_config,
                    detector_checkpoint_dir,
                )
                if use_separate_detector_policy
                else plain_policy
            )
            try:
                logging.info(
                    "Plain inversion eval mode=%s task_id=%s episodes=%s pending=%s",
                    args.eval_mode,
                    task_id,
                    num_trials,
                    len(pending_plain_episode_indices),
                )
                if use_separate_detector_policy:
                    logging.info(
                        "Using separate detector model for plain inversion config=%s checkpoint=%s",
                        detector_config_name,
                        detector_checkpoint_dir,
                    )
                for episode_idx in pending_plain_episode_indices:
                    episode_nonce = (task_id + 1) * 100_000 + episode_idx
                    initial_state = initial_states[episode_idx]
                    plain_result, plain_inversion = _run_rollout_with_inversion(
                        plain_policy,
                        plain_detector_policy,
                        task=task,
                        initial_state=initial_state,
                        args=args,
                        runtime_modules=runtime_modules,
                        episode_nonce=episode_nonce,
                    )
                    task_plain_episode_cache[episode_idx] = (plain_result, plain_inversion, episode_nonce)
                    if args.save_rollout_dir is not None:
                        _save_inversion_rollout(
                            save_dir=args.save_rollout_dir,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            episode_nonce=episode_nonce,
                            variant="plain",
                            result=plain_result,
                            inversion_traces=plain_inversion,
                            args=args,
                        )
            finally:
                if use_separate_detector_policy:
                    _release_policy(plain_detector_policy)
                _release_policy(plain_policy)
        else:
            logging.info("All plain episodes already saved for task_id=%s; reusing cached rollouts.", task_id)

        pending_marked_episode_indices: list[int] = []
        loaded_marked_episode_cache: dict[int, tuple[online_eval.RolloutResult, list[InversionChunkTrace], int]] = {}
        for episode_idx in range(num_trials):
            episode_nonce = (task_id + 1) * 100_000 + episode_idx
            loaded = False
            if args.resume_from_rollouts and args.save_rollout_dir is not None:
                marked_path = _saved_inversion_rollout_path(
                    save_dir=args.save_rollout_dir,
                    eval_mode=args.eval_mode,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    variant="watermarked",
                )
                if marked_path.exists():
                    marked_result, marked_inversion, saved_nonce = _load_saved_inversion_rollout(marked_path)
                    if saved_nonce == episode_nonce:
                        loaded_marked_episode_cache[episode_idx] = (marked_result, marked_inversion, saved_nonce)
                        loaded = True
                        logging.info(
                            "Resuming marked rollout from saved file task_id=%s episode_idx=%s path=%s",
                            task_id,
                            episode_idx,
                            marked_path,
                        )
                    else:
                        logging.warning(
                            "Ignoring saved marked rollout with mismatched nonce task_id=%s episode_idx=%s expected=%s found=%s path=%s",
                            task_id,
                            episode_idx,
                            episode_nonce,
                            saved_nonce,
                            marked_path,
                        )
            if not loaded:
                pending_marked_episode_indices.append(episode_idx)

        marked_policy = None
        marked_detector_policy = None
        if pending_marked_episode_indices:
            marked_policy = runtime_modules["policy_config"].create_trained_policy(
                train_config,
                args.checkpoint_dir,
                watermark_config=online_eval._make_watermark_config(args, telemetry_dim=int(train_config.model.action_dim)),
            )
            marked_detector_policy = (
                runtime_modules["policy_config"].create_trained_policy(
                    detector_train_config,
                    detector_checkpoint_dir,
                )
                if use_separate_detector_policy
                else marked_policy
            )
        try:
            logging.info(
                "Marked inversion eval mode=%s task_id=%s episodes=%s pending=%s",
                args.eval_mode,
                task_id,
                num_trials,
                len(pending_marked_episode_indices),
            )
            if pending_marked_episode_indices and use_separate_detector_policy:
                logging.info(
                    "Using separate detector model for marked inversion config=%s checkpoint=%s",
                    detector_config_name,
                    detector_checkpoint_dir,
                )
            for episode_idx in range(num_trials):
                episode_nonce = (task_id + 1) * 100_000 + episode_idx
                initial_state = initial_states[episode_idx]
                plain_result, plain_inversion, cached_nonce = task_plain_episode_cache[episode_idx]
                if cached_nonce != episode_nonce:
                    raise RuntimeError("Episode nonce mismatch while pairing plain and watermarked runs.")
                reference_config = online_eval._make_watermark_config(
                    args,
                    telemetry_dim=int(train_config.model.action_dim),
                )
                wrong_key_config = _wrong_key_reference_config(reference_config)
                if episode_idx in loaded_marked_episode_cache:
                    marked_result, marked_inversion, saved_nonce = loaded_marked_episode_cache[episode_idx]
                    if saved_nonce != episode_nonce:
                        raise RuntimeError("Episode nonce mismatch while loading saved watermarked run.")
                else:
                    if marked_policy is None:
                        raise RuntimeError("Marked policy is unavailable for pending episode execution.")
                    if marked_detector_policy is None:
                        raise RuntimeError("Marked detector policy is unavailable for pending episode execution.")
                    marked_result, marked_inversion = _run_rollout_with_inversion(
                        marked_policy,
                        marked_detector_policy,
                        task=task,
                        initial_state=initial_state,
                        args=args,
                        runtime_modules=runtime_modules,
                        episode_nonce=episode_nonce,
                    )
                    if args.save_rollout_dir is not None:
                        _save_inversion_rollout(
                            save_dir=args.save_rollout_dir,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            episode_nonce=episode_nonce,
                            variant="watermarked",
                            result=marked_result,
                            inversion_traces=marked_inversion,
                            args=args,
                        )

                episode_scorer = (
                    _score_episode_from_noise_samples if args.fm_latent_posterior else _score_episode_from_noise_traces
                )
                if args.fm_latent_posterior:
                    plain_summary, plain_chunk_scores = _score_episode_from_noise_samples_summary(
                        plain_inversion,
                        detector=args.detector,
                        reference_mode=args.reference_mode,
                        sample_rate_hz=args.sample_rate_hz,
                        freq_range=freq_range,
                        aggregator=args.window_aggregator,
                        score_step_scope=effective_score_step_scope,
                        max_windows=args.max_score_windows,
                        reference_config=reference_config,
                        episode_nonce=episode_nonce,
                        null_decoy_count=args.null_decoy_count,
                        subspace_rank=args.subspace_rank,
                    )
                    plain_score = float(plain_summary["episode_score_mean"])
                    marked_summary, marked_chunk_scores = _score_episode_from_noise_samples_summary(
                        marked_inversion,
                        detector=args.detector,
                        reference_mode=args.reference_mode,
                        sample_rate_hz=args.sample_rate_hz,
                        freq_range=freq_range,
                        aggregator=args.window_aggregator,
                        score_step_scope=effective_score_step_scope,
                        max_windows=args.max_score_windows,
                        reference_config=reference_config,
                        episode_nonce=episode_nonce,
                        null_decoy_count=args.null_decoy_count,
                        subspace_rank=args.subspace_rank,
                    )
                    marked_score = float(marked_summary["episode_score_mean"])
                else:
                    plain_score, plain_chunk_scores = episode_scorer(
                        plain_inversion,
                        detector=args.detector,
                        reference_mode=args.reference_mode,
                        sample_rate_hz=args.sample_rate_hz,
                        freq_range=freq_range,
                        aggregator=args.window_aggregator,
                        score_step_scope=effective_score_step_scope,
                        max_windows=args.max_score_windows,
                        reference_config=reference_config,
                        episode_nonce=episode_nonce,
                        null_decoy_count=args.null_decoy_count,
                        subspace_rank=args.subspace_rank,
                    )
                    plain_summary = _summarize_episode_score_samples(
                        np.zeros((0,), dtype=np.float32),
                        fallback_score=plain_score,
                    )
                    marked_score, marked_chunk_scores = episode_scorer(
                        marked_inversion,
                        detector=args.detector,
                        reference_mode=args.reference_mode,
                        sample_rate_hz=args.sample_rate_hz,
                        freq_range=freq_range,
                        aggregator=args.window_aggregator,
                        score_step_scope=effective_score_step_scope,
                        max_windows=args.max_score_windows,
                        reference_config=reference_config,
                        episode_nonce=episode_nonce,
                        null_decoy_count=args.null_decoy_count,
                        subspace_rank=args.subspace_rank,
                    )
                    marked_summary = _summarize_episode_score_samples(
                        np.zeros((0,), dtype=np.float32),
                        fallback_score=marked_score,
                    )
                negative_scores.append(plain_score)
                positive_scores.append(marked_score)
                if args.detector in {"wmf", "ace"}:
                    null_configs = _wrong_key_reference_configs(reference_config, count=args.null_decoy_count + 1)
                    decoy_config = null_configs[0]
                    decoy_null_configs = null_configs[1:]
                    wrong_key_scores.append(
                        episode_scorer(
                            _retarget_chunk_references(
                                plain_inversion,
                                reference_config=decoy_config,
                                sample_rate_hz=args.sample_rate_hz,
                                episode_nonce=episode_nonce,
                            ),
                            detector=args.detector,
                            reference_mode=args.reference_mode,
                            sample_rate_hz=args.sample_rate_hz,
                            freq_range=freq_range,
                            aggregator=args.window_aggregator,
                            score_step_scope=effective_score_step_scope,
                            max_windows=args.max_score_windows,
                            reference_config=reference_config,
                            episode_nonce=episode_nonce,
                            null_decoy_count=args.null_decoy_count,
                            subspace_rank=args.subspace_rank,
                            null_reference_configs=decoy_null_configs,
                        )[0]
                    )
                    wrong_key_scores.append(
                        episode_scorer(
                            _retarget_chunk_references(
                                marked_inversion,
                                reference_config=decoy_config,
                                sample_rate_hz=args.sample_rate_hz,
                                episode_nonce=episode_nonce,
                            ),
                            detector=args.detector,
                            reference_mode=args.reference_mode,
                            sample_rate_hz=args.sample_rate_hz,
                            freq_range=freq_range,
                            aggregator=args.window_aggregator,
                            score_step_scope=effective_score_step_scope,
                            max_windows=args.max_score_windows,
                            reference_config=reference_config,
                            episode_nonce=episode_nonce,
                            null_decoy_count=args.null_decoy_count,
                            subspace_rank=args.subspace_rank,
                            null_reference_configs=decoy_null_configs,
                        )[0]
                    )
                else:
                    wrong_key_scores.append(
                        episode_scorer(
                            _retarget_chunk_references(
                                plain_inversion,
                                reference_config=wrong_key_config,
                                sample_rate_hz=args.sample_rate_hz,
                                episode_nonce=episode_nonce,
                            ),
                            detector=args.detector,
                            reference_mode=args.reference_mode,
                            sample_rate_hz=args.sample_rate_hz,
                            freq_range=freq_range,
                            aggregator=args.window_aggregator,
                            score_step_scope=effective_score_step_scope,
                            max_windows=args.max_score_windows,
                        )[0]
                    )
                    wrong_key_scores.append(
                        episode_scorer(
                            _retarget_chunk_references(
                                marked_inversion,
                                reference_config=wrong_key_config,
                                sample_rate_hz=args.sample_rate_hz,
                                episode_nonce=episode_nonce,
                            ),
                            detector=args.detector,
                            reference_mode=args.reference_mode,
                            sample_rate_hz=args.sample_rate_hz,
                            freq_range=freq_range,
                            aggregator=args.window_aggregator,
                            score_step_scope=effective_score_step_scope,
                            max_windows=args.max_score_windows,
                        )[0]
                    )
                plain_rms = _episode_recovery_rms(plain_inversion, max_windows=args.max_score_windows)
                marked_rms = _episode_recovery_rms(marked_inversion, max_windows=args.max_score_windows)
                plain_recovery_rms.append(plain_rms)
                marked_recovery_rms.append(marked_rms)
                episode_records.append(
                    EpisodeScoreRecord(
                        task_id=task_id,
                        episode_idx=episode_idx,
                        eval_mode=args.eval_mode,
                        variant="plain",
                        episode_score=float(plain_score),
                        recovery_rms=float(plain_rms),
                        selected_window_count=int(plain_chunk_scores.size),
                        chunk_scores=np.asarray(plain_chunk_scores, dtype=np.float32),
                        episode_score_std=float(plain_summary["episode_score_std"]),
                        episode_score_q05=float(plain_summary["episode_score_q05"]),
                        episode_score_q95=float(plain_summary["episode_score_q95"]),
                        posterior_sample_count=int(plain_summary["posterior_sample_count"]),
                    )
                )
                episode_records.append(
                    EpisodeScoreRecord(
                        task_id=task_id,
                        episode_idx=episode_idx,
                        eval_mode=args.eval_mode,
                        variant="watermarked",
                        episode_score=float(marked_score),
                        recovery_rms=float(marked_rms),
                        selected_window_count=int(marked_chunk_scores.size),
                        chunk_scores=np.asarray(marked_chunk_scores, dtype=np.float32),
                        episode_score_std=float(marked_summary["episode_score_std"]),
                        episode_score_q05=float(marked_summary["episode_score_q05"]),
                        episode_score_q95=float(marked_summary["episode_score_q95"]),
                        posterior_sample_count=int(marked_summary["posterior_sample_count"]),
                    )
                )
        finally:
            if use_separate_detector_policy:
                _release_policy(marked_detector_policy)
            _release_policy(marked_policy)

    positive_scores_np = np.asarray(positive_scores, dtype=np.float32)
    negative_scores_np = np.asarray(negative_scores, dtype=np.float32)
    wrong_key_scores_np = np.asarray(wrong_key_scores, dtype=np.float32)
    positive_wrong_key_pvalues_np = _empirical_survival_pvalues(positive_scores_np, wrong_key_scores_np)
    negative_wrong_key_pvalues_np = _empirical_survival_pvalues(negative_scores_np, wrong_key_scores_np)
    auc = online_eval._roc_auc(positive_scores_np, negative_scores_np)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else online_eval._calibrate_threshold_for_target_fpr(negative_scores_np, args.target_fpr)
    )
    tpr, fpr = online_eval._binary_metrics(positive_scores_np, negative_scores_np, threshold)
    mode_summary = _summarize_eval_mode(episode_records, eval_mode=args.eval_mode)

    print("LIBERO action inversion eval")
    print(f"rollout_config_name={args.config_name}")
    print(f"rollout_checkpoint_dir={args.checkpoint_dir}")
    print(f"detector_config_name={detector_config_name}")
    print(f"detector_checkpoint_dir={detector_checkpoint_dir}")
    print(f"eval_mode={args.eval_mode}")
    print(f"inversion_steps={args.num_inversion_steps}")
    print(f"inversion_method={args.inversion_method}")
    print(f"refinement_steps={args.refinement_steps}")
    print(f"refinement_learning_rate={args.refinement_learning_rate}")
    print(f"refinement_latent_l2={args.refinement_latent_l2}")
    print(f"refinement_init_l2={args.refinement_init_l2}")
    print(f"detector={args.detector}")
    print(f"null_decoy_count={args.null_decoy_count}")
    print(f"subspace_rank={args.subspace_rank if args.subspace_rank is not None else 'none'}")
    print(f"reference_mode={args.reference_mode}")
    print(f"chunk_selection_strategy={args.chunk_selection_strategy}")
    print(f"chunk_selection_period={args.chunk_selection_period}")
    print(f"chunk_selection_count={args.chunk_selection_count}")
    print(f"chunk_selection_total_slots={args.chunk_selection_total_slots if args.chunk_selection_total_slots is not None else 'none'}")
    print(f"max_score_windows={args.max_score_windows if args.max_score_windows is not None else 'all'}")
    print(f"window_aggregator={args.window_aggregator}")
    print(f"score_step_scope={effective_score_step_scope}")
    if effective_score_step_scope != args.score_step_scope:
        print(f"requested_score_step_scope={args.score_step_scope}")
    print(online_eval._describe_scores("watermarked_scores", positive_scores_np))
    print(online_eval._describe_scores("plain_scores", negative_scores_np))
    print(online_eval._describe_scores("wrong_key_scores", wrong_key_scores_np))
    print(online_eval._describe_scores("watermarked_wrong_key_pvalues", positive_wrong_key_pvalues_np))
    print(online_eval._describe_scores("plain_wrong_key_pvalues", negative_wrong_key_pvalues_np))
    print(f"roc_auc={auc:.4f}")
    if args.threshold is None:
        print(f"target_fpr={args.target_fpr:.4f}")
        print("threshold_source=calibrated_from_plain_scores")
    else:
        print("threshold_source=manual")
    print(f"threshold={threshold:.4f}")
    print(f"tpr={tpr:.4f}")
    print(f"fpr={fpr:.4f}")
    print(f"tpr_at_1pct_fpr={float(mode_summary['tpr_at_1pct_fpr']):.4f}")
    print(f"tpr_at_10pct_fpr={float(mode_summary['tpr_at_10pct_fpr']):.4f}")
    print(f"pairwise_wm_gt_plain_accuracy={float(mode_summary['pairwise_wm_gt_plain_accuracy']):.4f}")
    print(f"plain_task_offset_var={float(mode_summary['plain_task_offset_var']):.6f}")
    print(f"watermarked_task_offset_var={float(mode_summary['watermarked_task_offset_var']):.6f}")
    print(f"wm_minus_plain_mean={float(mode_summary['wm_minus_plain_mean']):.6f}")
    print(f"plain_recovery_rms_mean={np.mean(plain_recovery_rms):.6f}")
    print(f"watermarked_recovery_rms_mean={np.mean(marked_recovery_rms):.6f}")
    if args.save_rollout_dir is not None:
        print(f"saved_rollout_dir={args.save_rollout_dir / args.eval_mode}")
    if args.save_report_dir is not None:
        _write_report_artifacts(
            report_dir=args.save_report_dir,
            args=args,
            records=episode_records,
            summary=mode_summary,
        )
        print(f"saved_report_dir={args.save_report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
