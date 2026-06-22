#!/usr/bin/env python3
"""Offline rescoring for saved LIBERO old-reverse inversion rollout caches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import argparse
import csv
import dataclasses
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import eval_libero_action_inversion as _base  # noqa: E402


@dataclasses.dataclass(frozen=True)
class SavedInversionRolloutRecord:
    path: pathlib.Path
    task_suite_name: str
    task_id: int
    episode_idx: int
    episode_nonce: int
    variant: str
    eval_mode: str
    secret_key: int
    beta: float
    sample_rate_hz: float
    freq_range: tuple[float, float]
    n_tones: int
    detector: str
    reference_mode: str
    score_step_scope: str
    window_aggregator: str
    max_score_windows: int | None
    null_decoy_count: int
    subspace_rank: int | None
    chunk_selection_strategy: str
    chunk_selection_period: int
    chunk_selection_count: int
    chunk_selection_total_slots: int | None
    result: _base.online_eval.RolloutResult
    inversion_traces: tuple[_base.InversionChunkTrace, ...]


@dataclasses.dataclass(frozen=True)
class EpisodeScoreRow:
    task_id: int
    episode_idx: int
    variant: str
    candidate_key: int
    is_true_key: bool
    episode_score: float
    z_score: float
    inversion_step: int
    selected_window_count: int
    recovery_rms: float
    episode_score_std: float = 0.0
    episode_score_q05: float = 0.0
    episode_score_q95: float = 0.0
    posterior_sample_count: int = 0
    identification_score: float = 0.0
    identification_score_std: float = 0.0
    identification_score_q05: float = 0.0
    identification_score_q95: float = 0.0
    identification_rank: int = 0


@dataclasses.dataclass(frozen=True)
class CachedEpisodeScore:
    candidate_key: int
    episode_score: float
    episode_score_std: float = 0.0
    episode_score_q05: float = 0.0
    episode_score_q95: float = 0.0
    posterior_sample_count: int = 0


@dataclasses.dataclass(frozen=True)
class ReferenceVariantConfig:
    mode: str = "identity"
    lag_search_steps: int = 0
    rate_search_factors: tuple[float, ...] = (1.0,)
    smooth_alphas: tuple[float, ...] = ()
    temperature: float = 0.25


def _score_fn_for_detector(detector: str):
    if detector == "wmf":
        return _base._wmf_score_from_vectors
    if detector == "ace":
        return _base._ace_score_from_vectors
    raise ValueError(f"Unsupported detector for feature scoring: {detector!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--candidate-key", type=int, default=None)
    parser.add_argument("--false-key-count", type=int, default=31)
    parser.add_argument("--group-sizes", type=int, nargs="*", default=[1, 2, 4, 8, 12])
    parser.add_argument("--group-samples", type=int, default=256)
    parser.add_argument("--inversion-steps", type=int, nargs="*", default=[8])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reference-variant-mode",
        choices=("identity", "control_nuisance_max", "control_nuisance_logmeanexp"),
        default="identity",
        help="Aggregate each reference window over small control-channel nuisance variants.",
    )
    parser.add_argument("--reference-lag-search-steps", type=int, default=0)
    parser.add_argument("--reference-rate-search-factors", type=float, nargs="*", default=[1.0])
    parser.add_argument("--reference-smooth-alphas", type=float, nargs="*", default=[])
    parser.add_argument("--reference-variant-temperature", type=float, default=0.25)
    parser.add_argument(
        "--feature-calibration-mode",
        choices=("identity", "window_key_zscore"),
        default="identity",
        help="Calibrate per-window candidate features across keys before WMF/ACE scoring.",
    )
    parser.add_argument(
        "--global-lag-search-steps",
        type=int,
        default=0,
        help="Score fixed episode-level reference lags in [-N, N] and calibrate after lag maximization.",
    )
    parser.add_argument(
        "--spectral-feature-bands",
        nargs="*",
        default=["full"],
        help="Feature bands to concatenate, using full or lo:hi / lo,hi Hz entries.",
    )
    parser.add_argument(
        "--episode-spectral-feature-bands",
        nargs="*",
        default=[],
        help="Extra full-episode spectral feature bands to append, using lo:hi / lo,hi Hz entries.",
    )
    return parser.parse_args(argv)


def _reference_variant_config_from_args(args: argparse.Namespace) -> ReferenceVariantConfig:
    rate_factors = tuple(float(rate) for rate in (getattr(args, "reference_rate_search_factors", None) or (1.0,)))
    smooth_alphas = tuple(float(alpha) for alpha in (getattr(args, "reference_smooth_alphas", None) or ()))
    return ReferenceVariantConfig(
        mode=str(getattr(args, "reference_variant_mode", "identity")),
        lag_search_steps=int(getattr(args, "reference_lag_search_steps", 0)),
        rate_search_factors=rate_factors or (1.0,),
        smooth_alphas=smooth_alphas,
        temperature=float(getattr(args, "reference_variant_temperature", 0.25)),
    )


def _parse_spectral_feature_band(raw_band: str) -> tuple[float, float] | None:
    raw_band = str(raw_band).strip().lower()
    if raw_band == "full":
        return None
    separator = ":" if ":" in raw_band else ","
    pieces = raw_band.split(separator)
    if len(pieces) != 2:
        raise ValueError(f"spectral feature band must be full or lo:hi Hz, got {raw_band!r}.")
    low_hz = float(pieces[0])
    high_hz = float(pieces[1])
    if not (0.0 <= low_hz < high_hz):
        raise ValueError(f"spectral feature band must satisfy 0 <= low < high, got {raw_band!r}.")
    return (low_hz, high_hz)


def _spectral_feature_bands_from_args(args: argparse.Namespace) -> tuple[tuple[float, float] | None, ...]:
    raw_bands = getattr(args, "spectral_feature_bands", None) or ["full"]
    bands = tuple(_parse_spectral_feature_band(raw_band) for raw_band in raw_bands)
    return bands or (None,)


def _episode_spectral_feature_bands_from_args(args: argparse.Namespace) -> tuple[tuple[float, float], ...]:
    raw_bands = getattr(args, "episode_spectral_feature_bands", None) or []
    bands = tuple(_parse_spectral_feature_band(raw_band) for raw_band in raw_bands)
    if any(band is None for band in bands):
        raise ValueError("episode_spectral_feature_bands must be numeric lo:hi bands, not full.")
    return tuple(band for band in bands if band is not None)


def _validate_reference_variant_config(config: ReferenceVariantConfig) -> None:
    if config.mode not in {"identity", "control_nuisance_max", "control_nuisance_logmeanexp"}:
        raise ValueError(f"Unsupported reference_variant_mode={config.mode!r}")
    if int(config.lag_search_steps) < 0:
        raise ValueError("reference_lag_search_steps must be >= 0.")
    if not config.rate_search_factors or any(float(rate) <= 0.0 for rate in config.rate_search_factors):
        raise ValueError("reference_rate_search_factors must be positive.")
    if any(float(alpha) <= 0.0 or float(alpha) > 1.0 for alpha in config.smooth_alphas):
        raise ValueError("reference_smooth_alphas must be in (0, 1].")
    if float(config.temperature) <= 0.0:
        raise ValueError("reference_variant_temperature must be > 0.")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.rollout_dir.exists():
        raise FileNotFoundError(f"rollout_dir does not exist: {args.rollout_dir}")
    if not args.rollout_dir.is_dir():
        raise NotADirectoryError(f"rollout_dir is not a directory: {args.rollout_dir}")
    if args.false_key_count <= 0:
        raise ValueError("false_key_count must be > 0.")
    if args.group_samples <= 0:
        raise ValueError("group_samples must be > 0.")
    if not args.group_sizes or any(int(size) <= 0 for size in args.group_sizes):
        raise ValueError("group_sizes must be non-empty positive integers.")
    if not args.inversion_steps or any(int(step) <= 0 for step in args.inversion_steps):
        raise ValueError("inversion_steps must be non-empty positive integers.")
    if args.global_lag_search_steps < 0:
        raise ValueError("global_lag_search_steps must be >= 0.")
    _validate_reference_variant_config(_reference_variant_config_from_args(args))
    _spectral_feature_bands_from_args(args)
    _episode_spectral_feature_bands_from_args(args)


def _load_inversion_traces(payload: np.lib.npyio.NpzFile) -> tuple[_base.InversionChunkTrace, ...]:
    if "chunk_chunk_index" not in payload:
        return ()
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
    traces = []
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
            _base.InversionChunkTrace(
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
            )
        )
    return tuple(traces)


def _load_saved_rollout(path: pathlib.Path) -> SavedInversionRolloutRecord:
    payload = np.load(path)
    result = _base.online_eval.RolloutResult(
        telemetry=np.asarray(payload["telemetry"], dtype=np.float32),
        success=bool(payload["success"].item()),
        chunk_size=int(payload["chunk_size"].item()),
        task_description=str(payload["task_description"].item()),
        steps=int(payload["steps"].item()),
        execution_segments=tuple(
            _base.online_eval.ExecutionSegment(
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
    )
    max_score_windows_raw = int(payload["max_score_windows"].item()) if "max_score_windows" in payload else -1
    subspace_rank_raw = int(payload["subspace_rank"].item()) if "subspace_rank" in payload else -1
    chunk_selection_total_slots_raw = int(payload["chunk_selection_total_slots"].item()) if "chunk_selection_total_slots" in payload else -1
    null_decoy_count_raw = int(payload["null_decoy_count"].item()) if "null_decoy_count" in payload else 32
    return SavedInversionRolloutRecord(
        path=path,
        task_suite_name=str(payload["task_suite_name"].item()) if "task_suite_name" in payload else "unknown",
        task_id=int(payload["task_id"].item()),
        episode_idx=int(payload["episode_idx"].item()),
        episode_nonce=int(payload["episode_nonce"].item()),
        variant=str(payload["variant"].item()),
        eval_mode=str(payload["eval_mode"].item()) if "eval_mode" in payload else "task_rollout",
        secret_key=int(payload["secret_key"].item()) if "secret_key" in payload else 17,
        beta=float(payload["beta"].item()) if "beta" in payload else 0.0,
        sample_rate_hz=float(payload["sample_rate_hz"].item()) if "sample_rate_hz" in payload else 20.0,
        freq_range=(
            float(payload["freq_min_hz"].item()) if "freq_min_hz" in payload else 1.0,
            float(payload["freq_max_hz"].item()) if "freq_max_hz" in payload else 2.0,
        ),
        n_tones=int(payload["n_tones"].item()) if "n_tones" in payload else 4,
        detector=str(payload["detector"].item()) if "detector" in payload else "wmf",
        reference_mode=str(payload["reference_mode"].item()) if "reference_mode" in payload else "gaussian",
        score_step_scope=str(payload["score_step_scope"].item()) if "score_step_scope" in payload else "full_chunk",
        window_aggregator=str(payload["window_aggregator"].item()) if "window_aggregator" in payload else "sum",
        max_score_windows=None if max_score_windows_raw < 0 else max_score_windows_raw,
        null_decoy_count=null_decoy_count_raw,
        subspace_rank=None if subspace_rank_raw < 0 else subspace_rank_raw,
        chunk_selection_strategy=str(payload["chunk_selection_strategy"].item()) if "chunk_selection_strategy" in payload else "stateful_online",
        chunk_selection_period=int(payload["chunk_selection_period"].item()) if "chunk_selection_period" in payload else 1,
        chunk_selection_count=int(payload["chunk_selection_count"].item()) if "chunk_selection_count" in payload else 1,
        chunk_selection_total_slots=None if chunk_selection_total_slots_raw < 0 else chunk_selection_total_slots_raw,
        result=result,
        inversion_traces=_load_inversion_traces(payload),
    )


def _collect_rollout_pairs(
    rollout_dir: pathlib.Path,
) -> list[tuple[SavedInversionRolloutRecord, SavedInversionRolloutRecord]]:
    records = [_load_saved_rollout(path) for path in sorted(rollout_dir.glob("*.npz"))]
    if not records:
        raise FileNotFoundError(f"No .npz rollouts found in {rollout_dir}")
    grouped: dict[tuple[str, int, int, int], dict[str, SavedInversionRolloutRecord]] = {}
    for record in records:
        key = (record.task_suite_name, record.task_id, record.episode_idx, record.episode_nonce)
        grouped.setdefault(key, {})
        grouped[key][record.variant] = record
    pairs = []
    for key in sorted(grouped):
        variants = grouped[key]
        if "plain" not in variants or "watermarked" not in variants:
            raise ValueError(f"Missing plain/watermarked pair for key={key}: variants={sorted(variants)}")
        pairs.append((variants["plain"], variants["watermarked"]))
    return pairs


def _trace_for_inversion_step(trace: _base.InversionChunkTrace, *, step_count: int) -> _base.InversionChunkTrace:
    cached = trace.recovered_noise_by_step.get(int(step_count))
    if cached is None:
        return trace
    return dataclasses.replace(trace, recovered_noise=np.asarray(cached, dtype=np.float32))


def _traces_for_inversion_step(
    traces: Sequence[_base.InversionChunkTrace],
    *,
    step_count: int,
) -> list[_base.InversionChunkTrace]:
    return [_trace_for_inversion_step(trace, step_count=step_count) for trace in traces]


def _smooth_reference_trace(reference: np.ndarray, *, alpha: float) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape[0] == 0 or float(alpha) >= 1.0:
        return reference
    smoothed = np.empty_like(reference)
    smoothed[0] = reference[0]
    for step in range(1, reference.shape[0]):
        smoothed[step] = float(alpha) * reference[step] + (1.0 - float(alpha)) * smoothed[step - 1]
    return smoothed


def _reference_variant_rates(config: ReferenceVariantConfig) -> tuple[float, ...]:
    rates = [1.0]
    for rate in config.rate_search_factors:
        if not any(abs(float(rate) - existing) < 1e-8 for existing in rates):
            rates.append(float(rate))
    return tuple(rates)


def _iter_reference_variants(
    reference: np.ndarray,
    *,
    config: ReferenceVariantConfig,
) -> tuple[np.ndarray, ...]:
    reference = np.asarray(reference, dtype=np.float32)
    if config.mode == "identity":
        return (reference,)

    variants = []
    seen_params = set()
    lag_values = range(-int(config.lag_search_steps), int(config.lag_search_steps) + 1)
    smooth_options: tuple[float | None, ...] = (None, *tuple(float(alpha) for alpha in config.smooth_alphas))
    for rate in _reference_variant_rates(config):
        rate_key = round(float(rate), 8)
        rate_warped = (
            reference
            if abs(float(rate) - 1.0) < 1e-8
            else _base.online_eval.wm._resample_trace(reference, float(rate))
        )
        for alpha in smooth_options:
            alpha_key = None if alpha is None else round(float(alpha), 8)
            smoothed = rate_warped if alpha is None else _smooth_reference_trace(rate_warped, alpha=float(alpha))
            for lag in lag_values:
                params = (rate_key, int(lag), alpha_key)
                if params in seen_params:
                    continue
                seen_params.add(params)
                shifted = smoothed if int(lag) == 0 else _base.online_eval.wm._shift_trace(smoothed, int(lag))
                variants.append(np.asarray(shifted, dtype=np.float32))
    return tuple(variants) if variants else (reference,)


def _aggregate_reference_variant_scores(
    scores: Sequence[float],
    *,
    config: ReferenceVariantConfig,
) -> float:
    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.size == 0:
        return 0.0
    if config.mode == "identity":
        return float(score_array[0])
    if config.mode == "control_nuisance_max":
        return float(np.max(score_array))
    if config.mode == "control_nuisance_logmeanexp":
        temperature = float(config.temperature)
        max_score = float(np.max(score_array))
        return float(max_score + temperature * np.log(np.mean(np.exp((score_array - max_score) / temperature))))
    raise ValueError(f"Unsupported reference_variant_mode={config.mode!r}")


def _score_chunk_noise_similarity_for_band(
    recovered_noise: np.ndarray,
    reference: np.ndarray,
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    spectral_feature_band: tuple[float, float] | None,
) -> float:
    if spectral_feature_band is None:
        return _base._score_chunk_noise_similarity(
            recovered_noise,
            reference,
            detector=detector,
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
        )
    recovered_noise, reference = _base.online_eval._align_reference_to_action_signal(recovered_noise, reference)
    recovered_noise = _base.online_eval.wm._band_limit(
        recovered_noise,
        sample_rate_hz=sample_rate_hz,
        freq_range=spectral_feature_band,
    )
    reference = _base.online_eval.wm._band_limit(
        reference,
        sample_rate_hz=sample_rate_hz,
        freq_range=spectral_feature_band,
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


def _score_chunk_noise_similarity_with_reference_variants(
    recovered_noise: np.ndarray,
    reference: np.ndarray,
    *,
    detector: str,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    reference_variant_config: ReferenceVariantConfig | None = None,
    spectral_feature_band: tuple[float, float] | None = None,
) -> float:
    config = reference_variant_config or ReferenceVariantConfig()
    if config.mode == "identity":
        return _score_chunk_noise_similarity_for_band(
            recovered_noise,
            reference,
            detector=detector,
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
            spectral_feature_band=spectral_feature_band,
        )
    scores = [
        _score_chunk_noise_similarity_for_band(
            recovered_noise,
            variant_reference,
            detector=detector,
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
            spectral_feature_band=spectral_feature_band,
        )
        for variant_reference in _iter_reference_variants(reference, config=config)
    ]
    return _aggregate_reference_variant_scores(scores, config=config)


def _window_score_vector_with_reference_variants(
    chunk_traces: Sequence[_base.InversionChunkTrace],
    *,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    score_step_scope: str,
    max_windows: int | None,
    base_detector: str = "cosine",
    reference_variant_config: ReferenceVariantConfig | None = None,
    spectral_feature_bands: Sequence[tuple[float, float] | None] = (None,),
) -> np.ndarray:
    config = reference_variant_config or ReferenceVariantConfig()
    bands = tuple(spectral_feature_bands) or (None,)
    if config.mode == "identity" and bands == (None,):
        return _base._window_score_vector(
            list(chunk_traces),
            reference_mode=reference_mode,
            sample_rate_hz=sample_rate_hz,
            freq_range=freq_range,
            score_step_scope=score_step_scope,
            max_windows=max_windows,
            base_detector=base_detector,
        )
    scores = []
    for trace in _base._selected_score_traces(list(chunk_traces), max_windows=max_windows):
        score_steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        for band in bands:
            scores.append(
                _score_chunk_noise_similarity_with_reference_variants(
                    np.asarray(trace.recovered_noise[:score_steps], dtype=np.float32),
                    np.asarray(trace.reference[:score_steps], dtype=np.float32),
                    detector=base_detector,
                    reference_mode=reference_mode,
                    sample_rate_hz=sample_rate_hz,
                    freq_range=freq_range,
                    reference_variant_config=config,
                    spectral_feature_band=band,
                )
            )
    return np.asarray(scores, dtype=np.float32)


def _episode_spectral_score_vector(
    chunk_traces: Sequence[_base.InversionChunkTrace],
    *,
    reference_mode: str,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
    score_step_scope: str,
    max_windows: int | None,
    base_detector: str = "cosine",
    reference_variant_config: ReferenceVariantConfig | None = None,
    spectral_feature_bands: Sequence[tuple[float, float]] = (),
) -> np.ndarray:
    bands = tuple(spectral_feature_bands)
    if not bands:
        return np.zeros((0,), dtype=np.float32)
    recovered_pieces = []
    reference_pieces = []
    for trace in _base._selected_score_traces(list(chunk_traces), max_windows=max_windows):
        score_steps = trace.reference.shape[0] if score_step_scope == "full_chunk" else trace.executed_steps
        recovered_pieces.append(np.asarray(trace.recovered_noise[:score_steps], dtype=np.float32))
        reference_pieces.append(np.asarray(trace.reference[:score_steps], dtype=np.float32))
    if not recovered_pieces or not reference_pieces:
        return np.zeros((len(bands),), dtype=np.float32)
    recovered_noise = np.concatenate(recovered_pieces, axis=0)
    reference = np.concatenate(reference_pieces, axis=0)
    return np.asarray(
        [
            _score_chunk_noise_similarity_with_reference_variants(
                recovered_noise,
                reference,
                detector=base_detector,
                reference_mode=reference_mode,
                sample_rate_hz=sample_rate_hz,
                freq_range=freq_range,
                reference_variant_config=reference_variant_config,
                spectral_feature_band=band,
            )
            for band in bands
        ],
        dtype=np.float32,
    )


def _lag_trace_references(
    traces: Sequence[_base.InversionChunkTrace],
    *,
    lag: int,
) -> list[_base.InversionChunkTrace]:
    if int(lag) == 0:
        return list(traces)
    return [
        dataclasses.replace(
            trace,
            reference=_base.online_eval.wm._shift_trace(np.asarray(trace.reference, dtype=np.float32), int(lag)),
        )
        for trace in traces
    ]


def _reference_config_for_candidate(
    record: SavedInversionRolloutRecord,
    *,
    candidate_key: int,
) -> _base.online_eval.wm.InternalNoiseWatermarkConfig:
    action_dim = 0
    for trace in record.inversion_traces:
        if trace.reference.ndim == 2 and trace.reference.shape[1] > 0:
            action_dim = int(trace.reference.shape[1])
            break
    if action_dim == 0:
        raise ValueError(f"Saved rollout {record.path} is missing chunk references needed for rescoring.")
    return _base.online_eval.wm.InternalNoiseWatermarkConfig(
        secret_key=int(candidate_key),
        control_freq=record.sample_rate_hz,
        beta=float(record.beta),
        freq_range=record.freq_range,
        n_tones=record.n_tones,
        watermark_dims=tuple(range(action_dim)),
        reference_mode=record.reference_mode,
        chunk_selection_strategy=record.chunk_selection_strategy,
        chunk_selection_period=record.chunk_selection_period,
        chunk_selection_count=record.chunk_selection_count,
        chunk_selection_total_slots=record.chunk_selection_total_slots,
    )


def _score_record_for_candidate(
    record: SavedInversionRolloutRecord,
    *,
    candidate_key: int,
    step_count: int,
    false_key_count: int,
) -> EpisodeScoreRow:
    rows = _score_record_candidates(
        record,
        candidate_keys=[int(candidate_key)],
        step_count=step_count,
        false_key_count=false_key_count,
    )
    return rows[0]


def _required_candidate_score_keys(
    candidate_keys: Sequence[int],
    *,
    false_key_count: int,
) -> list[int]:
    required = set()
    for candidate_key in candidate_keys:
        candidate_key = int(candidate_key)
        required.add(candidate_key)
        required.update(range(candidate_key + 1, candidate_key + 1 + false_key_count))
    return sorted(required)


def _required_candidate_feature_keys(
    candidate_keys: Sequence[int],
    *,
    false_key_count: int,
) -> list[int]:
    feature_keys = set(_required_candidate_score_keys(candidate_keys, false_key_count=false_key_count))
    for score_key in list(feature_keys):
        feature_keys.update(range(int(score_key) + 1, int(score_key) + 1 + false_key_count))
    return sorted(feature_keys)


def _score_record_candidate_raw(
    record: SavedInversionRolloutRecord,
    *,
    traces: Sequence[_base.InversionChunkTrace],
    candidate_key: int,
    false_key_count: int,
) -> CachedEpisodeScore:
    reference_config = _reference_config_for_candidate(record, candidate_key=candidate_key)
    retargeted = _base._retarget_chunk_references(
        list(traces),
        reference_config=reference_config,
        sample_rate_hz=record.sample_rate_hz,
        episode_nonce=record.episode_nonce,
    )
    null_reference_configs = _base._wrong_key_reference_configs(reference_config, count=false_key_count)
    if _base._posterior_sample_count(list(retargeted), max_windows=record.max_score_windows) > 0:
        sample_scores, _ = _base._posterior_episode_score_samples(
            retargeted,
            detector=record.detector,
            reference_mode=record.reference_mode,
            sample_rate_hz=record.sample_rate_hz,
            freq_range=record.freq_range,
            aggregator=record.window_aggregator,
            score_step_scope=record.score_step_scope,
            max_windows=record.max_score_windows,
            reference_config=reference_config,
            episode_nonce=record.episode_nonce,
            null_decoy_count=false_key_count,
            subspace_rank=record.subspace_rank,
            null_reference_configs=null_reference_configs,
        )
        summary = _base._summarize_episode_score_samples(sample_scores)
        return CachedEpisodeScore(
            candidate_key=int(candidate_key),
            episode_score=float(summary["episode_score_mean"]),
            episode_score_std=float(summary["episode_score_std"]),
            episode_score_q05=float(summary["episode_score_q05"]),
            episode_score_q95=float(summary["episode_score_q95"]),
            posterior_sample_count=int(summary["posterior_sample_count"]),
        )
    episode_score = _base._episode_score(
        retargeted,
        detector=record.detector,
        reference_mode=record.reference_mode,
        sample_rate_hz=record.sample_rate_hz,
        freq_range=record.freq_range,
        aggregator=record.window_aggregator,
        score_step_scope=record.score_step_scope,
        max_windows=record.max_score_windows,
        reference_config=reference_config,
        episode_nonce=record.episode_nonce,
        null_decoy_count=false_key_count,
        subspace_rank=record.subspace_rank,
        null_reference_configs=null_reference_configs,
    )
    return CachedEpisodeScore(
        candidate_key=int(candidate_key),
        episode_score=float(episode_score),
    )


def _posterior_candidate_feature_vectors(
    record: SavedInversionRolloutRecord,
    *,
    traces: Sequence[_base.InversionChunkTrace],
    candidate_keys: Sequence[int],
    reference_variant_config: ReferenceVariantConfig | None = None,
    reference_lag: int = 0,
    spectral_feature_bands: Sequence[tuple[float, float] | None] = (None,),
    episode_spectral_feature_bands: Sequence[tuple[float, float]] = (),
) -> dict[int, np.ndarray]:
    sample_count = _base._posterior_sample_count(list(traces), max_windows=record.max_score_windows)
    if sample_count == 0:
        return {int(candidate_key): np.zeros((0, 0), dtype=np.float32) for candidate_key in candidate_keys}

    config = reference_variant_config or ReferenceVariantConfig()
    feature_vectors: dict[int, list[np.ndarray]] = {int(candidate_key): [] for candidate_key in candidate_keys}
    for sample_index in range(sample_count):
        sample_traces = _base._posterior_sample_traces(list(traces), sample_index=sample_index)
        for candidate_key in candidate_keys:
            reference_config = _reference_config_for_candidate(record, candidate_key=int(candidate_key))
            retargeted = _base._retarget_chunk_references(
                list(sample_traces),
                reference_config=reference_config,
                sample_rate_hz=record.sample_rate_hz,
                episode_nonce=record.episode_nonce,
            )
            retargeted = _lag_trace_references(retargeted, lag=int(reference_lag))
            window_vector = _window_score_vector_with_reference_variants(
                retargeted,
                reference_mode=record.reference_mode,
                sample_rate_hz=record.sample_rate_hz,
                freq_range=record.freq_range,
                score_step_scope=record.score_step_scope,
                max_windows=record.max_score_windows,
                base_detector="cosine",
                reference_variant_config=config,
                spectral_feature_bands=spectral_feature_bands,
            )
            episode_vector = _episode_spectral_score_vector(
                retargeted,
                reference_mode=record.reference_mode,
                sample_rate_hz=record.sample_rate_hz,
                freq_range=record.freq_range,
                score_step_scope=record.score_step_scope,
                max_windows=record.max_score_windows,
                base_detector="cosine",
                reference_variant_config=config,
                spectral_feature_bands=episode_spectral_feature_bands,
            )
            feature_vectors[int(candidate_key)].append(np.concatenate([window_vector, episode_vector], axis=0))

    return {
        int(candidate_key): np.stack(vectors, axis=0).astype(np.float32, copy=False)
        if vectors
        else np.zeros((0, 0), dtype=np.float32)
        for candidate_key, vectors in feature_vectors.items()
    }


def _candidate_feature_vectors(
    record: SavedInversionRolloutRecord,
    *,
    traces: Sequence[_base.InversionChunkTrace],
    candidate_keys: Sequence[int],
    reference_variant_config: ReferenceVariantConfig | None = None,
    reference_lag: int = 0,
    spectral_feature_bands: Sequence[tuple[float, float] | None] = (None,),
    episode_spectral_feature_bands: Sequence[tuple[float, float]] = (),
) -> dict[int, np.ndarray]:
    config = reference_variant_config or ReferenceVariantConfig()
    feature_vectors = {}
    for candidate_key in candidate_keys:
        reference_config = _reference_config_for_candidate(record, candidate_key=int(candidate_key))
        retargeted = _base._retarget_chunk_references(
            list(traces),
            reference_config=reference_config,
            sample_rate_hz=record.sample_rate_hz,
            episode_nonce=record.episode_nonce,
        )
        retargeted = _lag_trace_references(retargeted, lag=int(reference_lag))
        window_vector = _window_score_vector_with_reference_variants(
            retargeted,
            reference_mode=record.reference_mode,
            sample_rate_hz=record.sample_rate_hz,
            freq_range=record.freq_range,
            score_step_scope=record.score_step_scope,
            max_windows=record.max_score_windows,
            base_detector="cosine",
            reference_variant_config=config,
            spectral_feature_bands=spectral_feature_bands,
        )
        episode_vector = _episode_spectral_score_vector(
            retargeted,
            reference_mode=record.reference_mode,
            sample_rate_hz=record.sample_rate_hz,
            freq_range=record.freq_range,
            score_step_scope=record.score_step_scope,
            max_windows=record.max_score_windows,
            base_detector="cosine",
            reference_variant_config=config,
            spectral_feature_bands=episode_spectral_feature_bands,
        )
        feature_vectors[int(candidate_key)] = np.concatenate([window_vector, episode_vector], axis=0)
    return feature_vectors


def _calibrate_feature_vectors_across_keys(
    feature_vectors: dict[int, np.ndarray],
    *,
    mode: str,
) -> dict[int, np.ndarray]:
    if mode == "identity":
        return {int(key): np.asarray(value, dtype=np.float32) for key, value in feature_vectors.items()}
    if mode != "window_key_zscore":
        raise ValueError(f"Unsupported feature_calibration_mode={mode!r}")

    calibrated: dict[int, np.ndarray] = {}
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for key, value in feature_vectors.items():
        array = np.asarray(value, dtype=np.float32)
        groups[(array.ndim, tuple(array.shape))].append(int(key))

    for (_, _), keys in groups.items():
        keys = sorted(keys)
        values = np.stack([np.asarray(feature_vectors[key], dtype=np.float32) for key in keys], axis=0)
        if len(keys) <= 1:
            for key in keys:
                calibrated[key] = np.zeros_like(np.asarray(feature_vectors[key], dtype=np.float32))
            continue
        for index, key in enumerate(keys):
            other_values = np.concatenate([values[:index], values[index + 1 :]], axis=0)
            mean = np.mean(other_values, axis=0)
            std = np.std(other_values, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            calibrated[key] = ((values[index] - mean) / std).astype(np.float32, copy=False)
    return calibrated


def _fixed_null_candidate_scores(
    *,
    detector: str,
    candidate_key: int,
    false_key_count: int,
    feature_vectors: dict[int, np.ndarray],
    subspace_rank: int | None,
) -> tuple[float, np.ndarray]:
    """Score one candidate under a shared false-key null bank.

    For WMF/ACE we intentionally use the same false-key set for both:
    1. the null bank that defines the detector for candidate_key, and
    2. the false-score distribution used by the outer z-normalization.

    In the notation discussed in the analysis notes, this implements N_k = D_k
    rather than a strictly disjoint null/distribution split.
    """
    candidate_key = int(candidate_key)
    true_vector = np.asarray(feature_vectors[candidate_key], dtype=np.float64)
    if true_vector.size == 0:
        return 0.0, np.zeros((0,), dtype=np.float32)
    null_distribution_keys = [int(false_key) for false_key in range(candidate_key + 1, candidate_key + 1 + false_key_count)]
    null_vectors = [
        np.asarray(feature_vectors[false_key], dtype=np.float64)
        for false_key in null_distribution_keys
        if np.asarray(feature_vectors[false_key]).shape == true_vector.shape
    ]
    if not null_vectors:
        return 0.0, np.zeros((0,), dtype=np.float32)
    null_matrix = np.asarray(null_vectors, dtype=np.float64)
    if detector == "wmf":
        score_fn = _base._wmf_score_from_vectors
    elif detector == "ace":
        score_fn = _base._ace_score_from_vectors
    else:
        raise ValueError(f"Unsupported detector for fixed-null scoring: {detector!r}")
    candidate_score = float(score_fn(true_vector, null_matrix, subspace_rank=subspace_rank))
    false_scores = np.asarray(
        [
            float(
                score_fn(
                    np.asarray(feature_vectors[false_key], dtype=np.float64),
                    null_matrix,
                    subspace_rank=subspace_rank,
                )
            )
            for false_key in null_distribution_keys
            if np.asarray(feature_vectors[false_key]).shape == true_vector.shape
        ],
        dtype=np.float32,
    )
    return candidate_score, false_scores


def _global_lag_fixed_null_candidate_scores(
    *,
    detector: str,
    candidate_key: int,
    false_key_count: int,
    feature_vectors_by_lag: dict[int, dict[int, np.ndarray]],
    subspace_rank: int | None,
) -> tuple[float, np.ndarray]:
    score_fn = _score_fn_for_detector(detector)
    candidate_key = int(candidate_key)
    null_distribution_keys = [int(false_key) for false_key in range(candidate_key + 1, candidate_key + 1 + false_key_count)]
    candidate_scores = []
    false_scores_by_key: dict[int, list[float]] = {false_key: [] for false_key in null_distribution_keys}

    for lag in sorted(feature_vectors_by_lag):
        feature_vectors = feature_vectors_by_lag[int(lag)]
        if candidate_key not in feature_vectors:
            continue
        true_vector = np.asarray(feature_vectors[candidate_key], dtype=np.float64)
        if true_vector.size == 0:
            continue
        null_vectors = [
            np.asarray(feature_vectors[false_key], dtype=np.float64)
            for false_key in null_distribution_keys
            if false_key in feature_vectors and np.asarray(feature_vectors[false_key]).shape == true_vector.shape
        ]
        if not null_vectors:
            continue
        null_matrix = np.asarray(null_vectors, dtype=np.float64)
        candidate_scores.append(float(score_fn(true_vector, null_matrix, subspace_rank=subspace_rank)))
        for false_key in null_distribution_keys:
            if false_key not in feature_vectors:
                continue
            false_vector = np.asarray(feature_vectors[false_key], dtype=np.float64)
            if false_vector.shape != true_vector.shape:
                continue
            false_scores_by_key[false_key].append(float(score_fn(false_vector, null_matrix, subspace_rank=subspace_rank)))

    candidate_score = float(max(candidate_scores)) if candidate_scores else 0.0
    false_scores = np.asarray(
        [max(scores) for false_key, scores in false_scores_by_key.items() if scores],
        dtype=np.float32,
    )
    return candidate_score, false_scores


def _posterior_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    false_key_count: int,
    feature_vectors: dict[int, np.ndarray],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    if detector == "wmf":
        score_fn = _base._wmf_score_from_vectors
    elif detector == "ace":
        score_fn = _base._ace_score_from_vectors
    else:
        raise ValueError(f"Unsupported detector for posterior feature scoring: {detector!r}")

    candidate_key = int(candidate_key)
    sample_vectors = np.asarray(feature_vectors[candidate_key], dtype=np.float64)
    if sample_vectors.ndim < 2 or sample_vectors.shape[0] == 0:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)

    sample_scores = []
    null_distribution_keys = [int(false_key) for false_key in range(candidate_key + 1, candidate_key + 1 + false_key_count)]
    for sample_index in range(sample_vectors.shape[0]):
        true_vector = np.asarray(sample_vectors[sample_index], dtype=np.float64)
        if true_vector.size == 0:
            sample_scores.append(0.0)
            continue
        null_vectors = []
        for false_key in null_distribution_keys:
            false_samples = np.asarray(feature_vectors[false_key], dtype=np.float64)
            if false_samples.ndim < 2 or sample_index >= false_samples.shape[0]:
                continue
            false_vector = np.asarray(false_samples[sample_index], dtype=np.float64)
            if false_vector.shape == true_vector.shape:
                null_vectors.append(false_vector)
        if not null_vectors:
            sample_scores.append(0.0)
            continue
        null_matrix = np.asarray(null_vectors, dtype=np.float64)
        sample_scores.append(float(score_fn(true_vector, null_matrix, subspace_rank=subspace_rank)))

    summary = _base._summarize_episode_score_samples(np.asarray(sample_scores, dtype=np.float32))
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=float(summary["episode_score_mean"]),
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _posterior_global_lag_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    false_key_count: int,
    feature_vectors_by_lag: dict[int, dict[int, np.ndarray]],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    score_fn = _score_fn_for_detector(detector)
    candidate_key = int(candidate_key)
    null_distribution_keys = [int(false_key) for false_key in range(candidate_key + 1, candidate_key + 1 + false_key_count)]
    sample_count = 0
    for feature_vectors in feature_vectors_by_lag.values():
        candidate_samples = np.asarray(feature_vectors.get(candidate_key, np.zeros((0, 0))), dtype=np.float64)
        if candidate_samples.ndim >= 2:
            sample_count = max(sample_count, int(candidate_samples.shape[0]))
    if sample_count == 0:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)

    sample_scores = []
    for sample_index in range(sample_count):
        lag_scores = []
        for lag in sorted(feature_vectors_by_lag):
            feature_vectors = feature_vectors_by_lag[int(lag)]
            candidate_samples = np.asarray(feature_vectors.get(candidate_key, np.zeros((0, 0))), dtype=np.float64)
            if candidate_samples.ndim < 2 or sample_index >= candidate_samples.shape[0]:
                continue
            true_vector = np.asarray(candidate_samples[sample_index], dtype=np.float64)
            if true_vector.size == 0:
                continue
            null_vectors = []
            for false_key in null_distribution_keys:
                false_samples = np.asarray(feature_vectors.get(false_key, np.zeros((0, 0))), dtype=np.float64)
                if false_samples.ndim < 2 or sample_index >= false_samples.shape[0]:
                    continue
                false_vector = np.asarray(false_samples[sample_index], dtype=np.float64)
                if false_vector.shape == true_vector.shape:
                    null_vectors.append(false_vector)
            if not null_vectors:
                continue
            null_matrix = np.asarray(null_vectors, dtype=np.float64)
            lag_scores.append(float(score_fn(true_vector, null_matrix, subspace_rank=subspace_rank)))
        sample_scores.append(float(max(lag_scores)) if lag_scores else 0.0)

    summary = _base._summarize_episode_score_samples(np.asarray(sample_scores, dtype=np.float32))
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=float(summary["episode_score_mean"]),
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _identification_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    candidate_keys: Sequence[int],
    feature_vectors: dict[int, np.ndarray],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    score_fn = _score_fn_for_detector(detector)
    candidate_key = int(candidate_key)
    true_vector = np.asarray(feature_vectors[candidate_key], dtype=np.float64)
    if true_vector.size == 0:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)
    peer_vectors = [
        np.asarray(feature_vectors[int(peer_key)], dtype=np.float64)
        for peer_key in candidate_keys
        if int(peer_key) != candidate_key and np.asarray(feature_vectors[int(peer_key)]).shape == true_vector.shape
    ]
    if not peer_vectors:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)
    episode_score = float(score_fn(true_vector, np.asarray(peer_vectors, dtype=np.float64), subspace_rank=subspace_rank))
    summary = _base._summarize_episode_score_samples(
        np.zeros((0,), dtype=np.float32),
        fallback_score=episode_score,
    )
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=episode_score,
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _global_lag_identification_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    candidate_keys: Sequence[int],
    feature_vectors_by_lag: dict[int, dict[int, np.ndarray]],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    candidate_key = int(candidate_key)
    scores = [
        _identification_cached_score_from_feature_vectors(
            detector=detector,
            candidate_key=candidate_key,
            candidate_keys=candidate_keys,
            feature_vectors=feature_vectors,
            subspace_rank=subspace_rank,
        ).episode_score
        for _, feature_vectors in sorted(feature_vectors_by_lag.items())
        if candidate_key in feature_vectors
    ]
    episode_score = float(max(scores)) if scores else 0.0
    summary = _base._summarize_episode_score_samples(
        np.zeros((0,), dtype=np.float32),
        fallback_score=episode_score,
    )
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=episode_score,
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _posterior_global_lag_identification_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    candidate_keys: Sequence[int],
    feature_vectors_by_lag: dict[int, dict[int, np.ndarray]],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    score_fn = _score_fn_for_detector(detector)
    candidate_key = int(candidate_key)
    sample_count = 0
    for feature_vectors in feature_vectors_by_lag.values():
        candidate_samples = np.asarray(feature_vectors.get(candidate_key, np.zeros((0, 0))), dtype=np.float64)
        if candidate_samples.ndim >= 2:
            sample_count = max(sample_count, int(candidate_samples.shape[0]))
    if sample_count == 0:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)

    sample_scores = []
    for sample_index in range(sample_count):
        lag_scores = []
        for lag in sorted(feature_vectors_by_lag):
            feature_vectors = feature_vectors_by_lag[int(lag)]
            candidate_samples = np.asarray(feature_vectors.get(candidate_key, np.zeros((0, 0))), dtype=np.float64)
            if candidate_samples.ndim < 2 or sample_index >= candidate_samples.shape[0]:
                continue
            true_vector = np.asarray(candidate_samples[sample_index], dtype=np.float64)
            if true_vector.size == 0:
                continue
            peer_vectors = []
            for peer_key in candidate_keys:
                peer_key = int(peer_key)
                if peer_key == candidate_key:
                    continue
                peer_samples = np.asarray(feature_vectors.get(peer_key, np.zeros((0, 0))), dtype=np.float64)
                if peer_samples.ndim < 2 or sample_index >= peer_samples.shape[0]:
                    continue
                peer_vector = np.asarray(peer_samples[sample_index], dtype=np.float64)
                if peer_vector.shape == true_vector.shape:
                    peer_vectors.append(peer_vector)
            if not peer_vectors:
                continue
            lag_scores.append(float(score_fn(true_vector, np.asarray(peer_vectors, dtype=np.float64), subspace_rank=subspace_rank)))
        sample_scores.append(float(max(lag_scores)) if lag_scores else 0.0)

    summary = _base._summarize_episode_score_samples(np.asarray(sample_scores, dtype=np.float32))
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=float(summary["episode_score_mean"]),
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _posterior_identification_cached_score_from_feature_vectors(
    *,
    detector: str,
    candidate_key: int,
    candidate_keys: Sequence[int],
    feature_vectors: dict[int, np.ndarray],
    subspace_rank: int | None,
) -> CachedEpisodeScore:
    score_fn = _score_fn_for_detector(detector)
    candidate_key = int(candidate_key)
    sample_vectors = np.asarray(feature_vectors[candidate_key], dtype=np.float64)
    if sample_vectors.ndim < 2 or sample_vectors.shape[0] == 0:
        return CachedEpisodeScore(candidate_key=candidate_key, episode_score=0.0)

    sample_scores = []
    for sample_index in range(sample_vectors.shape[0]):
        true_vector = np.asarray(sample_vectors[sample_index], dtype=np.float64)
        if true_vector.size == 0:
            sample_scores.append(0.0)
            continue
        peer_vectors = []
        for peer_key in candidate_keys:
            peer_key = int(peer_key)
            if peer_key == candidate_key:
                continue
            peer_samples = np.asarray(feature_vectors[peer_key], dtype=np.float64)
            if peer_samples.ndim < 2 or sample_index >= peer_samples.shape[0]:
                continue
            peer_vector = np.asarray(peer_samples[sample_index], dtype=np.float64)
            if peer_vector.shape == true_vector.shape:
                peer_vectors.append(peer_vector)
        if not peer_vectors:
            sample_scores.append(0.0)
            continue
        sample_scores.append(
            float(score_fn(true_vector, np.asarray(peer_vectors, dtype=np.float64), subspace_rank=subspace_rank))
        )

    summary = _base._summarize_episode_score_samples(np.asarray(sample_scores, dtype=np.float32))
    return CachedEpisodeScore(
        candidate_key=candidate_key,
        episode_score=float(summary["episode_score_mean"]),
        episode_score_std=float(summary["episode_score_std"]),
        episode_score_q05=float(summary["episode_score_q05"]),
        episode_score_q95=float(summary["episode_score_q95"]),
        posterior_sample_count=int(summary["posterior_sample_count"]),
    )


def _apply_identification_ranks(rows: Sequence[EpisodeScoreRow]) -> list[EpisodeScoreRow]:
    ranked = sorted(rows, key=lambda row: (float(row.identification_score), -int(row.candidate_key)), reverse=True)
    rank_by_key = {int(row.candidate_key): index + 1 for index, row in enumerate(ranked)}
    return [
        dataclasses.replace(row, identification_rank=int(rank_by_key[int(row.candidate_key)]))
        for row in rows
    ]


def _score_record_candidates(
    record: SavedInversionRolloutRecord,
    *,
    candidate_keys: Sequence[int],
    step_count: int,
    false_key_count: int,
    reference_variant_config: ReferenceVariantConfig | None = None,
    feature_calibration_mode: str = "identity",
    global_lag_search_steps: int = 0,
    spectral_feature_bands: Sequence[tuple[float, float] | None] = (None,),
    episode_spectral_feature_bands: Sequence[tuple[float, float]] = (),
) -> list[EpisodeScoreRow]:
    candidate_keys = [int(candidate_key) for candidate_key in candidate_keys]
    traces = _traces_for_inversion_step(record.inversion_traces, step_count=step_count)
    selected_window_count = len(_base._selected_score_traces(list(traces), max_windows=record.max_score_windows))
    recovery_rms = _base._episode_recovery_rms(list(traces), max_windows=record.max_score_windows)
    posterior_sample_count = _base._posterior_sample_count(list(traces), max_windows=record.max_score_windows)
    config = reference_variant_config or ReferenceVariantConfig()
    feature_kwargs = {"reference_variant_config": config} if config.mode != "identity" else {}
    bands = tuple(spectral_feature_bands) or (None,)
    feature_options = dict(feature_kwargs)
    if bands != (None,):
        feature_options["spectral_feature_bands"] = bands
    episode_bands = tuple(episode_spectral_feature_bands)
    if episode_bands:
        feature_options["episode_spectral_feature_bands"] = episode_bands
    if record.detector in {"wmf", "ace"} and posterior_sample_count == 0:
        required_score_keys = _required_candidate_score_keys(candidate_keys, false_key_count=false_key_count)
        if int(global_lag_search_steps) > 0:
            feature_vectors_by_lag = {
                int(lag): _calibrate_feature_vectors_across_keys(
                    _candidate_feature_vectors(
                        record,
                            traces=traces,
                            candidate_keys=required_score_keys,
                            reference_lag=int(lag),
                            **feature_options,
                        ),
                    mode=feature_calibration_mode,
                )
                for lag in range(-int(global_lag_search_steps), int(global_lag_search_steps) + 1)
            }
            identification_score_cache = {
                candidate_key: _global_lag_identification_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    candidate_keys=candidate_keys,
                    feature_vectors_by_lag=feature_vectors_by_lag,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in candidate_keys
            }
        else:
            feature_vectors = _calibrate_feature_vectors_across_keys(
                _candidate_feature_vectors(
                    record,
                    traces=traces,
                    candidate_keys=required_score_keys,
                    **feature_options,
                ),
                mode=feature_calibration_mode,
            )
            feature_vectors_by_lag = {}
            identification_score_cache = {
                candidate_key: _identification_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    candidate_keys=candidate_keys,
                    feature_vectors=feature_vectors,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in candidate_keys
            }
        rows = []
        for candidate_key in candidate_keys:
            if int(global_lag_search_steps) > 0:
                candidate_score, false_scores_np = _global_lag_fixed_null_candidate_scores(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    false_key_count=false_key_count,
                    feature_vectors_by_lag=feature_vectors_by_lag,
                    subspace_rank=record.subspace_rank,
                )
            else:
                candidate_score, false_scores_np = _fixed_null_candidate_scores(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    false_key_count=false_key_count,
                    feature_vectors=feature_vectors,
                    subspace_rank=record.subspace_rank,
                )
            false_std = float(np.std(false_scores_np))
            if false_std < 1e-6:
                false_std = 1.0
            z_score = float((float(candidate_score) - float(np.mean(false_scores_np))) / false_std) if false_scores_np.size else 0.0
            summary = _base._summarize_episode_score_samples(
                np.zeros((0,), dtype=np.float32),
                fallback_score=float(candidate_score),
            )
            identification_score = identification_score_cache[candidate_key]
            rows.append(
                EpisodeScoreRow(
                    task_id=record.task_id,
                    episode_idx=record.episode_idx,
                    variant=record.variant,
                    candidate_key=int(candidate_key),
                    is_true_key=int(candidate_key) == int(record.secret_key),
                    episode_score=float(candidate_score),
                    z_score=float(z_score),
                    inversion_step=int(step_count),
                    selected_window_count=int(selected_window_count),
                    recovery_rms=float(recovery_rms),
                    episode_score_std=float(summary["episode_score_std"]),
                    episode_score_q05=float(summary["episode_score_q05"]),
                    episode_score_q95=float(summary["episode_score_q95"]),
                    posterior_sample_count=int(summary["posterior_sample_count"]),
                    identification_score=float(identification_score.episode_score),
                    identification_score_std=float(identification_score.episode_score_std),
                    identification_score_q05=float(identification_score.episode_score_q05),
                    identification_score_q95=float(identification_score.episode_score_q95),
                )
            )
        return _apply_identification_ranks(rows)
    if record.detector in {"wmf", "ace"} and posterior_sample_count > 0:
        if int(global_lag_search_steps) > 0:
            posterior_required_keys = _required_candidate_feature_keys(candidate_keys, false_key_count=false_key_count)
            feature_vectors_by_lag = {
                int(lag): _calibrate_feature_vectors_across_keys(
                    _posterior_candidate_feature_vectors(
                        record,
                        traces=traces,
                        candidate_keys=posterior_required_keys,
                        reference_lag=int(lag),
                        **feature_options,
                    ),
                    mode=feature_calibration_mode,
                )
                for lag in range(-int(global_lag_search_steps), int(global_lag_search_steps) + 1)
            }
            raw_score_cache = {
                candidate_key: _posterior_global_lag_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    false_key_count=false_key_count,
                    feature_vectors_by_lag=feature_vectors_by_lag,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in _required_candidate_score_keys(candidate_keys, false_key_count=false_key_count)
            }
            identification_score_cache = {
                candidate_key: _posterior_global_lag_identification_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    candidate_keys=candidate_keys,
                    feature_vectors_by_lag=feature_vectors_by_lag,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in candidate_keys
            }
        else:
            feature_vectors = _calibrate_feature_vectors_across_keys(
                _posterior_candidate_feature_vectors(
                    record,
                    traces=traces,
                    candidate_keys=_required_candidate_feature_keys(candidate_keys, false_key_count=false_key_count),
                    **feature_options,
                ),
                mode=feature_calibration_mode,
            )
            raw_score_cache = {
                candidate_key: _posterior_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    false_key_count=false_key_count,
                    feature_vectors=feature_vectors,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in _required_candidate_score_keys(candidate_keys, false_key_count=false_key_count)
            }
            identification_score_cache = {
                candidate_key: _posterior_identification_cached_score_from_feature_vectors(
                    detector=record.detector,
                    candidate_key=int(candidate_key),
                    candidate_keys=candidate_keys,
                    feature_vectors=feature_vectors,
                    subspace_rank=record.subspace_rank,
                )
                for candidate_key in candidate_keys
            }
    else:
        raw_score_cache = {
            score.candidate_key: score
            for score in (
                _score_record_candidate_raw(
                    record,
                    traces=traces,
                    candidate_key=score_key,
                    false_key_count=false_key_count,
                )
                for score_key in _required_candidate_score_keys(candidate_keys, false_key_count=false_key_count)
            )
        }
        identification_score_cache = {
            candidate_key: raw_score_cache[candidate_key]
            for candidate_key in candidate_keys
        }
    rows = []
    for candidate_key in candidate_keys:
        candidate_key = int(candidate_key)
        false_scores_np = np.asarray(
            [
                raw_score_cache[int(false_key)].episode_score
                for false_key in range(candidate_key + 1, candidate_key + 1 + false_key_count)
            ],
            dtype=np.float32,
        )
        false_std = float(np.std(false_scores_np))
        if false_std < 1e-6:
            false_std = 1.0
        cached_score = raw_score_cache[candidate_key]
        identification_score = identification_score_cache[candidate_key]
        z_score = float((float(cached_score.episode_score) - float(np.mean(false_scores_np))) / false_std)
        rows.append(
            EpisodeScoreRow(
                task_id=record.task_id,
                episode_idx=record.episode_idx,
                variant=record.variant,
                candidate_key=candidate_key,
                is_true_key=candidate_key == int(record.secret_key),
                episode_score=float(cached_score.episode_score),
                z_score=float(z_score),
                inversion_step=int(step_count),
                selected_window_count=int(selected_window_count),
                recovery_rms=float(recovery_rms),
                episode_score_std=float(cached_score.episode_score_std),
                episode_score_q05=float(cached_score.episode_score_q05),
                episode_score_q95=float(cached_score.episode_score_q95),
                posterior_sample_count=int(cached_score.posterior_sample_count),
                identification_score=float(identification_score.episode_score),
                identification_score_std=float(identification_score.episode_score_std),
                identification_score_q05=float(identification_score.episode_score_q05),
                identification_score_q95=float(identification_score.episode_score_q95),
            )
        )
    return _apply_identification_ranks(rows)


def _presence_auc(rows: Sequence[EpisodeScoreRow], *, use_z_score: bool = True) -> float:
    positive = np.asarray(
        [row.z_score if use_z_score else row.episode_score for row in rows if row.variant == "watermarked" and row.is_true_key],
        dtype=np.float32,
    )
    negative = np.asarray(
        [row.z_score if use_z_score else row.episode_score for row in rows if row.variant == "plain" and row.is_true_key],
        dtype=np.float32,
    )
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float(_base.online_eval._roc_auc(positive, negative))


def _attribution_top1_accuracy(rows: Sequence[dict[str, object]]) -> float:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["episode_idx"]), str(row["variant"]))].append(row)
    wins = 0
    total = 0
    for key, group in grouped.items():
        if key[2] != "watermarked":
            continue
        total += 1
        best_row = max(group, key=lambda item: float(item["episode_score"]))
        if bool(best_row["is_true_key"]):
            wins += 1
    if total == 0:
        return float("nan")
    return float(wins / total)


def _identification_true_key_rows(rows: Sequence[EpisodeScoreRow]) -> list[EpisodeScoreRow]:
    return [row for row in rows if row.variant == "watermarked" and row.is_true_key]


def _identification_topk_accuracy(rows: Sequence[EpisodeScoreRow], *, top_k: int) -> float:
    if top_k <= 0:
        return float("nan")
    true_rows = _identification_true_key_rows(rows)
    if not true_rows:
        return float("nan")
    wins = sum(int(int(row.identification_rank) <= int(top_k)) for row in true_rows)
    return float(wins / len(true_rows))


def _identification_mean_rank(rows: Sequence[EpisodeScoreRow]) -> float:
    true_rows = _identification_true_key_rows(rows)
    if not true_rows:
        return float("nan")
    return float(np.mean([int(row.identification_rank) for row in true_rows], dtype=np.float32))


def _identification_mean_reciprocal_rank(rows: Sequence[EpisodeScoreRow]) -> float:
    true_rows = _identification_true_key_rows(rows)
    if not true_rows:
        return float("nan")
    return float(np.mean([1.0 / float(row.identification_rank) for row in true_rows], dtype=np.float32))


def _max_same_task_group_size(rows: Sequence[EpisodeScoreRow], *, variant: str) -> int:
    counts: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        if row.variant != variant:
            continue
        counts[int(row.task_id)].add((int(row.task_id), int(row.episode_idx)))
    if not counts:
        return 0
    return max(len(values) for values in counts.values())


def _sample_group_rows(
    rows: Sequence[EpisodeScoreRow],
    *,
    grouping_mode: str,
    group_size: int,
    group_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if group_size <= 0:
        return []
    grouped_by_task: dict[int, list[EpisodeScoreRow]] = defaultdict(list)
    for row in rows:
        grouped_by_task[int(row.task_id)].append(row)
    rng = np.random.default_rng(seed)
    sampled_rows: list[dict[str, object]] = []
    if grouping_mode == "cross_task":
        if len(rows) < group_size:
            return []
        for group_index in range(group_samples):
            selection = rng.choice(len(rows), size=group_size, replace=False)
            group = [rows[int(idx)] for idx in selection]
            sampled_rows.append(
                {
                    "grouping_mode": grouping_mode,
                    "group_size": group_size,
                    "group_index": group_index,
                    "variant": group[0].variant,
                    "group_score": float(sum(float(item.z_score) for item in group)),
                    "task_id": "mixed",
                }
            )
        return sampled_rows
    if grouping_mode == "same_task":
        eligible_tasks = [task_id for task_id, values in grouped_by_task.items() if len(values) >= group_size]
        if not eligible_tasks:
            return []
        for group_index in range(group_samples):
            task_id = int(eligible_tasks[group_index % len(eligible_tasks)])
            candidates = grouped_by_task[task_id]
            selection = rng.choice(len(candidates), size=group_size, replace=False)
            group = [candidates[int(idx)] for idx in selection]
            sampled_rows.append(
                {
                    "grouping_mode": grouping_mode,
                    "group_size": group_size,
                    "group_index": group_index,
                    "variant": group[0].variant,
                    "group_score": float(sum(float(item.z_score) for item in group)),
                    "task_id": task_id,
                }
            )
        return sampled_rows
    raise ValueError(f"Unsupported grouping_mode: {grouping_mode}")


def _identification_group_row(
    episode_groups: Sequence[Sequence[EpisodeScoreRow]],
    *,
    grouping_mode: str,
    group_size: int,
    group_index: int,
    task_id: int | str,
) -> dict[str, object]:
    candidate_scores: dict[int, float] = defaultdict(float)
    true_keys = set()
    for episode_rows in episode_groups:
        episode_true_keys = {int(row.candidate_key) for row in episode_rows if row.is_true_key}
        if len(episode_true_keys) != 1:
            raise ValueError("Each episode group must contain exactly one true key row.")
        true_keys.update(episode_true_keys)
        for row in episode_rows:
            candidate_scores[int(row.candidate_key)] += float(row.identification_score)
    if len(true_keys) != 1:
        raise ValueError("Identification groups require a consistent true key across episodes.")
    true_key = int(next(iter(true_keys)))
    ranked = sorted(candidate_scores.items(), key=lambda item: (float(item[1]), -int(item[0])), reverse=True)
    predicted_key = int(ranked[0][0]) if ranked else -1
    rank_by_key = {int(candidate_key): index + 1 for index, (candidate_key, _) in enumerate(ranked)}
    true_rank = int(rank_by_key.get(true_key, 0))
    true_score = float(candidate_scores.get(true_key, 0.0))
    best_wrong_score = float(max((score for candidate_key, score in ranked if int(candidate_key) != true_key), default=0.0))
    return {
        "grouping_mode": grouping_mode,
        "group_size": int(group_size),
        "group_index": int(group_index),
        "variant": "watermarked",
        "task_id": task_id,
        "true_key": int(true_key),
        "predicted_key": int(predicted_key),
        "true_rank": int(true_rank),
        "top1_correct": int(true_rank == 1),
        "top3_correct": int(true_rank > 0 and true_rank <= 3),
        "reciprocal_rank": float(1.0 / true_rank) if true_rank > 0 else 0.0,
        "true_score": float(true_score),
        "best_wrong_score": float(best_wrong_score),
        "true_minus_best_wrong_score": float(true_score - best_wrong_score),
    }


def _sample_identification_group_rows(
    rows: Sequence[EpisodeScoreRow],
    *,
    grouping_mode: str,
    group_size: int,
    group_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if group_size <= 0:
        return []
    watermarked_rows = [row for row in rows if row.variant == "watermarked"]
    episodes: dict[tuple[int, int], list[EpisodeScoreRow]] = defaultdict(list)
    for row in watermarked_rows:
        episodes[(int(row.task_id), int(row.episode_idx))].append(row)
    if not episodes:
        return []
    grouped_by_task: dict[int, list[list[EpisodeScoreRow]]] = defaultdict(list)
    all_episode_groups: list[list[EpisodeScoreRow]] = []
    for (task_id, _episode_idx), episode_rows in sorted(episodes.items()):
        grouped_by_task[int(task_id)].append(list(episode_rows))
        all_episode_groups.append(list(episode_rows))
    rng = np.random.default_rng(seed)
    sampled_rows: list[dict[str, object]] = []
    if grouping_mode == "cross_task":
        if len(all_episode_groups) < group_size:
            return []
        for group_index in range(group_samples):
            selection = rng.choice(len(all_episode_groups), size=group_size, replace=False)
            sampled_rows.append(
                _identification_group_row(
                    [all_episode_groups[int(idx)] for idx in selection],
                    grouping_mode=grouping_mode,
                    group_size=group_size,
                    group_index=group_index,
                    task_id="mixed",
                )
            )
        return sampled_rows
    if grouping_mode == "same_task":
        eligible_tasks = [task_id for task_id, values in grouped_by_task.items() if len(values) >= group_size]
        if not eligible_tasks:
            return []
        for group_index in range(group_samples):
            task_id = int(eligible_tasks[group_index % len(eligible_tasks)])
            candidates = grouped_by_task[task_id]
            selection = rng.choice(len(candidates), size=group_size, replace=False)
            sampled_rows.append(
                _identification_group_row(
                    [candidates[int(idx)] for idx in selection],
                    grouping_mode=grouping_mode,
                    group_size=group_size,
                    group_index=group_index,
                    task_id=task_id,
                )
            )
        return sampled_rows
    raise ValueError(f"Unsupported grouping_mode: {grouping_mode}")


def _group_auc(group_rows: Sequence[dict[str, object]]) -> float:
    positive = np.asarray([float(row["group_score"]) for row in group_rows if row["variant"] == "watermarked"], dtype=np.float32)
    negative = np.asarray([float(row["group_score"]) for row in group_rows if row["variant"] == "plain"], dtype=np.float32)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float(_base.online_eval._roc_auc(positive, negative))


def _identification_group_metrics_by_size(group_rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in group_rows:
        grouped[int(row["group_size"])].append(row)
    summary: dict[str, dict[str, float | int]] = {}
    for group_size in sorted(grouped):
        rows_for_size = grouped[group_size]
        summary[str(group_size)] = {
            "group_count": len(rows_for_size),
            "top1_accuracy": float(np.mean([float(row["top1_correct"]) for row in rows_for_size], dtype=np.float32)),
            "top3_accuracy": float(np.mean([float(row["top3_correct"]) for row in rows_for_size], dtype=np.float32)),
            "mean_reciprocal_rank": float(np.mean([float(row["reciprocal_rank"]) for row in rows_for_size], dtype=np.float32)),
            "true_key_mean_rank": float(np.mean([float(row["true_rank"]) for row in rows_for_size], dtype=np.float32)),
        }
    return summary


def _record_to_metadata_row(record: SavedInversionRolloutRecord) -> dict[str, object]:
    return {
        "path": str(record.path),
        "task_suite_name": record.task_suite_name,
        "task_id": record.task_id,
        "episode_idx": record.episode_idx,
        "episode_nonce": record.episode_nonce,
        "variant": record.variant,
        "eval_mode": record.eval_mode,
        "secret_key": record.secret_key,
        "beta": record.beta,
        "success": bool(record.result.success),
        "steps": record.result.steps,
        "chunk_selection_count": record.chunk_selection_count,
    }


def _episode_row_to_dict(row: EpisodeScoreRow) -> dict[str, object]:
    return {
        "task_id": row.task_id,
        "episode_idx": row.episode_idx,
        "variant": row.variant,
        "candidate_key": row.candidate_key,
        "is_true_key": row.is_true_key,
        "episode_score": row.episode_score,
        "episode_score_std": row.episode_score_std,
        "episode_score_q05": row.episode_score_q05,
        "episode_score_q95": row.episode_score_q95,
        "posterior_sample_count": row.posterior_sample_count,
        "z_score": row.z_score,
        "identification_score": row.identification_score,
        "identification_score_std": row.identification_score_std,
        "identification_score_q05": row.identification_score_q05,
        "identification_score_q95": row.identification_score_q95,
        "identification_rank": row.identification_rank,
        "inversion_step": row.inversion_step,
        "selected_window_count": row.selected_window_count,
        "recovery_rms": row.recovery_rms,
    }


def _write_csv(path: pathlib.Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _spectral_feature_band_to_json(band: tuple[float, float] | None) -> str | list[float]:
    if band is None:
        return "full"
    return [float(band[0]), float(band[1])]


def _summary_markdown(
    *,
    baseline_presence_auc: float,
    identification_top1_accuracy: float,
    identification_top3_accuracy: float,
    identification_mean_reciprocal_rank: float,
    identification_true_key_mean_rank: float,
    same_task_identification_metrics: dict[str, dict[str, float | int]],
    cross_task_identification_metrics: dict[str, dict[str, float | int]],
    same_task_rows: Sequence[dict[str, object]],
    cross_task_rows: Sequence[dict[str, object]],
    inversion_step_rows: Sequence[dict[str, object]],
    max_same_task_group_size: int,
    rollout_axes: dict[str, object],
    offline_axes: dict[str, object],
) -> str:
    lines = [
        "# Old Reverse Multi-Trajectory Summary",
        "",
        "## Baseline",
        "",
        f"- single-trajectory presence AUC: `{baseline_presence_auc:.4f}`" if np.isfinite(baseline_presence_auc) else "- single-trajectory presence AUC: `nan`",
        f"- maximum feasible same-task group size from cached data: `{max_same_task_group_size}`",
        "",
        "## Identification",
        "",
        f"- top-1 accuracy: `{identification_top1_accuracy:.4f}`" if np.isfinite(identification_top1_accuracy) else "- top-1 accuracy: `nan`",
        f"- top-3 accuracy: `{identification_top3_accuracy:.4f}`" if np.isfinite(identification_top3_accuracy) else "- top-3 accuracy: `nan`",
        f"- mean reciprocal rank: `{identification_mean_reciprocal_rank:.4f}`" if np.isfinite(identification_mean_reciprocal_rank) else "- mean reciprocal rank: `nan`",
        f"- true-key mean rank: `{identification_true_key_mean_rank:.4f}`" if np.isfinite(identification_true_key_mean_rank) else "- true-key mean rank: `nan`",
        "",
        "## Group Identification",
        "",
    ]
    for label, metrics_by_size in (("same-task", same_task_identification_metrics), ("cross-task", cross_task_identification_metrics)):
        if not metrics_by_size:
            lines.append(f"- {label}: no feasible identification groups from the cached rollouts")
            continue
        for group_size, metrics in metrics_by_size.items():
            lines.append(
                f"- {label} N={group_size}: top-1 `{float(metrics['top1_accuracy']):.4f}`, top-3 `{float(metrics['top3_accuracy']):.4f}`, MRR `{float(metrics['mean_reciprocal_rank']):.4f}`, mean-rank `{float(metrics['true_key_mean_rank']):.4f}`"
            )
    lines.extend(
        [
            "",
        "## Grouping",
        "",
        ]
    )
    for label, rows in (("same-task", same_task_rows), ("cross-task", cross_task_rows)):
        by_size = defaultdict(list)
        for row in rows:
            by_size[int(row["group_size"])].append(row)
        if not by_size:
            lines.append(f"- {label}: no feasible groups from the cached rollouts")
            continue
        for group_size in sorted(by_size):
            auc = _group_auc(by_size[group_size])
            lines.append(f"- {label} N={group_size}: group AUC `{auc:.4f}`" if np.isfinite(auc) else f"- {label} N={group_size}: group AUC `nan`")
    lines.extend(
        [
            "",
            "## Inversion Sweep",
            "",
        ]
    )
    for row in inversion_step_rows:
        lines.append(
            f"- step={row['inversion_step']}: single AUC `{row['single_trajectory_auc']:.4f}`, same-task `{row['same_task_group_auc']}`, cross-task `{row['cross_task_group_auc']}`, recovery RMS `{row['recovery_rms_mean']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- rollout-level axes: `{json.dumps(rollout_axes, sort_keys=True)}`",
            f"- offline-only axes: `{json.dumps(offline_axes, sort_keys=True)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    reference_variant_config = _reference_variant_config_from_args(args)
    spectral_feature_bands = _spectral_feature_bands_from_args(args)
    episode_spectral_feature_bands = _episode_spectral_feature_bands_from_args(args)
    output_dir = args.output_dir or (args.rollout_dir / "rescoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_rollout_pairs(args.rollout_dir)
    records = [record for pair in pairs for record in pair]
    per_rollout_rows = [_record_to_metadata_row(record) for record in records]
    true_key = int(args.candidate_key) if args.candidate_key is not None else int(pairs[0][1].secret_key)
    candidate_keys = [true_key] + [true_key + 1 + idx for idx in range(args.false_key_count)]
    inversion_steps = sorted({int(step) for step in args.inversion_steps})

    episode_rows: list[EpisodeScoreRow] = []
    for step_count in inversion_steps:
        for record in records:
            episode_rows.extend(
                _score_record_candidates(
                    record,
                    candidate_keys=candidate_keys,
                    step_count=step_count,
                    false_key_count=args.false_key_count,
                    reference_variant_config=reference_variant_config,
                    feature_calibration_mode=args.feature_calibration_mode,
                    global_lag_search_steps=args.global_lag_search_steps,
                    spectral_feature_bands=spectral_feature_bands,
                    episode_spectral_feature_bands=episode_spectral_feature_bands,
                )
            )
    baseline_rows = [row for row in episode_rows if row.inversion_step == inversion_steps[-1]]
    baseline_presence_auc = _presence_auc(baseline_rows, use_z_score=True)
    attribution_rows = [_episode_row_to_dict(row) for row in baseline_rows]
    attribution_accuracy = _attribution_top1_accuracy(attribution_rows)
    identification_top1_accuracy = _identification_topk_accuracy(baseline_rows, top_k=1)
    identification_top3_accuracy = _identification_topk_accuracy(baseline_rows, top_k=3)
    identification_mrr = _identification_mean_reciprocal_rank(baseline_rows)
    identification_mean_rank = _identification_mean_rank(baseline_rows)
    false_key_distribution = [
        row["episode_score"]
        for row in attribution_rows
        if not bool(row["is_true_key"])
    ]

    baseline_true_rows = [row for row in baseline_rows if row.is_true_key]
    same_task_group_rows: list[dict[str, object]] = []
    cross_task_group_rows: list[dict[str, object]] = []
    same_task_identification_rows: list[dict[str, object]] = []
    cross_task_identification_rows: list[dict[str, object]] = []
    for group_size in sorted({int(size) for size in args.group_sizes}):
        same_task_group_rows.extend(
            _sample_group_rows(
                [row for row in baseline_true_rows if row.variant == "plain"],
                grouping_mode="same_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + group_size,
            )
        )
        same_task_group_rows.extend(
            _sample_group_rows(
                [row for row in baseline_true_rows if row.variant == "watermarked"],
                grouping_mode="same_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + 1000 + group_size,
            )
        )
        cross_task_group_rows.extend(
            _sample_group_rows(
                [row for row in baseline_true_rows if row.variant == "plain"],
                grouping_mode="cross_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + 2000 + group_size,
            )
        )
        cross_task_group_rows.extend(
            _sample_group_rows(
                [row for row in baseline_true_rows if row.variant == "watermarked"],
                grouping_mode="cross_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + 3000 + group_size,
            )
        )
        same_task_identification_rows.extend(
            _sample_identification_group_rows(
                baseline_rows,
                grouping_mode="same_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + 8000 + group_size,
            )
        )
        cross_task_identification_rows.extend(
            _sample_identification_group_rows(
                baseline_rows,
                grouping_mode="cross_task",
                group_size=group_size,
                group_samples=args.group_samples,
                seed=args.seed + 9000 + group_size,
            )
        )

    inversion_step_rows: list[dict[str, object]] = []
    for step_count in inversion_steps:
        step_rows = [row for row in episode_rows if row.inversion_step == step_count and row.is_true_key]
        step_same = []
        step_cross = []
        for group_size in sorted({int(size) for size in args.group_sizes}):
            step_same.extend(
                _sample_group_rows(
                    [row for row in step_rows if row.variant == "plain"],
                    grouping_mode="same_task",
                    group_size=group_size,
                    group_samples=args.group_samples,
                    seed=args.seed + 4000 + 10 * step_count + group_size,
                )
            )
            step_same.extend(
                _sample_group_rows(
                    [row for row in step_rows if row.variant == "watermarked"],
                    grouping_mode="same_task",
                    group_size=group_size,
                    group_samples=args.group_samples,
                    seed=args.seed + 5000 + 10 * step_count + group_size,
                )
            )
            step_cross.extend(
                _sample_group_rows(
                    [row for row in step_rows if row.variant == "plain"],
                    grouping_mode="cross_task",
                    group_size=group_size,
                    group_samples=args.group_samples,
                    seed=args.seed + 6000 + 10 * step_count + group_size,
                )
            )
            step_cross.extend(
                _sample_group_rows(
                    [row for row in step_rows if row.variant == "watermarked"],
                    grouping_mode="cross_task",
                    group_size=group_size,
                    group_samples=args.group_samples,
                    seed=args.seed + 7000 + 10 * step_count + group_size,
                )
            )
        inversion_step_rows.append(
            {
                "inversion_step": step_count,
                "single_trajectory_auc": _presence_auc(step_rows, use_z_score=True),
                "same_task_group_auc": _group_auc(step_same),
                "cross_task_group_auc": _group_auc(step_cross),
                "recovery_rms_mean": float(np.mean([row.recovery_rms for row in step_rows], dtype=np.float32)) if step_rows else float("nan"),
            }
        )

    summary_payload = {
        "baseline_presence_auc": baseline_presence_auc,
        "attribution_top1_accuracy": attribution_accuracy,
        "identification_top1_accuracy": identification_top1_accuracy,
        "identification_top3_accuracy": identification_top3_accuracy,
        "identification_mean_reciprocal_rank": identification_mrr,
        "identification_true_key_mean_rank": identification_mean_rank,
        "same_task_group_identification": _identification_group_metrics_by_size(same_task_identification_rows),
        "cross_task_group_identification": _identification_group_metrics_by_size(cross_task_identification_rows),
        "false_key_score_mean": float(np.mean(false_key_distribution)) if false_key_distribution else float("nan"),
        "false_key_score_std": float(np.std(false_key_distribution)) if false_key_distribution else float("nan"),
        "max_same_task_group_size": _max_same_task_group_size(baseline_true_rows, variant="watermarked"),
        "reference_variant_config": dataclasses.asdict(reference_variant_config),
        "feature_calibration_mode": args.feature_calibration_mode,
        "global_lag_search_steps": int(args.global_lag_search_steps),
        "spectral_feature_bands": [_spectral_feature_band_to_json(band) for band in spectral_feature_bands],
        "episode_spectral_feature_bands": [_spectral_feature_band_to_json(band) for band in episode_spectral_feature_bands],
    }

    _write_csv(output_dir / "per_rollout.csv", per_rollout_rows)
    _write_json(output_dir / "per_rollout.json", per_rollout_rows)
    _write_csv(output_dir / "per_episode_scores.csv", [_episode_row_to_dict(row) for row in episode_rows])
    _write_csv(output_dir / "per_group_scores.csv", [*same_task_group_rows, *cross_task_group_rows])
    _write_csv(output_dir / "identification_group_scores.csv", [*same_task_identification_rows, *cross_task_identification_rows])
    _write_csv(output_dir / "attribution_scores.csv", attribution_rows)
    _write_csv(output_dir / "inversion_step_sweep.csv", inversion_step_rows)
    _write_json(output_dir / "summary.json", summary_payload)
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            baseline_presence_auc=baseline_presence_auc,
            identification_top1_accuracy=identification_top1_accuracy,
            identification_top3_accuracy=identification_top3_accuracy,
            identification_mean_reciprocal_rank=identification_mrr,
            identification_true_key_mean_rank=identification_mean_rank,
            same_task_identification_metrics=_identification_group_metrics_by_size(same_task_identification_rows),
            cross_task_identification_metrics=_identification_group_metrics_by_size(cross_task_identification_rows),
            same_task_rows=same_task_group_rows,
            cross_task_rows=cross_task_group_rows,
            inversion_step_rows=inversion_step_rows,
            max_same_task_group_size=_max_same_task_group_size(baseline_true_rows, variant="watermarked"),
            rollout_axes={
                "task_suite_names": sorted({record.task_suite_name for record in records}),
                "chunk_selection_counts": sorted({record.chunk_selection_count for record in records}),
            },
            offline_axes={
                "group_sizes": sorted({int(size) for size in args.group_sizes}),
                "candidate_keys": candidate_keys,
                "inversion_steps": inversion_steps,
                "reference_variant_config": dataclasses.asdict(reference_variant_config),
                "feature_calibration_mode": args.feature_calibration_mode,
                "global_lag_search_steps": int(args.global_lag_search_steps),
                "spectral_feature_bands": [_spectral_feature_band_to_json(band) for band in spectral_feature_bands],
                "episode_spectral_feature_bands": [_spectral_feature_band_to_json(band) for band in episode_spectral_feature_bands],
            },
        ),
        encoding="utf-8",
    )

    print("Saved LIBERO action inversion rescoring")
    print(f"rollout_dir={args.rollout_dir}")
    print(f"output_dir={output_dir}")
    print(f"baseline_presence_auc={baseline_presence_auc:.4f}" if np.isfinite(baseline_presence_auc) else "baseline_presence_auc=nan")
    print(f"attribution_top1_accuracy={attribution_accuracy:.4f}" if np.isfinite(attribution_accuracy) else "attribution_top1_accuracy=nan")
    print(f"identification_top1_accuracy={identification_top1_accuracy:.4f}" if np.isfinite(identification_top1_accuracy) else "identification_top1_accuracy=nan")
    print(f"reference_variant_mode={reference_variant_config.mode}")
    print(f"feature_calibration_mode={args.feature_calibration_mode}")
    print(f"global_lag_search_steps={int(args.global_lag_search_steps)}")
    print(f"spectral_feature_bands={[_spectral_feature_band_to_json(band) for band in spectral_feature_bands]}")
    print(f"episode_spectral_feature_bands={[_spectral_feature_band_to_json(band) for band in episode_spectral_feature_bands]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
