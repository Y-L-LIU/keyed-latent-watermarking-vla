#!/usr/bin/env python3
"""Re-score saved LIBERO rollout NPZs with the current internal-watermark detector."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import dataclasses
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import eval_libero_internal_watermark as online_eval  # noqa: E402


@dataclasses.dataclass(frozen=True)
class SavedRolloutRecord:
    path: pathlib.Path
    task_id: int
    episode_idx: int
    episode_nonce: int
    variant: str
    result: online_eval.RolloutResult


@dataclasses.dataclass(frozen=True)
class AblationSpec:
    name: str
    dims: tuple[int, ...]
    freq_range: tuple[float, float]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=pathlib.Path, required=True)
    parser.add_argument("--secret-key", type=int, default=17)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--freq-min-hz", type=float, default=1.0)
    parser.add_argument("--freq-max-hz", type=float, default=2.0)
    parser.add_argument("--n-tones", type=int, default=4)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--lag-search-steps", type=int, default=2)
    parser.add_argument(
        "--rate-search-factors",
        type=float,
        nargs="*",
        default=online_eval.DEFAULT_RATE_SEARCH_FACTORS,
    )
    parser.add_argument("--whitebox-diagnostics", action="store_true")
    parser.add_argument("--diagnostic-group-size", type=int, default=4)
    parser.add_argument("--skip-ablations", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.rollout_dir.exists():
        raise FileNotFoundError(f"rollout_dir does not exist: {args.rollout_dir}")
    if not args.rollout_dir.is_dir():
        raise NotADirectoryError(f"rollout_dir is not a directory: {args.rollout_dir}")
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("target_fpr must be in [0, 1).")
    if args.diagnostic_group_size <= 0:
        raise ValueError("diagnostic_group_size must be > 0.")


def _load_execution_segments(payload: np.lib.npyio.NpzFile) -> tuple[online_eval.ExecutionSegment, ...]:
    if "segment_chunk_index" not in payload:
        return ()
    return tuple(
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
    )


def _load_chunk_traces(payload: np.lib.npyio.NpzFile) -> tuple[online_eval.ChunkTrace, ...]:
    if "chunk_chunk_index" not in payload:
        return ()
    return tuple(
        online_eval.ChunkTrace(
            chunk_index=int(chunk_index),
            start_step=int(start_step),
            end_step=int(end_step),
            executed_steps=int(executed_steps),
            base_noise=np.asarray(base_noise, dtype=np.float32),
            applied_noise=np.asarray(applied_noise, dtype=np.float32),
            reference=np.asarray(reference, dtype=np.float32),
            predicted_actions=np.asarray(predicted_actions, dtype=np.float32),
        )
        for chunk_index, start_step, end_step, executed_steps, base_noise, applied_noise, reference, predicted_actions in zip(
            payload["chunk_chunk_index"],
            payload["chunk_start_step"],
            payload["chunk_end_step"],
            payload["chunk_executed_steps"],
            payload["chunk_base_noise"],
            payload["chunk_applied_noise"],
            payload["chunk_reference"],
            payload["chunk_predicted_actions"],
            strict=True,
        )
    )


def _load_saved_rollout(path: pathlib.Path) -> SavedRolloutRecord:
    payload = np.load(path)
    result = online_eval.RolloutResult(
        telemetry=np.asarray(payload["telemetry"], dtype=np.float32),
        success=bool(payload["success"].item()),
        chunk_size=int(payload["chunk_size"].item()),
        task_description=str(payload["task_description"].item()),
        steps=int(payload["steps"].item()),
        execution_segments=_load_execution_segments(payload),
        chunk_traces=_load_chunk_traces(payload),
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
    return SavedRolloutRecord(
        path=path,
        task_id=int(payload["task_id"].item()),
        episode_idx=int(payload["episode_idx"].item()),
        episode_nonce=int(payload["episode_nonce"].item()),
        variant=str(payload["variant"].item()),
        result=result,
    )


def _collect_rollout_pairs(
    rollout_dir: pathlib.Path,
) -> list[tuple[SavedRolloutRecord, SavedRolloutRecord]]:
    records = [_load_saved_rollout(path) for path in sorted(rollout_dir.glob("*.npz"))]
    if not records:
        raise FileNotFoundError(f"No .npz rollouts found in {rollout_dir}")

    grouped: dict[tuple[int, int, int], dict[str, SavedRolloutRecord]] = {}
    for record in records:
        key = (record.task_id, record.episode_idx, record.episode_nonce)
        grouped.setdefault(key, {})
        if record.variant in grouped[key]:
            raise ValueError(f"Duplicate variant={record.variant!r} for pair key={key}")
        grouped[key][record.variant] = record

    pairs: list[tuple[SavedRolloutRecord, SavedRolloutRecord]] = []
    for key in sorted(grouped):
        variants = grouped[key]
        if "plain" not in variants or "watermarked" not in variants:
            raise ValueError(f"Missing plain/watermarked pair for key={key}: variants={sorted(variants)}")
        pairs.append((variants["plain"], variants["watermarked"]))
    return pairs


def _make_ablation_specs(base_freq_range: tuple[float, float]) -> list[AblationSpec]:
    f_min, f_max = base_freq_range
    midpoint = 0.5 * (f_min + f_max)
    band_specs = [("base", (f_min, f_max))]
    if f_max - f_min >= 0.5 and midpoint > f_min and midpoint < f_max:
        band_specs.append(("low", (f_min, midpoint)))
        band_specs.append(("high", (midpoint, f_max)))

    channel_specs = [
        ("pos", (0, 1, 2)),
        ("rot", (3, 4, 5)),
        ("posrot", (0, 1, 2, 3, 4, 5)),
    ]
    return [
        AblationSpec(name=f"{channel_name}_{band_name}", dims=dims, freq_range=freq_range)
        for channel_name, dims in channel_specs
        for band_name, freq_range in band_specs
    ]


def _mean_plain_behavior_spectrum(
    pairs: Sequence[tuple[SavedRolloutRecord, SavedRolloutRecord]],
    *,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common_freqs = np.linspace(0.0, sample_rate_hz / 2.0, num=128, dtype=np.float32)
    arm_powers = []
    pos_powers = []
    rot_powers = []
    for plain_record, _ in pairs:
        telemetry = online_eval._prepare_arm_detector_trace(plain_record.result.telemetry)
        freqs, psd_pair, _ = online_eval._welch_spectra(telemetry, telemetry, sample_rate_hz=sample_rate_hz)
        psd = psd_pair[0]
        arm_power = np.mean(psd[:, : min(6, psd.shape[1])], axis=1)
        pos_power = np.mean(psd[:, : min(3, psd.shape[1])], axis=1)
        if psd.shape[1] > 3:
            rot_power = np.mean(psd[:, 3 : min(6, psd.shape[1])], axis=1)
        else:
            rot_power = np.zeros_like(pos_power)
        arm_powers.append(np.interp(common_freqs, freqs, arm_power, left=0.0, right=0.0))
        pos_powers.append(np.interp(common_freqs, freqs, pos_power, left=0.0, right=0.0))
        rot_powers.append(np.interp(common_freqs, freqs, rot_power, left=0.0, right=0.0))
    return (
        common_freqs,
        np.mean(np.stack(arm_powers, axis=0), axis=0),
        np.mean(np.stack(pos_powers, axis=0), axis=0),
        np.mean(np.stack(rot_powers, axis=0), axis=0),
    )


def _top_frequency_peaks(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    top_k: int = 3,
    min_freq_hz: float = 0.1,
) -> list[tuple[float, float]]:
    mask = freqs >= min_freq_hz
    freqs = freqs[mask]
    power = power[mask]
    if freqs.size == 0:
        return []
    order = np.argsort(power)[::-1][:top_k]
    return [(float(freqs[idx]), float(power[idx])) for idx in order]


def _format_frequency_peaks(peaks: Sequence[tuple[float, float]]) -> str:
    if not peaks:
        return "[]"
    return "[" + ", ".join(f"{freq:.2f}Hz:{power:.4e}" for freq, power in peaks) + "]"


def _band_power(freqs: np.ndarray, power: np.ndarray, freq_range: tuple[float, float]) -> float:
    f_min, f_max = freq_range
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return 0.0
    return float(np.mean(power[mask]))


def _score_rollout_with_spec(
    result: online_eval.RolloutResult,
    *,
    secret_key: int,
    sample_rate_hz: float,
    n_tones: int,
    episode_nonce: int,
    threshold: float,
    lag_search_steps: int,
    rate_search_factors: Sequence[float],
    spec: AblationSpec,
) -> float:
    telemetry = online_eval._prepare_arm_detector_trace(result.telemetry)
    reference_config = online_eval.wm.InternalNoiseWatermarkConfig(
        secret_key=secret_key,
        control_freq=sample_rate_hz,
        beta=0.0,
        freq_range=spec.freq_range,
        n_tones=n_tones,
        watermark_dims=tuple(range(result.telemetry.shape[1])),
    )
    reference = online_eval._build_reference_trace_from_segments(
        total_length=result.telemetry.shape[0],
        action_dim=result.telemetry.shape[1],
        sample_rate_hz=sample_rate_hz,
        config=reference_config,
        episode_nonce=episode_nonce,
        execution_segments=result.execution_segments,
    )
    reference = online_eval._prepare_arm_detector_trace(reference)
    telemetry = telemetry[:, spec.dims]
    reference = reference[:, spec.dims]

    best_score = float("-inf")
    for rate in tuple(rate_search_factors) or (1.0,):
        warped_reference = online_eval.wm._resample_trace(reference, rate)
        for lag in range(-lag_search_steps, lag_search_steps + 1):
            shifted_reference = online_eval.wm._shift_trace(warped_reference, lag)
            score, _ = online_eval._multichannel_band_coherence_score(
                telemetry,
                shifted_reference,
                sample_rate_hz=sample_rate_hz,
                freq_range=spec.freq_range,
            )
            best_score = max(best_score, score)
    return best_score


def _evaluate_ablations(
    pairs: Sequence[tuple[SavedRolloutRecord, SavedRolloutRecord]],
    *,
    secret_key: int,
    sample_rate_hz: float,
    n_tones: int,
    target_fpr: float,
    threshold: float | None,
    lag_search_steps: int,
    rate_search_factors: Sequence[float],
    base_freq_range: tuple[float, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for spec in _make_ablation_specs(base_freq_range):
        negative_scores = []
        positive_scores = []
        for plain_record, marked_record in pairs:
            negative_scores.append(
                _score_rollout_with_spec(
                    plain_record.result,
                    secret_key=secret_key,
                    sample_rate_hz=sample_rate_hz,
                    n_tones=n_tones,
                    episode_nonce=plain_record.episode_nonce,
                    threshold=0.0,
                    lag_search_steps=lag_search_steps,
                    rate_search_factors=rate_search_factors,
                    spec=spec,
                )
            )
            positive_scores.append(
                _score_rollout_with_spec(
                    marked_record.result,
                    secret_key=secret_key,
                    sample_rate_hz=sample_rate_hz,
                    n_tones=n_tones,
                    episode_nonce=marked_record.episode_nonce,
                    threshold=0.0,
                    lag_search_steps=lag_search_steps,
                    rate_search_factors=rate_search_factors,
                    spec=spec,
                )
            )
        positive_scores_np = np.asarray(positive_scores, dtype=np.float32)
        negative_scores_np = np.asarray(negative_scores, dtype=np.float32)
        spec_threshold = (
            float(threshold)
            if threshold is not None
            else online_eval._calibrate_threshold_for_target_fpr(negative_scores_np, target_fpr)
        )
        tpr, fpr = online_eval._binary_metrics(positive_scores_np, negative_scores_np, spec_threshold)
        rows.append(
            {
                "name": spec.name,
                "auc": online_eval._roc_auc(positive_scores_np, negative_scores_np),
                "wm_mean": float(np.mean(positive_scores_np)),
                "plain_mean": float(np.mean(negative_scores_np)),
                "threshold": spec_threshold,
                "tpr": tpr,
                "fpr": fpr,
            }
        )
    return rows


def _format_ablation_rows(rows: Sequence[dict[str, float | str]]) -> list[str]:
    lines = ["ablation_scores:"]
    for row in rows:
        lines.append(
            "  "
            f"{row['name']}: auc={row['auc']:.4f} "
            f"wm_mean={row['wm_mean']:.4f} plain_mean={row['plain_mean']:.4f} "
            f"threshold={row['threshold']:.4f} tpr={row['tpr']:.4f} fpr={row['fpr']:.4f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    success_plain: list[float] = []
    success_marked: list[float] = []
    steps_plain: list[int] = []
    steps_marked: list[int] = []
    whitebox_summaries: list[online_eval.WhiteboxDiagnosticSummary] = []

    pairs = _collect_rollout_pairs(args.rollout_dir)
    for plain_record, marked_record in pairs:
        telemetry_dim = marked_record.result.telemetry.shape[1]
        detect_kwargs = dict(
            secret_key=args.secret_key,
            sample_rate_hz=args.sample_rate_hz,
            freq_range=(args.freq_min_hz, args.freq_max_hz),
            n_tones=args.n_tones,
            watermark_dims=tuple(range(telemetry_dim)),
            episode_nonce=marked_record.episode_nonce,
            threshold=args.threshold if args.threshold is not None else 0.0,
            lag_search_steps=args.lag_search_steps,
            rate_search_factors=args.rate_search_factors,
        )
        negative_scores.append(online_eval._detect_presence_for_rollout(plain_record.result, **detect_kwargs).score)
        positive_scores.append(online_eval._detect_presence_for_rollout(marked_record.result, **detect_kwargs).score)
        success_plain.append(float(plain_record.result.success))
        success_marked.append(float(marked_record.result.success))
        steps_plain.append(plain_record.result.steps)
        steps_marked.append(marked_record.result.steps)
        if args.whitebox_diagnostics and plain_record.result.chunk_traces and marked_record.result.chunk_traces:
            whitebox_summaries.append(
                online_eval._summarize_whitebox_pair(
                    plain_record.result,
                    marked_record.result,
                    group_size=args.diagnostic_group_size,
                )
            )

    positive_scores_np = np.asarray(positive_scores, dtype=np.float32)
    negative_scores_np = np.asarray(negative_scores, dtype=np.float32)
    auc = online_eval._roc_auc(positive_scores_np, negative_scores_np)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else online_eval._calibrate_threshold_for_target_fpr(negative_scores_np, args.target_fpr)
    )
    tpr, fpr = online_eval._binary_metrics(positive_scores_np, negative_scores_np, threshold)

    print("Saved LIBERO internal watermark eval")
    print(f"rollout_dir={args.rollout_dir}")
    print(f"num_pairs={len(pairs)}")
    print(online_eval._describe_scores("watermarked_scores", positive_scores_np))
    print(online_eval._describe_scores("plain_scores", negative_scores_np))
    print(f"roc_auc={auc:.4f}")
    if args.threshold is None:
        print(f"target_fpr={args.target_fpr:.4f}")
        print("threshold_source=calibrated_from_plain_scores")
    else:
        print("threshold_source=manual")
    print(f"threshold={threshold:.4f}")
    print(f"tpr={tpr:.4f}")
    print(f"fpr={fpr:.4f}")
    print(f"plain_success_rate={np.mean(success_plain):.4f}")
    print(f"watermarked_success_rate={np.mean(success_marked):.4f}")
    print(f"success_rate_delta={np.mean(success_marked) - np.mean(success_plain):.4f}")
    print(f"plain_mean_steps={np.mean(steps_plain):.2f}")
    print(f"watermarked_mean_steps={np.mean(steps_marked):.2f}")
    behavior_freqs, behavior_arm, behavior_pos, behavior_rot = _mean_plain_behavior_spectrum(
        pairs,
        sample_rate_hz=args.sample_rate_hz,
    )
    print("behavior_spectrum_plain=enabled")
    print(f"behavior_spectrum_arm_top={_format_frequency_peaks(_top_frequency_peaks(behavior_freqs, behavior_arm))}")
    print(f"behavior_spectrum_pos_top={_format_frequency_peaks(_top_frequency_peaks(behavior_freqs, behavior_pos))}")
    print(f"behavior_spectrum_rot_top={_format_frequency_peaks(_top_frequency_peaks(behavior_freqs, behavior_rot))}")
    print(
        "behavior_spectrum_secret_band_power="
        f"arm:{_band_power(behavior_freqs, behavior_arm, (args.freq_min_hz, args.freq_max_hz)):.4e} "
        f"pos:{_band_power(behavior_freqs, behavior_pos, (args.freq_min_hz, args.freq_max_hz)):.4e} "
        f"rot:{_band_power(behavior_freqs, behavior_rot, (args.freq_min_hz, args.freq_max_hz)):.4e}"
    )
    if whitebox_summaries:
        print("whitebox_diagnostics=enabled")
        print(f"whitebox_common_chunks_mean={np.mean([s.common_chunk_count for s in whitebox_summaries]):.2f}")
        print(f"whitebox_common_steps_mean={np.mean([s.common_step_count for s in whitebox_summaries]):.2f}")
        print(
            "whitebox_internal_noise_delta_rms_mean="
            f"{np.mean([s.internal_noise_delta_rms for s in whitebox_summaries]):.6f}"
        )
        print(f"whitebox_action_delta_rms_mean={np.mean([s.action_delta_rms for s in whitebox_summaries]):.6f}")
        print(
            "whitebox_telemetry_delta_rms_mean="
            f"{np.mean([s.telemetry_delta_rms for s in whitebox_summaries]):.6f}"
        )
        mean_group_action = np.mean(
            np.stack([s.group_internal_to_action for s in whitebox_summaries], axis=0),
            axis=0,
        )
        mean_group_telemetry = np.mean(
            np.stack([s.group_internal_to_telemetry for s in whitebox_summaries], axis=0),
            axis=0,
        )
        print(f"whitebox_group_internal_to_action={np.array2string(mean_group_action, precision=4)}")
        print(f"whitebox_group_internal_to_telemetry={np.array2string(mean_group_telemetry, precision=4)}")
    elif args.whitebox_diagnostics:
        print("whitebox_diagnostics=requested_but_unavailable")
    if not args.skip_ablations:
        ablation_rows = _evaluate_ablations(
            pairs,
            secret_key=args.secret_key,
            sample_rate_hz=args.sample_rate_hz,
            n_tones=args.n_tones,
            target_fpr=args.target_fpr,
            threshold=args.threshold,
            lag_search_steps=args.lag_search_steps,
            rate_search_factors=args.rate_search_factors,
            base_freq_range=(args.freq_min_hz, args.freq_max_hz),
        )
        for line in _format_ablation_rows(ablation_rows):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
