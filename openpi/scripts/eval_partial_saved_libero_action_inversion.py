#!/usr/bin/env python3
"""Lightweight partial snapshot rescoring for saved LIBERO inversion rollouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import eval_libero_action_inversion as _base  # noqa: E402
from scripts import eval_saved_libero_action_inversion as _saved  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", required=True, help="Suite spec in the form name=/abs/rollout_dir")
    parser.add_argument("--output-path", type=pathlib.Path, required=True)
    parser.add_argument("--false-key-count", type=int, default=15)
    parser.add_argument("--group-sizes", type=int, nargs="*", default=[1, 2, 4, 5, 8, 12])
    parser.add_argument("--group-samples", type=int, default=256)
    parser.add_argument("--step-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def _parse_suite_spec(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid suite spec {spec!r}; expected name=/abs/rollout_dir")
    name, raw_path = spec.split("=", 1)
    suite_name = name.strip()
    rollout_dir = pathlib.Path(raw_path.strip())
    if not suite_name:
        raise ValueError(f"Invalid suite spec {spec!r}; suite name is empty")
    return suite_name, rollout_dir


def _json_default(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Unsupported value: {value!r}")


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _fast_load(path: pathlib.Path) -> _saved.SavedInversionRolloutRecord:
    with np.load(path) as payload:
        max_score_windows_raw = int(payload["max_score_windows"].item()) if "max_score_windows" in payload else -1
        subspace_rank_raw = int(payload["subspace_rank"].item()) if "subspace_rank" in payload else -1
        chunk_selection_total_slots_raw = int(payload["chunk_selection_total_slots"].item()) if "chunk_selection_total_slots" in payload else -1
        cached_steps = tuple(int(step) for step in payload["chunk_cached_inversion_steps"]) if "chunk_cached_inversion_steps" in payload else ()
        cached_by_step = payload["chunk_recovered_noise_by_step"] if "chunk_recovered_noise_by_step" in payload else None
        observed_actions = payload["chunk_observed_actions"] if "chunk_observed_actions" in payload else np.zeros((len(payload["chunk_chunk_index"]), 0, 0), dtype=np.float32)
        traces = []
        for index, (chunk_index, executed_steps, selected, observed_action, reference, recovered_noise, injected_noise, raw_actions) in enumerate(
            zip(
                payload["chunk_chunk_index"],
                payload["chunk_executed_steps"],
                payload["chunk_selected"],
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
                for step_offset, cached_step in enumerate(cached_steps):
                    recovered_noise_by_step[int(cached_step)] = np.asarray(cached_by_step[index, step_offset], dtype=np.float32)
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
                    recovered_noise_by_step=recovered_noise_by_step,
                )
            )
        return _saved.SavedInversionRolloutRecord(
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
            null_decoy_count=int(payload["null_decoy_count"].item()) if "null_decoy_count" in payload else 32,
            subspace_rank=None if subspace_rank_raw < 0 else subspace_rank_raw,
            chunk_selection_strategy=str(payload["chunk_selection_strategy"].item()) if "chunk_selection_strategy" in payload else "stateful_online",
            chunk_selection_period=int(payload["chunk_selection_period"].item()) if "chunk_selection_period" in payload else 1,
            chunk_selection_count=int(payload["chunk_selection_count"].item()) if "chunk_selection_count" in payload else 1,
            chunk_selection_total_slots=None if chunk_selection_total_slots_raw < 0 else chunk_selection_total_slots_raw,
            result=_base.online_eval.RolloutResult(
                telemetry=np.asarray(payload["telemetry"], dtype=np.float32),
                success=bool(payload["success"].item()),
                chunk_size=int(payload["chunk_size"].item()),
                task_description=str(payload["task_description"].item()),
                steps=int(payload["steps"].item()),
                execution_segments=tuple(),
                chunk_traces=(),
                executed_actions=np.zeros((0, 0), dtype=np.float32),
            ),
            inversion_traces=tuple(traces),
        )


def _collect_pairs(rollout_dir: pathlib.Path) -> list[tuple[_saved.SavedInversionRolloutRecord, _saved.SavedInversionRolloutRecord]]:
    grouped: dict[tuple[str, int, int, int], dict[str, _saved.SavedInversionRolloutRecord]] = {}
    for path in sorted(rollout_dir.glob("*.npz")):
        record = _fast_load(path)
        key = (record.task_suite_name, record.task_id, record.episode_idx, record.episode_nonce)
        grouped.setdefault(key, {})[record.variant] = record
    return [(variants["plain"], variants["watermarked"]) for _, variants in sorted(grouped.items()) if "plain" in variants and "watermarked" in variants]


def _tpr_at_fpr(positive: Sequence[float], negative: Sequence[float], *, target_fpr: float = 0.01) -> tuple[float, float]:
    if not positive or not negative:
        return float("nan"), float("nan")
    negative_np = np.asarray(negative, dtype=np.float32)
    positive_np = np.asarray(positive, dtype=np.float32)
    threshold = _base.online_eval._calibrate_threshold_for_target_fpr(negative_np, target_fpr)
    tpr, fpr = _base.online_eval._binary_metrics(positive_np, negative_np, threshold)
    return float(tpr), float(fpr)


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32), dtype=np.float32))


def _suite_summary(
    *,
    suite_name: str,
    rollout_dir: pathlib.Path,
    false_key_count: int,
    group_sizes: Sequence[int],
    group_samples: int,
    step_count: int,
    seed: int,
) -> dict[str, object]:
    npz_paths = sorted(rollout_dir.glob("*.npz"))
    pairs = _collect_pairs(rollout_dir)
    if not pairs:
        return {"suite": suite_name, "num_npz_current": len(npz_paths), "num_pairs": 0}
    records = [record for pair in pairs for record in pair]
    true_key = int(pairs[0][1].secret_key)
    candidate_keys = [true_key] + [true_key + 1 + idx for idx in range(false_key_count)]
    rows: list[_saved.EpisodeScoreRow] = []
    for record in records:
        rows.extend(
            _saved._score_record_candidates(
                record,
                candidate_keys=candidate_keys,
                step_count=step_count,
                false_key_count=false_key_count,
            )
        )
    true_rows = [row for row in rows if row.is_true_key]
    watermarked_true_rows = [row for row in true_rows if row.variant == "watermarked"]
    plain_true_rows = [row for row in true_rows if row.variant == "plain"]
    positive = [row.z_score for row in true_rows if row.variant == "watermarked"]
    negative = [row.z_score for row in true_rows if row.variant == "plain"]
    single_tpr, single_fpr = _tpr_at_fpr(positive, negative)
    attribution_rows = [_saved._episode_row_to_dict(row) for row in rows]
    summary = {
        "suite": suite_name,
        "num_npz_current": len(npz_paths),
        "num_pairs": len(pairs),
        "false_key_count": false_key_count,
        "single_auc": _saved._presence_auc(rows, use_z_score=True),
        "single_tpr_at_1pct_fpr": single_tpr,
        "single_fpr_at_threshold": single_fpr,
        "watermarked_z_mean": _mean_or_nan([row.z_score for row in watermarked_true_rows]),
        "plain_z_mean": _mean_or_nan([row.z_score for row in plain_true_rows]),
        "watermarked_episode_score_mean": _mean_or_nan([row.episode_score for row in watermarked_true_rows]),
        "plain_episode_score_mean": _mean_or_nan([row.episode_score for row in plain_true_rows]),
        "attribution_top1_accuracy": _saved._attribution_top1_accuracy(attribution_rows),
        "max_same_task_group_size": _saved._max_same_task_group_size(true_rows, variant="watermarked"),
        "same_task": {},
        "cross_task": {},
        "mixed_group": {},
    }
    for group_size in group_sizes:
        same_rows = []
        cross_rows = []
        same_rows.extend(
            _saved._sample_group_rows(
                [row for row in true_rows if row.variant == "plain"],
                grouping_mode="same_task",
                group_size=int(group_size),
                group_samples=group_samples,
                seed=seed + int(group_size),
            )
        )
        same_rows.extend(
            _saved._sample_group_rows(
                [row for row in true_rows if row.variant == "watermarked"],
                grouping_mode="same_task",
                group_size=int(group_size),
                group_samples=group_samples,
                seed=seed + 1000 + int(group_size),
            )
        )
        cross_rows.extend(
            _saved._sample_group_rows(
                [row for row in true_rows if row.variant == "plain"],
                grouping_mode="cross_task",
                group_size=int(group_size),
                group_samples=group_samples,
                seed=seed + 2000 + int(group_size),
            )
        )
        cross_rows.extend(
            _saved._sample_group_rows(
                [row for row in true_rows if row.variant == "watermarked"],
                grouping_mode="cross_task",
                group_size=int(group_size),
                group_samples=group_samples,
                seed=seed + 3000 + int(group_size),
            )
        )
        mixed_rows = list(cross_rows)
        for label, grouped_rows in (("same_task", same_rows), ("cross_task", cross_rows), ("mixed_group", mixed_rows)):
            positive_group = [float(row["group_score"]) for row in grouped_rows if row["variant"] == "watermarked"]
            negative_group = [float(row["group_score"]) for row in grouped_rows if row["variant"] == "plain"]
            group_tpr, group_fpr = _tpr_at_fpr(positive_group, negative_group)
            summary[label][str(group_size)] = {
                "auc": _saved._group_auc(grouped_rows),
                "tpr_at_1pct_fpr": group_tpr,
                "fpr_at_threshold": group_fpr,
                "samples_per_class": len(positive_group),
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suites = [_parse_suite_spec(spec) for spec in args.suite]
    payload = {
        "step_count": int(args.step_count),
        "false_key_count": int(args.false_key_count),
        "group_sizes": [int(size) for size in args.group_sizes],
        "completed_suites": 0,
        "suites": [],
    }
    _write_json(args.output_path, payload)
    for suite_name, rollout_dir in suites:
        payload["suites"].append(
            _suite_summary(
                suite_name=suite_name,
                rollout_dir=rollout_dir,
                false_key_count=int(args.false_key_count),
                group_sizes=[int(size) for size in args.group_sizes],
                group_samples=int(args.group_samples),
                step_count=int(args.step_count),
                seed=int(args.seed),
            )
        )
        payload["completed_suites"] = len(payload["suites"])
        _write_json(args.output_path, payload)
        print(f"completed_suite={suite_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
