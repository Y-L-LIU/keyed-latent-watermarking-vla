import pathlib

import numpy as np

from scripts import eval_partial_saved_libero_action_inversion as _script
from scripts import eval_saved_libero_action_inversion as _saved


def _make_test_record(*, variant: str, secret_key: int = 17) -> _saved.SavedInversionRolloutRecord:
    trace = _saved._base.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((3, 2), dtype=np.float32),
        recovered_noise=np.full((3, 2), 1.0, dtype=np.float32),
        injected_noise=np.full((3, 2), 2.0, dtype=np.float32),
        raw_actions=np.full((3, 2), 3.0, dtype=np.float32),
        selected=True,
    )
    result = _saved._base.online_eval.RolloutResult(
        telemetry=np.zeros((0,), dtype=np.float32),
        success=True,
        chunk_size=3,
        task_description="test",
        steps=2,
        execution_segments=(),
        chunk_traces=(),
        executed_actions=np.zeros((0, 0), dtype=np.float32),
    )
    return _saved.SavedInversionRolloutRecord(
        path=pathlib.Path(f"/tmp/{variant}.npz"),
        task_suite_name="libero_spatial",
        task_id=0,
        episode_idx=0 if variant == "plain" else 1,
        episode_nonce=11,
        variant=variant,
        eval_mode="task_rollout",
        secret_key=secret_key,
        beta=0.02,
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        n_tones=4,
        detector="wmf",
        reference_mode="gaussian",
        score_step_scope="full_chunk",
        window_aggregator="sum",
        max_score_windows=5,
        null_decoy_count=4,
        subspace_rank=None,
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=1,
        chunk_selection_count=5,
        chunk_selection_total_slots=5,
        result=result,
        inversion_traces=(trace,),
    )


def test_parse_suite_spec_parses_name_and_path():
    name, path = _script._parse_suite_spec("libero_goal=/tmp/goal")

    assert name == "libero_goal"
    assert path == pathlib.Path("/tmp/goal")


def test_suite_summary_batches_candidate_scoring_per_record(monkeypatch, tmp_path):
    plain = _make_test_record(variant="plain")
    watermarked = _make_test_record(variant="watermarked")
    calls = []

    def fake_collect_pairs(rollout_dir):
        assert rollout_dir == tmp_path
        return [(plain, watermarked)]

    def fake_score_record_candidates(record, *, candidate_keys, step_count, false_key_count):
        calls.append((record.variant, tuple(candidate_keys), step_count, false_key_count))
        true_z = -1.0 if record.variant == "plain" else 1.0
        rows = []
        for candidate_key in candidate_keys:
            is_true = int(candidate_key) == int(record.secret_key)
            rows.append(
                _saved.EpisodeScoreRow(
                    task_id=record.task_id,
                    episode_idx=record.episode_idx,
                    variant=record.variant,
                    candidate_key=int(candidate_key),
                    is_true_key=is_true,
                    episode_score=float(candidate_key),
                    z_score=true_z if is_true else 0.0,
                    inversion_step=int(step_count),
                    selected_window_count=5,
                    recovery_rms=0.1,
                )
            )
        return rows

    monkeypatch.setattr(_script, "_collect_pairs", fake_collect_pairs)
    monkeypatch.setattr(_saved, "_score_record_candidates", fake_score_record_candidates)

    summary = _script._suite_summary(
        suite_name="libero_spatial",
        rollout_dir=tmp_path,
        false_key_count=3,
        group_sizes=[1],
        group_samples=8,
        step_count=8,
        seed=7,
    )

    assert summary["suite"] == "libero_spatial"
    assert summary["num_pairs"] == 1
    assert summary["watermarked_z_mean"] == 1.0
    assert summary["plain_z_mean"] == -1.0
    assert summary["watermarked_episode_score_mean"] == 17.0
    assert summary["plain_episode_score_mean"] == 17.0
    assert [call[0] for call in calls] == ["plain", "watermarked"]
    assert all(call[2:] == (8, 3) for call in calls)
