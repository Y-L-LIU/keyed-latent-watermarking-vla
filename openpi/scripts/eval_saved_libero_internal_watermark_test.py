import numpy as np

from openpi.policies import watermark as wm
from scripts import eval_libero_internal_watermark as _online
from scripts import eval_saved_libero_internal_watermark as _saved


def _write_rollout_npz(
    out_path,
    *,
    telemetry: np.ndarray,
    variant: str,
    task_id: int = 0,
    episode_idx: int = 0,
    episode_nonce: int = 100000,
) -> None:
    np.savez_compressed(
        out_path,
        telemetry=np.asarray(telemetry, dtype=np.float32),
        success=np.asarray(True),
        chunk_size=np.asarray(5, dtype=np.int32),
        steps=np.asarray(telemetry.shape[0], dtype=np.int32),
        task_id=np.asarray(task_id, dtype=np.int32),
        episode_idx=np.asarray(episode_idx, dtype=np.int32),
        episode_nonce=np.asarray(episode_nonce, dtype=np.int64),
        variant=np.asarray(variant),
        task_description=np.asarray("saved test"),
        segment_chunk_index=np.asarray([0], dtype=np.int32),
        segment_start_step=np.asarray([0], dtype=np.int32),
        segment_end_step=np.asarray([telemetry.shape[0]], dtype=np.int32),
        segment_executed_steps=np.asarray([telemetry.shape[0]], dtype=np.int32),
    )


def test_load_saved_rollout_reconstructs_segments(tmp_path):
    telemetry = np.arange(24, dtype=np.float32).reshape(3, 8)
    path = tmp_path / "task_000_episode_000_plain.npz"
    _write_rollout_npz(path, telemetry=telemetry, variant="plain", task_id=3, episode_idx=7, episode_nonce=1234)

    record = _saved._load_saved_rollout(path)

    assert record.task_id == 3
    assert record.episode_idx == 7
    assert record.episode_nonce == 1234
    assert record.variant == "plain"
    np.testing.assert_allclose(record.result.telemetry, telemetry)
    assert record.result.execution_segments == (
        _online.ExecutionSegment(chunk_index=0, start_step=0, end_step=3, executed_steps=3),
    )


def test_collect_rollout_pairs_matches_plain_and_watermarked(tmp_path):
    telemetry = np.zeros((4, 8), dtype=np.float32)
    _write_rollout_npz(tmp_path / "a_plain.npz", telemetry=telemetry, variant="plain", task_id=1, episode_idx=2)
    _write_rollout_npz(
        tmp_path / "b_marked.npz",
        telemetry=telemetry,
        variant="watermarked",
        task_id=1,
        episode_idx=2,
    )

    pairs = _saved._collect_rollout_pairs(tmp_path)

    assert len(pairs) == 1
    plain_record, marked_record = pairs[0]
    assert plain_record.variant == "plain"
    assert marked_record.variant == "watermarked"
    assert plain_record.task_id == marked_record.task_id == 1
    assert plain_record.episode_idx == marked_record.episode_idx == 2


def test_parse_args_defaults_rate_search_and_ablations():
    args = _saved._parse_args(["--rollout-dir", "/tmp/example"])

    assert args.freq_min_hz == 1.0
    assert args.freq_max_hz == 2.0
    assert args.rate_search_factors == [0.95, 0.975, 1.0, 1.025, 1.05]
    assert args.skip_ablations is False


def test_main_reports_scores_from_saved_rollouts(tmp_path, capsys):
    sample_rate_hz = 20.0
    freq_range = (0.5, 3.0)
    episode_nonce = 100000
    config = wm.InternalNoiseWatermarkConfig(
        secret_key=17,
        control_freq=sample_rate_hz,
        beta=0.0,
        freq_range=freq_range,
        n_tones=4,
        watermark_dims=tuple(range(8)),
    )
    segments = (_online.ExecutionSegment(chunk_index=0, start_step=0, end_step=64, executed_steps=64),)
    reference = _online._build_reference_trace_from_segments(
        total_length=64,
        action_dim=8,
        sample_rate_hz=sample_rate_hz,
        config=config,
        episode_nonce=episode_nonce,
        execution_segments=segments,
    )
    plain_telemetry = np.zeros((64, 8), dtype=np.float32)
    watermarked_telemetry = np.cumsum(reference, axis=0).astype(np.float32)
    _write_rollout_npz(
        tmp_path / "task_000_episode_000_plain.npz",
        telemetry=plain_telemetry,
        variant="plain",
        episode_nonce=episode_nonce,
    )
    _write_rollout_npz(
        tmp_path / "task_000_episode_000_watermarked.npz",
        telemetry=watermarked_telemetry,
        variant="watermarked",
        episode_nonce=episode_nonce,
    )

    exit_code = _saved.main(
        [
            "--rollout-dir",
            str(tmp_path),
            "--secret-key",
            "17",
            "--sample-rate-hz",
            "20",
            "--freq-min-hz",
            "0.5",
            "--freq-max-hz",
            "3.0",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Saved LIBERO internal watermark eval" in captured
    assert "num_pairs=1" in captured
    assert "roc_auc=" in captured
    assert "watermarked_scores:" in captured
    assert "plain_scores:" in captured
    assert "behavior_spectrum_plain=enabled" in captured
    assert "behavior_spectrum_pos_top=" in captured
    assert "behavior_spectrum_rot_top=" in captured
    assert "ablation_scores:" in captured
    assert "pos_base:" in captured
    assert "rot_base:" in captured
    assert "posrot_base:" in captured
