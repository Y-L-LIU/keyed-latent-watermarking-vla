import dataclasses
import numpy as np
import pytest

from scripts import eval_saved_libero_action_inversion as _script


def _make_test_record(*, secret_key: int = 17) -> _script.SavedInversionRolloutRecord:
    trace = _script._base.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((3, 2), dtype=np.float32),
        recovered_noise=np.full((3, 2), 1.0, dtype=np.float32),
        injected_noise=np.full((3, 2), 2.0, dtype=np.float32),
        raw_actions=np.full((3, 2), 3.0, dtype=np.float32),
        selected=True,
    )
    result = _script._base.online_eval.RolloutResult(
        telemetry=np.zeros((0,), dtype=np.float32),
        success=True,
        chunk_size=3,
        task_description="test",
        steps=2,
        execution_segments=(),
        chunk_traces=(),
        executed_actions=np.zeros((0, 0), dtype=np.float32),
    )
    return _script.SavedInversionRolloutRecord(
        path=_script.pathlib.Path("/tmp/test_rollout.npz"),
        task_suite_name="libero_spatial",
        task_id=0,
        episode_idx=0,
        episode_nonce=11,
        variant="watermarked",
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


def test_select_recovered_noise_for_step_prefers_cached_noise():
    trace = _script._base.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((3, 2), dtype=np.float32),
        recovered_noise=np.full((3, 2), 1.0, dtype=np.float32),
        injected_noise=np.full((3, 2), 2.0, dtype=np.float32),
        raw_actions=np.full((3, 2), 3.0, dtype=np.float32),
        recovered_noise_by_step={
            1: np.full((3, 2), 4.0, dtype=np.float32),
            2: np.full((3, 2), 5.0, dtype=np.float32),
        },
    )

    selected = _script._trace_for_inversion_step(trace, step_count=2)

    np.testing.assert_allclose(selected.recovered_noise, np.full((3, 2), 5.0, dtype=np.float32))


def test_max_same_task_group_size_uses_per_task_episode_counts():
    rows = [
        _script.EpisodeScoreRow(task_id=0, episode_idx=0, variant="plain", candidate_key=17, is_true_key=False, episode_score=0.1, z_score=-0.1, inversion_step=8, selected_window_count=5, recovery_rms=0.3),
        _script.EpisodeScoreRow(task_id=0, episode_idx=1, variant="plain", candidate_key=17, is_true_key=False, episode_score=0.2, z_score=-0.2, inversion_step=8, selected_window_count=5, recovery_rms=0.4),
        _script.EpisodeScoreRow(task_id=1, episode_idx=0, variant="plain", candidate_key=17, is_true_key=False, episode_score=0.3, z_score=-0.3, inversion_step=8, selected_window_count=5, recovery_rms=0.5),
    ]

    assert _script._max_same_task_group_size(rows, variant="plain") == 2


def test_attribution_top1_accuracy_counts_true_key_wins():
    rows = [
        {"task_id": 0, "episode_idx": 0, "variant": "watermarked", "candidate_key": 17, "is_true_key": True, "episode_score": 1.2},
        {"task_id": 0, "episode_idx": 0, "variant": "watermarked", "candidate_key": 18, "is_true_key": False, "episode_score": 0.7},
        {"task_id": 1, "episode_idx": 0, "variant": "watermarked", "candidate_key": 17, "is_true_key": True, "episode_score": 0.6},
        {"task_id": 1, "episode_idx": 0, "variant": "watermarked", "candidate_key": 18, "is_true_key": False, "episode_score": 0.9},
    ]

    assert _script._attribution_top1_accuracy(rows) == 0.5


def test_identification_metrics_use_identification_rank_for_watermarked_episodes():
    rows = [
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=0,
            variant="watermarked",
            candidate_key=17,
            is_true_key=True,
            episode_score=1.0,
            z_score=0.1,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.3,
            identification_score=2.0,
            identification_rank=1,
        ),
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=0,
            variant="watermarked",
            candidate_key=18,
            is_true_key=False,
            episode_score=3.0,
            z_score=0.2,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.3,
            identification_score=1.0,
            identification_rank=2,
        ),
        _script.EpisodeScoreRow(
            task_id=1,
            episode_idx=0,
            variant="watermarked",
            candidate_key=17,
            is_true_key=True,
            episode_score=2.0,
            z_score=0.3,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.4,
            identification_score=0.5,
            identification_rank=2,
        ),
        _script.EpisodeScoreRow(
            task_id=1,
            episode_idx=0,
            variant="watermarked",
            candidate_key=18,
            is_true_key=False,
            episode_score=4.0,
            z_score=0.4,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.4,
            identification_score=1.5,
            identification_rank=1,
        ),
        _script.EpisodeScoreRow(
            task_id=2,
            episode_idx=0,
            variant="plain",
            candidate_key=17,
            is_true_key=True,
            episode_score=5.0,
            z_score=0.5,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.5,
            identification_score=10.0,
            identification_rank=1,
        ),
    ]

    assert _script._identification_topk_accuracy(rows, top_k=1) == 0.5
    assert _script._identification_topk_accuracy(rows, top_k=3) == 1.0
    assert _script._identification_mean_rank(rows) == 1.5
    assert _script._identification_mean_reciprocal_rank(rows) == 0.75


def test_sample_identification_group_rows_aggregates_scores_across_episodes():
    rows = [
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=0,
            variant="watermarked",
            candidate_key=17,
            is_true_key=True,
            episode_score=1.0,
            z_score=0.1,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.2,
            identification_score=1.0,
            identification_rank=2,
        ),
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=0,
            variant="watermarked",
            candidate_key=18,
            is_true_key=False,
            episode_score=2.0,
            z_score=0.2,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.2,
            identification_score=3.0,
            identification_rank=1,
        ),
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=1,
            variant="watermarked",
            candidate_key=17,
            is_true_key=True,
            episode_score=4.0,
            z_score=0.3,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.3,
            identification_score=5.0,
            identification_rank=1,
        ),
        _script.EpisodeScoreRow(
            task_id=0,
            episode_idx=1,
            variant="watermarked",
            candidate_key=18,
            is_true_key=False,
            episode_score=1.0,
            z_score=0.4,
            inversion_step=8,
            selected_window_count=5,
            recovery_rms=0.3,
            identification_score=1.0,
            identification_rank=2,
        ),
    ]

    group_rows = _script._sample_identification_group_rows(
        rows,
        grouping_mode="same_task",
        group_size=2,
        group_samples=1,
        seed=7,
    )

    assert len(group_rows) == 1
    assert group_rows[0]["true_key"] == 17
    assert group_rows[0]["predicted_key"] == 17
    assert group_rows[0]["true_rank"] == 1
    assert group_rows[0]["top1_correct"] == 1
    assert group_rows[0]["top3_correct"] == 1
    assert group_rows[0]["true_minus_best_wrong_score"] == pytest.approx(2.0)


def test_identification_group_metrics_by_size_summarizes_group_rows():
    rows = [
        {
            "group_size": 2,
            "top1_correct": 1,
            "top3_correct": 1,
            "reciprocal_rank": 1.0,
            "true_rank": 1,
        },
        {
            "group_size": 2,
            "top1_correct": 0,
            "top3_correct": 1,
            "reciprocal_rank": 0.5,
            "true_rank": 2,
        },
    ]

    summary = _script._identification_group_metrics_by_size(rows)

    assert summary == {
        "2": {
            "group_count": 2,
            "top1_accuracy": 0.5,
            "top3_accuracy": 1.0,
            "mean_reciprocal_rank": 0.75,
            "true_key_mean_rank": 1.5,
        }
    }


def test_required_candidate_score_keys_covers_false_key_normalization_tail():
    assert _script._required_candidate_score_keys([17, 18, 19], false_key_count=2) == [17, 18, 19, 20, 21]


def test_reference_config_for_candidate_preserves_rollout_beta():
    record = _make_test_record()

    config = _script._reference_config_for_candidate(record, candidate_key=23)

    assert config.secret_key == 23
    assert config.beta == 0.02


def test_parse_args_defaults_to_final_inversion_step_only(tmp_path):
    args = _script._parse_args(["--rollout-dir", str(tmp_path)])

    assert args.inversion_steps == [8]
    assert _script._reference_variant_config_from_args(args) == _script.ReferenceVariantConfig()
    assert args.feature_calibration_mode == "identity"
    assert args.global_lag_search_steps == 0
    assert _script._spectral_feature_bands_from_args(args) == (None,)


def test_parse_spectral_feature_bands_accepts_full_and_numeric_bands(tmp_path):
    args = _script._parse_args(
        [
            "--rollout-dir",
            str(tmp_path),
            "--spectral-feature-bands",
            "full",
            "0.2:1.2",
            "0.2,1.6",
        ]
    )

    assert _script._spectral_feature_bands_from_args(args) == (None, (0.2, 1.2), (0.2, 1.6))


def test_spectral_low_band_ignores_high_frequency_mismatch():
    sample_rate_hz = 20.0
    steps = np.arange(80, dtype=np.float32) / sample_rate_hz
    low = np.sin(2.0 * np.pi * 0.5 * steps)
    high = np.sin(2.0 * np.pi * 4.0 * steps)
    reference = (low + high)[:, None].astype(np.float32)
    recovered = (low - high)[:, None].astype(np.float32)

    full_score = _script._score_chunk_noise_similarity_with_reference_variants(
        recovered,
        reference,
        detector="cosine",
        reference_mode="gaussian",
        sample_rate_hz=sample_rate_hz,
        freq_range=(1.0, 2.0),
    )
    low_score = _script._score_chunk_noise_similarity_with_reference_variants(
        recovered,
        reference,
        detector="cosine",
        reference_mode="gaussian",
        sample_rate_hz=sample_rate_hz,
        freq_range=(1.0, 2.0),
        spectral_feature_band=(0.2, 1.2),
    )

    assert full_score < 0.1
    assert low_score > 0.9


def test_episode_spectral_feature_band_can_use_low_frequency_across_short_windows():
    sample_rate_hz = 20.0
    steps = np.arange(80, dtype=np.float32) / sample_rate_hz
    low = np.sin(2.0 * np.pi * 0.5 * steps)
    high = np.sin(2.0 * np.pi * 4.0 * steps)
    reference = (low + high)[:, None].astype(np.float32)
    recovered = (low - high)[:, None].astype(np.float32)
    traces = [
        _script._base.InversionChunkTrace(
            chunk_index=index,
            executed_steps=5,
            reference=reference[5 * index : 5 * (index + 1)],
            recovered_noise=recovered[5 * index : 5 * (index + 1)],
            injected_noise=np.zeros((5, 1), dtype=np.float32),
            raw_actions=np.zeros((5, 1), dtype=np.float32),
            selected=True,
        )
        for index in range(16)
    ]

    scores = _script._episode_spectral_score_vector(
        traces,
        reference_mode="gaussian",
        sample_rate_hz=sample_rate_hz,
        freq_range=(1.0, 2.0),
        score_step_scope="full_chunk",
        max_windows=None,
        base_detector="cosine",
        spectral_feature_bands=((0.2, 1.2),),
    )

    np.testing.assert_allclose(scores, [1.0], atol=1e-5)


def test_window_key_zscore_feature_calibration_removes_window_common_mode():
    feature_vectors = {
        17: np.asarray([10.0, 7.0], dtype=np.float32),
        18: np.asarray([8.0, 7.0], dtype=np.float32),
        19: np.asarray([12.0, 7.0], dtype=np.float32),
    }

    calibrated = _script._calibrate_feature_vectors_across_keys(
        feature_vectors,
        mode="window_key_zscore",
    )

    np.testing.assert_allclose(calibrated[17], [0.0, 0.0])
    np.testing.assert_allclose(calibrated[18], [-3.0, 0.0])
    np.testing.assert_allclose(calibrated[19], [3.0, 0.0])


def test_global_lag_fixed_null_scores_calibrates_false_keys_after_lag_max(monkeypatch):
    feature_vectors_by_lag = {
        -1: {
            17: np.asarray([0.0], dtype=np.float32),
            18: np.asarray([10.0], dtype=np.float32),
            19: np.asarray([20.0], dtype=np.float32),
        },
        0: {
            17: np.asarray([30.0], dtype=np.float32),
            18: np.asarray([8.0], dtype=np.float32),
            19: np.asarray([22.0], dtype=np.float32),
        },
    }
    monkeypatch.setattr(
        _script._base,
        "_wmf_score_from_vectors",
        lambda feature, null_matrix, *, subspace_rank=None: float(feature[0] - np.mean(null_matrix[:, 0])),
    )

    candidate_score, false_scores = _script._global_lag_fixed_null_candidate_scores(
        detector="wmf",
        candidate_key=17,
        false_key_count=2,
        feature_vectors_by_lag=feature_vectors_by_lag,
        subspace_rank=None,
    )

    assert candidate_score == pytest.approx(15.0)
    np.testing.assert_allclose(false_scores, [-5.0, 7.0])


def test_posterior_global_lag_score_maxes_lag_per_sample(monkeypatch):
    feature_vectors_by_lag = {
        -1: {
            17: np.asarray([[0.0], [5.0]], dtype=np.float32),
            18: np.asarray([[10.0], [20.0]], dtype=np.float32),
            19: np.asarray([[20.0], [40.0]], dtype=np.float32),
        },
        0: {
            17: np.asarray([[30.0], [1.0]], dtype=np.float32),
            18: np.asarray([[8.0], [4.0]], dtype=np.float32),
            19: np.asarray([[22.0], [6.0]], dtype=np.float32),
        },
    }
    monkeypatch.setattr(
        _script._base,
        "_wmf_score_from_vectors",
        lambda feature, null_matrix, *, subspace_rank=None: float(feature[0] - np.mean(null_matrix[:, 0])),
    )

    score = _script._posterior_global_lag_cached_score_from_feature_vectors(
        detector="wmf",
        candidate_key=17,
        false_key_count=2,
        feature_vectors_by_lag=feature_vectors_by_lag,
        subspace_rank=None,
    )

    assert score.episode_score == pytest.approx(5.5)
    assert score.episode_score_std == pytest.approx(9.5)
    assert score.posterior_sample_count == 2


def test_control_nuisance_lag_search_recovers_shifted_reference_window():
    reference = np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    recovered = _script._base.online_eval.wm._shift_trace(reference, lag=1)
    trace = _script._base.InversionChunkTrace(
        chunk_index=0,
        executed_steps=4,
        reference=reference,
        recovered_noise=recovered,
        injected_noise=np.zeros_like(reference),
        raw_actions=np.zeros_like(reference),
        selected=True,
    )

    baseline = _script._window_score_vector_with_reference_variants(
        [trace],
        reference_mode="gaussian",
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        score_step_scope="full_chunk",
        max_windows=None,
        base_detector="cosine",
        reference_variant_config=_script.ReferenceVariantConfig(),
    )
    robust = _script._window_score_vector_with_reference_variants(
        [trace],
        reference_mode="gaussian",
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        score_step_scope="full_chunk",
        max_windows=None,
        base_detector="cosine",
        reference_variant_config=_script.ReferenceVariantConfig(
            mode="control_nuisance_max",
            lag_search_steps=1,
        ),
    )

    assert baseline[0] == pytest.approx(0.0)
    assert robust[0] == pytest.approx(1.0)


def test_validate_args_rejects_invalid_reference_variant_knobs(tmp_path):
    args = _script._parse_args(
        [
            "--rollout-dir",
            str(tmp_path),
            "--reference-variant-mode",
            "control_nuisance_max",
            "--reference-lag-search-steps",
            "-1",
        ]
    )

    with pytest.raises(ValueError, match="reference_lag_search_steps"):
        _script._validate_args(args)


def test_score_record_candidates_reuses_false_key_raw_scores(monkeypatch):
    record = dataclasses.replace(_make_test_record(), detector="cosine")
    seen = []

    def fake_raw_score(record, *, traces, candidate_key, false_key_count):
        del record, traces, false_key_count
        seen.append(int(candidate_key))
        return _script.CachedEpisodeScore(candidate_key=int(candidate_key), episode_score=float(candidate_key))

    monkeypatch.setattr(_script, "_score_record_candidate_raw", fake_raw_score)

    rows = _script._score_record_candidates(
        record,
        candidate_keys=[17, 18, 19],
        step_count=8,
        false_key_count=2,
    )

    assert seen == [17, 18, 19, 20, 21]
    assert [row.candidate_key for row in rows] == [17, 18, 19]
    np.testing.assert_allclose([row.episode_score for row in rows], [17.0, 18.0, 19.0])
    np.testing.assert_allclose([row.z_score for row in rows], [-3.0, -3.0, -3.0])
    np.testing.assert_allclose([row.episode_score_std for row in rows], [0.0, 0.0, 0.0])
    np.testing.assert_allclose([row.posterior_sample_count for row in rows], [0, 0, 0])


def test_score_record_candidates_uses_fixed_null_detector_for_wmf(monkeypatch):
    record = dataclasses.replace(_make_test_record(), detector="wmf")

    monkeypatch.setattr(
        _script,
        "_candidate_feature_vectors",
        lambda record, *, traces, candidate_keys: {
            17: np.asarray([1.0], dtype=np.float32),
            18: np.asarray([10.0], dtype=np.float32),
            19: np.asarray([20.0], dtype=np.float32),
        },
    )

    calls = []

    def fake_wmf_score(feature, null_matrix, *, subspace_rank=None):
        del subspace_rank
        calls.append((float(feature[0]), tuple(float(value) for value in null_matrix[:, 0])))
        return float(feature[0] - np.mean(null_matrix[:, 0]))

    monkeypatch.setattr(_script._base, "_wmf_score_from_vectors", fake_wmf_score)

    rows = _script._score_record_candidates(
        record,
        candidate_keys=[17],
        step_count=8,
        false_key_count=2,
    )

    assert calls == [
        (1.0, (10.0, 20.0)),
        (10.0, (10.0, 20.0)),
        (20.0, (10.0, 20.0)),
    ]
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0].episode_score, -14.0)
    np.testing.assert_allclose(rows[0].z_score, -2.8)


def test_score_record_candidates_adds_identification_scores_from_candidate_set(monkeypatch):
    record = dataclasses.replace(_make_test_record(), detector="wmf")

    monkeypatch.setattr(
        _script,
        "_candidate_feature_vectors",
        lambda record, *, traces, candidate_keys: {
            17: np.asarray([10.0], dtype=np.float32),
            18: np.asarray([1.0], dtype=np.float32),
            19: np.asarray([2.0], dtype=np.float32),
            20: np.asarray([100.0], dtype=np.float32),
            21: np.asarray([200.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        _script._base,
        "_wmf_score_from_vectors",
        lambda feature, null_matrix, *, subspace_rank=None: float(feature[0] - np.mean(null_matrix[:, 0])),
    )

    rows = _script._score_record_candidates(
        record,
        candidate_keys=[17, 18, 19],
        step_count=8,
        false_key_count=2,
    )

    np.testing.assert_allclose([row.identification_score for row in rows], [8.5, -5.0, -3.5])
    assert [row.identification_rank for row in rows] == [1, 3, 2]


def test_required_candidate_feature_keys_covers_null_bank_tail():
    assert _script._required_candidate_feature_keys([17], false_key_count=2) == [17, 18, 19, 20, 21]


def test_score_record_candidates_uses_shared_posterior_feature_cache_for_wmf(monkeypatch):
    trace = dataclasses.replace(
        _make_test_record().inversion_traces[0],
        posterior_recovered_noise_samples=np.full((2, 3, 2), 5.0, dtype=np.float32),
    )
    record = dataclasses.replace(_make_test_record(), detector="wmf", inversion_traces=(trace,))
    seen_feature_keys = []

    def fake_posterior_feature_vectors(record, *, traces, candidate_keys):
        del record, traces
        seen_feature_keys.append(tuple(int(key) for key in candidate_keys))
        return {
            17: np.asarray([[1.0], [2.0]], dtype=np.float32),
            18: np.asarray([[10.0], [20.0]], dtype=np.float32),
            19: np.asarray([[20.0], [40.0]], dtype=np.float32),
            20: np.asarray([[30.0], [60.0]], dtype=np.float32),
            21: np.asarray([[40.0], [80.0]], dtype=np.float32),
        }

    monkeypatch.setattr(_script, "_posterior_candidate_feature_vectors", fake_posterior_feature_vectors, raising=False)
    monkeypatch.setattr(
        _script,
        "_score_record_candidate_raw",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw score path should not run for posterior wmf")),
    )
    monkeypatch.setattr(
        _script._base,
        "_wmf_score_from_vectors",
        lambda feature, null_matrix, *, subspace_rank=None: float(feature[0] - np.mean(null_matrix[:, 0])),
    )

    rows = _script._score_record_candidates(
        record,
        candidate_keys=[17],
        step_count=8,
        false_key_count=2,
    )

    assert seen_feature_keys == [(17, 18, 19, 20, 21)]
    assert len(rows) == 1
    assert rows[0].episode_score == pytest.approx(-21.0)
    assert rows[0].episode_score_std == pytest.approx(7.0)
    assert rows[0].episode_score_q05 == pytest.approx(-27.3)
    assert rows[0].episode_score_q95 == pytest.approx(-14.7)
    assert rows[0].posterior_sample_count == 2
    assert rows[0].z_score == pytest.approx(1.5)


def test_load_inversion_traces_reads_posterior_fields(tmp_path):
    path = tmp_path / "rollout.npz"
    np.savez(
        path,
        chunk_chunk_index=np.asarray([0], dtype=np.int32),
        chunk_executed_steps=np.asarray([2], dtype=np.int32),
        chunk_selected=np.asarray([True]),
        chunk_reference=np.ones((1, 3, 2), dtype=np.float32),
        chunk_recovered_noise=np.full((1, 3, 2), 1.0, dtype=np.float32),
        chunk_injected_noise=np.full((1, 3, 2), 2.0, dtype=np.float32),
        chunk_raw_actions=np.full((1, 3, 2), 3.0, dtype=np.float32),
        chunk_observed_actions=np.full((1, 3, 2), 4.0, dtype=np.float32),
        chunk_posterior_recovered_noise_samples=np.full((1, 2, 3, 2), 5.0, dtype=np.float32),
        chunk_posterior_recovered_noise_mean=np.full((1, 3, 2), 6.0, dtype=np.float32),
        chunk_posterior_recovered_noise_std=np.full((1, 3, 2), 0.5, dtype=np.float32),
    )

    traces = _script._load_inversion_traces(np.load(path))

    assert len(traces) == 1
    np.testing.assert_allclose(traces[0].posterior_recovered_noise_samples, np.full((2, 3, 2), 5.0, dtype=np.float32))
    np.testing.assert_allclose(traces[0].posterior_recovered_noise_mean, np.full((3, 2), 6.0, dtype=np.float32))
    np.testing.assert_allclose(traces[0].posterior_recovered_noise_std, np.full((3, 2), 0.5, dtype=np.float32))


def test_score_record_candidate_raw_uses_posterior_sample_scores(monkeypatch):
    trace = dataclasses.replace(
        _make_test_record().inversion_traces[0],
        posterior_recovered_noise_samples=np.full((2, 3, 2), 5.0, dtype=np.float32),
    )
    record = dataclasses.replace(_make_test_record(), detector="cosine", inversion_traces=(trace,))

    monkeypatch.setattr(
        _script._base,
        "_retarget_chunk_references",
        lambda traces, **kwargs: list(traces),
    )
    monkeypatch.setattr(
        _script._base,
        "_posterior_episode_score_samples",
        lambda *args, **kwargs: (
            np.asarray([1.0, 3.0], dtype=np.float32),
            np.asarray([[0.25], [0.75]], dtype=np.float32),
        ),
    )

    score = _script._score_record_candidate_raw(
        record,
        traces=record.inversion_traces,
        candidate_key=17,
        false_key_count=2,
    )

    assert score.episode_score == pytest.approx(2.0)
    assert score.episode_score_std == pytest.approx(1.0)
    assert score.episode_score_q05 == pytest.approx(1.1)
    assert score.episode_score_q95 == pytest.approx(2.9)
    assert score.posterior_sample_count == 2
