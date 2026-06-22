import numpy as np

from scripts import eval_libero_action_inversion_postprocess_robustness as _script


def test_parse_args_separates_robustness_and_base_argv():
    robustness_args, base_argv = _script._parse_args(
        [
            "--controller-postprocess",
            "clip",
            "--controller-clip-limit",
            "0.15",
            "--run-tag",
            "smoke",
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--task-suite-name",
            "libero_goal",
        ]
    )

    assert robustness_args.controller_postprocess == "clip"
    assert robustness_args.controller_clip_limit == 0.15
    assert robustness_args.run_tag == "smoke"
    assert base_argv == [
        "--checkpoint-dir",
        "gs://example/checkpoint",
        "--task-suite-name",
        "libero_goal",
    ]


def test_controller_postprocessor_clip_limits_actions():
    controller = _script.ControllerPostprocessor(
        config=_script.RobustnessConfig(
            controller_postprocess="clip",
            controller_clip_limit=0.2,
            controller_smooth_alpha=0.5,
            controller_jitter_std=0.0,
            controller_delay_steps=0,
            seed=7,
        ),
        action_dim=2,
    )

    processed = controller.apply_chunk(
        np.asarray([[0.5, -0.3], [0.1, -0.4]], dtype=np.float32),
        episode_nonce=10,
        chunk_index=0,
    )

    np.testing.assert_allclose(
        processed,
        np.asarray([[0.2, -0.2], [0.1, -0.2]], dtype=np.float32),
    )


def test_controller_postprocessor_smooth_carries_state_across_chunks():
    controller = _script.ControllerPostprocessor(
        config=_script.RobustnessConfig(
            controller_postprocess="smooth",
            controller_clip_limit=1.0,
            controller_smooth_alpha=0.5,
            controller_jitter_std=0.0,
            controller_delay_steps=0,
            seed=3,
        ),
        action_dim=1,
    )

    first = controller.apply_chunk(np.asarray([[1.0], [1.0]], dtype=np.float32), episode_nonce=5, chunk_index=0)
    second = controller.apply_chunk(np.asarray([[0.0], [0.0]], dtype=np.float32), episode_nonce=5, chunk_index=1)

    np.testing.assert_allclose(first, np.asarray([[0.5], [0.75]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(second, np.asarray([[0.375], [0.1875]], dtype=np.float32), atol=1e-6)


def test_controller_postprocessor_delay_uses_previous_emitted_action():
    controller = _script.ControllerPostprocessor(
        config=_script.RobustnessConfig(
            controller_postprocess="delay",
            controller_clip_limit=1.0,
            controller_smooth_alpha=0.5,
            controller_jitter_std=0.0,
            controller_delay_steps=1,
            seed=11,
        ),
        action_dim=1,
    )

    first = controller.apply_chunk(np.asarray([[1.0], [2.0]], dtype=np.float32), episode_nonce=9, chunk_index=0)
    second = controller.apply_chunk(np.asarray([[3.0]], dtype=np.float32), episode_nonce=9, chunk_index=1)

    np.testing.assert_allclose(first, np.asarray([[1.0], [1.0]], dtype=np.float32))
    np.testing.assert_allclose(second, np.asarray([[2.0]], dtype=np.float32))


def test_controller_postprocessor_jitter_is_deterministic_per_chunk():
    config = _script.RobustnessConfig(
        controller_postprocess="jitter",
        controller_clip_limit=1.0,
        controller_smooth_alpha=0.5,
        controller_jitter_std=0.1,
        controller_delay_steps=0,
        seed=23,
    )
    controller_a = _script.ControllerPostprocessor(config=config, action_dim=2)
    controller_b = _script.ControllerPostprocessor(config=config, action_dim=2)
    chunk = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    processed_a = controller_a.apply_chunk(chunk, episode_nonce=4, chunk_index=2)
    processed_b = controller_b.apply_chunk(chunk, episode_nonce=4, chunk_index=2)

    np.testing.assert_allclose(processed_a, processed_b)
