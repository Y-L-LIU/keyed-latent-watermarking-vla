import numpy as np
import pytest
from types import SimpleNamespace

from scripts import eval_libero_action_inversion as _script


def test_parse_args_defaults_inversion_eval_to_whitebox_presence():
    args = _script._parse_args(["--checkpoint-dir", "gs://example/checkpoint"])

    assert args.num_tasks == 1
    assert args.num_trials_per_task == 2
    assert args.num_inversion_steps == 10
    assert args.target_fpr == 0.01
    assert args.detector == "cosine"
    assert args.reference_mode == "bandpass"
    assert args.window_aggregator == "sum"
    assert args.score_step_scope == "executed"
    assert args.inversion_method == "reverse"
    assert args.refinement_steps == 0
    assert args.null_decoy_count == 32
    assert args.subspace_rank is None
    assert args.chunk_selection_strategy == "periodic"
    assert args.chunk_selection_total_slots is None
    assert args.eval_mode == "task_rollout"
    assert args.probe_duration_sec == 10.0
    assert args.max_rollout_steps is None
    assert args.detector_config_name is None
    assert args.detector_checkpoint_dir is None


def test_parse_args_accepts_separate_detector_checkpoint():
    args = _script._parse_args(
        [
            "--config-name",
            "pi05_libero90_from_libero",
            "--checkpoint-dir",
            "/tmp/rollout",
            "--detector-config-name",
            "pi05_libero",
            "--detector-checkpoint-dir",
            "/tmp/base",
        ]
    )

    assert args.config_name == "pi05_libero90_from_libero"
    assert args.checkpoint_dir == "/tmp/rollout"
    assert args.detector_config_name == "pi05_libero"
    assert args.detector_checkpoint_dir == "/tmp/base"


def test_parse_args_accepts_probe_configuration():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--eval-mode",
            "probe_verification",
            "--probe-duration-sec",
            "12.5",
            "--probe-pattern",
            "circle",
            "--probe-amplitude",
            "0.03",
            "--probe-axis-mode",
            "xy",
            "--probe-gripper-mode",
            "hold_closed",
            "--probe-replan-interval",
            "3",
            "--probe-speed-scale",
            "0.2",
        ]
    )

    assert args.eval_mode == "probe_verification"
    assert args.probe_duration_sec == 12.5
    assert args.probe_pattern == "circle"
    assert args.probe_amplitude == 0.03
    assert args.probe_axis_mode == "xy"
    assert args.probe_gripper_mode == "hold_closed"
    assert args.probe_replan_interval == 3
    assert args.probe_speed_scale == 0.2


def test_parse_args_accepts_fm_channel_inverse_flags():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--fm-channel-inverse",
            "--obs-sigma",
            "0.002",
            "--fm-guide-scale",
            "0.75",
            "--fm-guide-schedule",
            "const",
        ]
    )

    assert args.fm_channel_inverse is True
    assert args.obs_sigma == 0.002
    assert args.fm_guide_scale == 0.75
    assert args.fm_guide_schedule == "const"


def test_parse_args_accepts_fm_full_latent_map_flags():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--fm-full-latent-map",
            "--full-map-no-warm-start",
            "--latent-map-iters",
            "40",
            "--latent-map-lr",
            "0.03",
            "--latent-prior-weight",
            "0.5",
        ]
    )

    assert args.fm_full_latent_map is True
    assert args.full_map_no_warm_start is True
    assert args.fm_latent_map is False
    assert args.fm_latent_posterior is False
    assert args.latent_map_iters == 40
    assert args.latent_map_lr == 0.03
    assert args.latent_prior_weight == 0.5


def test_parse_args_accepts_resume_from_rollouts():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--save-rollout-dir",
            "/tmp/rollouts",
            "--resume-from-rollouts",
        ]
    )

    assert args.resume_from_rollouts is True
    assert str(args.save_rollout_dir) == "/tmp/rollouts"


def test_parse_args_accepts_fm_latent_flags():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--fm-latent-map",
            "--latent-map-iters",
            "25",
            "--latent-map-lr",
            "0.05",
            "--latent-prior-weight",
            "0.3",
            "--latent-init-from-bridge",
        ]
    )

    assert args.fm_latent_map is True
    assert args.fm_latent_posterior is False
    assert args.latent_map_iters == 25
    assert args.latent_map_lr == 0.05
    assert args.latent_prior_weight == 0.3
    assert args.latent_init_from_bridge is True


def test_parse_args_accepts_fm_latent_posterior_flags():
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "gs://example/checkpoint",
            "--fm-latent-posterior",
            "--map-num-starts",
            "4",
            "--map-random-seed",
            "123",
            "--posterior-step-size",
            "0.002",
            "--posterior-burnin",
            "30",
            "--posterior-thinning",
            "7",
            "--posterior-num-samples",
            "4",
        ]
    )

    assert args.fm_latent_posterior is True
    assert args.fm_latent_map is False
    assert args.map_num_starts == 4
    assert args.map_random_seed == 123
    assert args.posterior_step_size == 0.002
    assert args.posterior_burnin == 30
    assert args.posterior_thinning == 7
    assert args.posterior_num_samples == 4


def test_complete_raw_actions_from_channel_observation_supports_jax_policy(monkeypatch):
    class _IdentityJaxModel:
        def predict_velocity_and_endpoint(self, model_inputs, x_t, timestep):  # noqa: ANN001, ARG002
            return x_t * 0.0, x_t

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})

    completed = _script._complete_raw_actions_from_channel_observation(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 3.0, dtype=np.float32),
        args=SimpleNamespace(obs_sigma=1e-4, fm_guide_scale=0.5, fm_guide_schedule="linear_decay"),
    )

    assert completed.shape == (10, 32)
    np.testing.assert_allclose(completed[:, :7], 3.0, rtol=0.0, atol=1e-5)


def test_complete_raw_actions_from_channel_observation_sanitizes_nonfinite_hidden_channels(monkeypatch):
    class _NaNHiddenJaxModel:
        action_dim = 32

        def predict_velocity_and_endpoint(self, model_inputs, x_t, timestep):  # noqa: ANN001, ARG002
            endpoint = x_t.at[:, :, 7:].set(np.nan)
            return x_t * 0.0, endpoint

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _NaNHiddenJaxModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})

    completed = _script._complete_raw_actions_from_channel_observation(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 3.0, dtype=np.float32),
        args=SimpleNamespace(obs_sigma=1e-4, fm_guide_scale=0.5, fm_guide_schedule="linear_decay"),
    )

    assert completed.shape == (10, 32)
    assert np.isfinite(completed).all()
    np.testing.assert_allclose(completed[:, :7], 3.0, rtol=0.0, atol=1e-5)


def test_recover_noise_from_channel_observation_latent_supports_jax_map(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})

    out = _script._recover_noise_from_channel_observation_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 2.5, dtype=np.float32),
        raw_action_chunk=np.zeros((10, 32), dtype=np.float32),
        args=SimpleNamespace(
            obs_sigma=0.1,
            fm_latent_map=True,
            fm_latent_posterior=False,
            latent_map_iters=120,
            latent_map_lr=0.2,
            latent_prior_weight=0.0,
            posterior_step_size=1e-3,
            posterior_burnin=5,
            posterior_thinning=2,
            posterior_num_samples=3,
            latent_init_from_bridge=False,
        ),
    )

    assert out["recovered_noise"].shape == (10, 32)
    np.testing.assert_allclose(out["recovered_noise"][:, :7], 2.5, rtol=0.0, atol=7e-2)
    assert out["posterior_init_mode"] == "map"
    assert jnp.isfinite(jnp.asarray(out["recovered_noise"])).all()


def test_recover_noise_from_full_action_latent_supports_jax_map(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})
    monkeypatch.setattr(
        _script,
        "_recover_noise_from_actions",
        lambda policy, obs, raw_actions, args: np.zeros_like(raw_actions, dtype=np.float32),
    )

    target = np.full((10, 32), 1.25, dtype=np.float32)
    out = _script._recover_noise_from_full_action_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        raw_action_chunk=target,
        args=SimpleNamespace(
            obs_sigma=0.1,
            latent_map_iters=120,
            latent_map_lr=0.2,
            latent_prior_weight=0.0,
            posterior_num_samples=3,
            fm_full_latent_map=True,
        ),
    )

    assert out["recovered_noise"].shape == (10, 32)
    np.testing.assert_allclose(out["recovered_noise"], target, rtol=0.0, atol=7e-2)
    assert out["posterior_init_mode"] == "old_reverse"
    assert jnp.isfinite(jnp.asarray(out["recovered_noise"])).all()


def test_recover_noise_from_full_action_latent_supports_jax_map_without_warm_start(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})

    def _unexpected_old_reverse(policy, obs, raw_actions, args):  # noqa: ANN001, ARG001
        raise AssertionError("old reverse warm start should be skipped")

    monkeypatch.setattr(_script, "_recover_noise_from_actions", _unexpected_old_reverse)

    target = np.full((10, 32), 1.25, dtype=np.float32)
    out = _script._recover_noise_from_full_action_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        raw_action_chunk=target,
        args=SimpleNamespace(
            obs_sigma=0.1,
            latent_map_iters=120,
            latent_map_lr=0.2,
            latent_prior_weight=0.0,
            posterior_num_samples=3,
            fm_full_latent_map=True,
            full_map_no_warm_start=True,
        ),
    )

    assert out["recovered_noise"].shape == (10, 32)
    np.testing.assert_allclose(out["recovered_noise"], target, rtol=0.0, atol=7e-2)
    assert out["posterior_init_mode"] == "random"
    assert jnp.isfinite(jnp.asarray(out["recovered_noise"])).all()


def test_recover_noise_from_channel_observation_latent_supports_jax_posterior(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})

    out = _script._recover_noise_from_channel_observation_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 1.5, dtype=np.float32),
        raw_action_chunk=np.zeros((10, 32), dtype=np.float32),
        args=SimpleNamespace(
            obs_sigma=0.1,
            fm_latent_map=False,
            fm_latent_posterior=True,
            latent_map_iters=20,
            latent_map_lr=0.1,
            latent_prior_weight=1.0,
            posterior_step_size=1e-2,
            posterior_burnin=5,
            posterior_thinning=2,
            posterior_num_samples=3,
            latent_init_from_bridge=False,
        ),
    )

    assert out["recovered_noise"].shape == (10, 32)
    assert out["posterior_recovered_noise_samples"].shape == (3, 10, 32)
    assert jnp.isfinite(jnp.asarray(out["posterior_recovered_noise_samples"])).all()
    assert jnp.isfinite(jnp.asarray(out["posterior_recovered_noise_mean"])).all()


def test_recover_noise_from_channel_observation_latent_jax_posterior_warm_starts_from_map(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})
    monkeypatch.setattr(
        _script,
        "_latent_bridge_warm_start",
        lambda policy, obs, env_action_chunk, args: (
            jnp.zeros((1, 10, 32), dtype=jnp.float32),
            "bridge_old_reverse",
        ),
    )
    monkeypatch.setattr(
        _script,
        "_optimize_latent_with_adam_jax",
        lambda **kwargs: jnp.full((1, 10, 32), 4.0, dtype=jnp.float32),
    )

    out = _script._recover_noise_from_channel_observation_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 1.5, dtype=np.float32),
        raw_action_chunk=np.zeros((10, 32), dtype=np.float32),
        args=SimpleNamespace(
            obs_sigma=0.1,
            fm_latent_map=False,
            fm_latent_posterior=True,
            latent_map_iters=20,
            latent_map_lr=0.1,
            latent_prior_weight=1.0,
            posterior_step_size=0.0,
            posterior_burnin=0,
            posterior_thinning=1,
            posterior_num_samples=2,
            latent_init_from_bridge=True,
        ),
    )

    np.testing.assert_allclose(
        out["posterior_recovered_noise_samples"],
        np.full((2, 10, 32), 4.0, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert out["posterior_init_mode"] == "map_from_old_reverse"
    assert out["posterior_chain_init"] == "map_from_old_reverse"


def test_recover_noise_from_channel_observation_latent_jax_posterior_multistart_selects_best_map(monkeypatch):
    import jax.numpy as jnp

    class _IdentityJaxLatentModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyOutputTransform:
        transforms = ()

    class _DummyPolicy:
        _is_pytorch_model = False
        _model = _IdentityJaxLatentModel()
        _sample_kwargs = {"num_steps": 4}
        _output_transform = _DummyOutputTransform()

    monkeypatch.setattr(_script, "_prepare_policy_inputs", lambda policy, obs: ("observation", {}, None))
    monkeypatch.setattr(_script, "_prepare_jax_sampling_context", lambda policy, observation: {"dummy": 1})
    monkeypatch.setattr(
        _script,
        "_latent_bridge_warm_start",
        lambda policy, obs, env_action_chunk, args: (
            jnp.full((1, 10, 32), 3.0, dtype=jnp.float32),
            "bridge_old_reverse",
        ),
    )
    monkeypatch.setattr(
        _script,
        "_build_map_restart_initial_latents",
        lambda z_seed, num_starts, seed: np.asarray(
            [
                np.full((10, 32), 3.0, dtype=np.float32),
                np.full((10, 32), 5.0, dtype=np.float32),
                np.full((10, 32), 1.5, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
    )
    optimize_calls = []

    def _fake_optimize(*, init_noise, loss_fn, num_steps, learning_rate):  # noqa: ANN001
        del loss_fn, num_steps, learning_rate
        base = float(np.asarray(init_noise)[0, 0, 0])
        optimize_calls.append(base)
        if base == 3.0:
            return jnp.full((1, 10, 32), 3.0, dtype=jnp.float32)
        if base == 5.0:
            return jnp.full((1, 10, 32), 6.0, dtype=jnp.float32)
        return jnp.full((1, 10, 32), 1.0, dtype=jnp.float32)

    monkeypatch.setattr(_script, "_optimize_latent_with_adam_jax", _fake_optimize)

    out = _script._recover_noise_from_channel_observation_latent(
        _DummyPolicy(),
        obs={"prompt": "goal", "chunk_index": 0, "episode_nonce": 101},
        env_action_chunk=np.full((10, 7), 1.0, dtype=np.float32),
        raw_action_chunk=np.zeros((10, 32), dtype=np.float32),
        args=SimpleNamespace(
            obs_sigma=1.0,
            fm_latent_map=False,
            fm_latent_posterior=True,
            latent_map_iters=20,
            latent_map_lr=0.1,
            latent_prior_weight=0.0,
            posterior_step_size=0.0,
            posterior_burnin=0,
            posterior_thinning=1,
            posterior_num_samples=2,
            latent_init_from_bridge=True,
            map_num_starts=3,
            map_random_seed=0,
        ),
    )

    assert optimize_calls == [3.0, 5.0, 1.5]
    np.testing.assert_allclose(out["map_restart_recovered_noise"][0], np.full((10, 32), 3.0, dtype=np.float32))
    np.testing.assert_allclose(out["map_restart_recovered_noise"][1], np.full((10, 32), 6.0, dtype=np.float32))
    np.testing.assert_allclose(out["map_restart_recovered_noise"][2], np.full((10, 32), 1.0, dtype=np.float32))
    np.testing.assert_allclose(out["map_restart_energies"], np.asarray([2.0, 12.5, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert out["map_best_restart_index"] == 2
    np.testing.assert_allclose(out["posterior_recovered_noise_samples"], np.full((2, 10, 32), 1.0, dtype=np.float32))
    assert out["posterior_init_mode"] == "map"
    assert out["posterior_chain_init"] == "map"


def test_recover_noise_from_channel_observation_latent_posterior_uses_map_warm_start_for_pytorch(monkeypatch):
    import torch

    class _IdentityTorchModel:
        action_dim = 32

        def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
            return z_init

    class _DummyPolicy:
        _is_pytorch_model = True
        _model = _IdentityTorchModel()
        _pytorch_device = torch.device("cpu")
        _sample_kwargs = {"num_steps": 4}

    monkeypatch.setattr(
        _script,
        "_prepare_pytorch_channel_observation_context",
        lambda policy, obs, env_action_chunk: (
            ("inputs",),
            torch.zeros((1, 10, 7), dtype=torch.float32),
            torch.zeros((5,), dtype=torch.float32),
        ),
    )

    z_map = torch.full((1, 10, 32), 2.5, dtype=torch.float32)
    z_map_alt = torch.full((1, 10, 32), 4.0, dtype=torch.float32)
    solve_calls = []

    class _FakeMAPSolver:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, D401
            pass

        def solve(self, *, model_inputs, y_obs, time_grid, z_init):  # noqa: ANN001
            del model_inputs, y_obs, time_grid
            solve_calls.append(None if z_init is None else float(z_init[0, 0, 0].item()))
            if z_init is None:
                return {
                    "z_map": z_map,
                    "final_obs_mse": torch.tensor(0.25, dtype=torch.float32),
                }
            return {
                "z_map": z_map_alt,
                "final_obs_mse": torch.tensor(0.5, dtype=torch.float32),
            }

    class _FakePosteriorSampler:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, D401
            pass

        def sample(self, *, model_inputs, y_obs, time_grid, z_init):  # noqa: ANN001
            del model_inputs, y_obs, time_grid
            assert torch.equal(z_init, z_map)
            return {
                "z_samples": z_map[:, None, ...].repeat(1, 2, 1, 1),
            }

    monkeypatch.setattr(_script, "FMLatentMAPSolver", _FakeMAPSolver)
    monkeypatch.setattr(_script, "FMLatentPosteriorSampler", _FakePosteriorSampler)

    out = _script._recover_noise_from_channel_observation_latent(
        _DummyPolicy(),
        obs={"prompt": "goal"},
        env_action_chunk=np.full((10, 7), 1.5, dtype=np.float32),
        raw_action_chunk=np.zeros((10, 32), dtype=np.float32),
        args=SimpleNamespace(
            obs_sigma=0.1,
            fm_latent_map=False,
            fm_latent_posterior=True,
            latent_map_iters=5,
            latent_map_lr=0.1,
            latent_prior_weight=1.0,
            posterior_step_size=1e-2,
            posterior_burnin=0,
            posterior_thinning=1,
            posterior_num_samples=2,
            latent_init_from_bridge=False,
            map_num_starts=2,
            map_random_seed=0,
        ),
    )

    assert solve_calls == [None, 0.1257302165031433]
    np.testing.assert_allclose(out["recovered_noise"], np.full((10, 32), 2.5, dtype=np.float32))
    np.testing.assert_allclose(out["map_restart_recovered_noise"][0], np.full((10, 32), 2.5, dtype=np.float32))
    np.testing.assert_allclose(out["map_restart_recovered_noise"][1], np.full((10, 32), 4.0, dtype=np.float32))
    assert out["map_best_restart_index"] == 0
    assert out["posterior_init_mode"] == "map"
    assert out["posterior_chain_init"] == "map"


def test_probe_helpers_resolve_steps_and_prompt():
    assert _script._probe_total_steps(duration_sec=9.2, sample_rate_hz=20.0) == 184
    assert _script._resolve_task_prompt(
        "open the drawer",
        eval_mode="probe_verification",
        probe_prompt="perform a low-speed verification sweep",
    ) == "perform a low-speed verification sweep"
    assert _script._resolve_task_prompt(
        "open the drawer",
        eval_mode="task_rollout",
        probe_prompt="perform a low-speed verification sweep",
    ) == "open the drawer"


def test_pad_trace_sequences_allows_empty_sequences():
    traces = [
        SimpleNamespace(sequence=np.zeros((0, 0), dtype=np.float32)),
        SimpleNamespace(sequence=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)),
    ]

    padded = _script._pad_trace_sequences(traces, attr="sequence", pad_to=3)

    assert padded.shape == (2, 3, 3)
    assert np.isnan(padded[0]).all()
    np.testing.assert_allclose(padded[1, :2], traces[1].sequence)


def test_integrate_reverse_flow_recovers_initial_noise_for_constant_velocity():
    noise = np.full((6, 3), 2.0, dtype=np.float32)
    velocity = np.full((6, 3), 0.5, dtype=np.float32)
    action = noise - velocity

    recovered = _script._integrate_reverse_flow(
        action,
        num_steps=10,
        velocity_fn=lambda _x_t, _time: velocity,
    )

    np.testing.assert_allclose(recovered, noise, rtol=1e-6, atol=1e-6)


def test_score_episode_from_noise_traces_bypasses_old_reverse(monkeypatch):
    monkeypatch.setattr(
        _script,
        "_recover_noise_from_actions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old reverse should not be called")),
    )
    matching = np.ones((4, 2), dtype=np.float32)
    traces = [
        _script.InversionChunkTrace(
            chunk_index=0,
            executed_steps=4,
            selected=True,
            reference=matching,
            recovered_noise=matching,
            injected_noise=matching,
            raw_actions=matching,
        )
    ]

    score, chunk_scores = _script._score_episode_from_noise_traces(
        traces,
        detector="cosine",
        reference_mode="gaussian",
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        aggregator="sum",
        score_step_scope="executed",
        max_windows=None,
    )

    assert score > 0.99
    np.testing.assert_allclose(chunk_scores, [1.0], rtol=0.0, atol=1e-6)


def test_score_episode_from_noise_samples_averages_sample_scores():
    matching = np.ones((4, 2), dtype=np.float32)
    traces = [
        _script.InversionChunkTrace(
            chunk_index=0,
            executed_steps=4,
            selected=True,
            reference=matching,
            recovered_noise=np.zeros_like(matching),
            injected_noise=matching,
            raw_actions=matching,
            posterior_recovered_noise_samples=np.asarray([matching, -matching], dtype=np.float32),
        )
    ]

    score, chunk_scores = _script._score_episode_from_noise_samples(
        traces,
        detector="cosine",
        reference_mode="gaussian",
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        aggregator="sum",
        score_step_scope="executed",
        max_windows=None,
    )

    assert score == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(chunk_scores, [0.0], rtol=0.0, atol=1e-6)


def test_posterior_episode_score_samples_returns_samplewise_scores():
    matching = np.ones((4, 2), dtype=np.float32)
    traces = [
        _script.InversionChunkTrace(
            chunk_index=0,
            executed_steps=4,
            selected=True,
            reference=matching,
            recovered_noise=np.zeros_like(matching),
            injected_noise=matching,
            raw_actions=matching,
            posterior_recovered_noise_samples=np.asarray([matching, -matching], dtype=np.float32),
        )
    ]

    sample_scores, sample_chunk_scores = _script._posterior_episode_score_samples(
        traces,
        detector="cosine",
        reference_mode="gaussian",
        sample_rate_hz=20.0,
        freq_range=(1.0, 2.0),
        aggregator="sum",
        score_step_scope="executed",
        max_windows=None,
    )

    np.testing.assert_allclose(sample_scores, [1.0, -1.0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(sample_chunk_scores[:, 0], [1.0, -1.0], rtol=0.0, atol=1e-6)


def test_save_inversion_rollout_persists_offline_rescore_metadata(tmp_path):
    result = _script.online_eval.RolloutResult(
        telemetry=np.zeros((3, 7), dtype=np.float32),
        success=True,
        chunk_size=4,
        task_description="pick up the mug",
        steps=3,
        execution_segments=(
            _script.online_eval.ExecutionSegment(
                chunk_index=0,
                start_step=0,
                end_step=2,
                executed_steps=2,
            ),
        ),
        chunk_traces=(),
        executed_actions=np.ones((2, 7), dtype=np.float32),
    )
    trace = _script.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((4, 3), dtype=np.float32),
        recovered_noise=np.full((4, 3), 2.0, dtype=np.float32),
        injected_noise=np.full((4, 3), 3.0, dtype=np.float32),
        raw_actions=np.full((4, 7), 4.0, dtype=np.float32),
        selected=True,
        prompt="pick up the mug",
        observation_state=np.arange(7, dtype=np.float32),
        observation_image=np.full((2, 2, 3), 5, dtype=np.uint8),
        observation_wrist_image=np.full((2, 2, 3), 6, dtype=np.uint8),
        map_restart_recovered_noise=np.asarray(
            [
                np.full((4, 3), 6.0, dtype=np.float32),
                np.full((4, 3), 2.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        map_restart_energies=np.asarray([1.5, 0.5], dtype=np.float32),
        map_best_restart_index=1,
        recovered_noise_by_step={
            1: np.full((4, 3), 7.0, dtype=np.float32),
            2: np.full((4, 3), 8.0, dtype=np.float32),
        },
        posterior_recovered_noise_samples=np.asarray(
            [
                np.full((4, 3), 9.0, dtype=np.float32),
                np.full((4, 3), 10.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        posterior_recovered_noise_mean=np.full((4, 3), 9.5, dtype=np.float32),
        posterior_recovered_noise_std=np.full((4, 3), 0.5, dtype=np.float32),
    )
    args = SimpleNamespace(
        eval_mode="task_rollout",
        max_rollout_steps=None,
        detector="wmf",
        reference_mode="gaussian",
        probe_duration_sec=10.0,
        probe_pattern="axis_sweep",
        probe_amplitude=0.04,
        probe_axis_mode="xyz",
        probe_gripper_mode="hold_open",
        probe_replan_interval=5,
        probe_speed_scale=0.35,
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=1,
        chunk_selection_count=5,
        chunk_selection_total_slots=None,
        max_score_windows=None,
        window_aggregator="sum",
        score_step_scope="full_chunk",
        num_inversion_steps=8,
        inversion_method="reverse",
        refinement_steps=0,
        refinement_learning_rate=0.05,
        refinement_latent_l2=1e-4,
        refinement_init_l2=1e-3,
        task_suite_name="libero_spatial",
        fm_channel_inverse=False,
        fm_full_latent_map=True,
        fm_latent_map=False,
        fm_latent_posterior=False,
        map_num_starts=2,
        map_random_seed=0,
    )

    out_path = _script._save_inversion_rollout(
        save_dir=tmp_path,
        task_id=2,
        episode_idx=1,
        episode_nonce=200001,
        variant="watermarked",
        result=result,
        inversion_traces=[trace],
        args=args,
    )

    payload = np.load(out_path)

    assert payload["task_suite_name"].item() == "libero_spatial"
    assert bool(payload["fm_full_latent_map"].item()) is True
    assert np.array_equal(payload["executed_actions"], np.ones((2, 7), dtype=np.float32))
    assert payload["chunk_prompt"][0].item() == "pick up the mug"
    np.testing.assert_allclose(payload["chunk_observation_state"][0], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(payload["chunk_observation_image"][0], np.full((2, 2, 3), 5, dtype=np.uint8))
    np.testing.assert_array_equal(payload["chunk_observation_wrist_image"][0], np.full((2, 2, 3), 6, dtype=np.uint8))
    np.testing.assert_allclose(payload["chunk_map_restart_recovered_noise"][0, 0], np.full((4, 3), 6.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_map_restart_recovered_noise"][0, 1], np.full((4, 3), 2.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_map_restart_energies"][0], np.asarray([1.5, 0.5], dtype=np.float32))
    np.testing.assert_array_equal(payload["chunk_map_best_restart_index"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(payload["chunk_cached_inversion_steps"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_allclose(payload["chunk_recovered_noise_by_step"][0, 0], np.full((4, 3), 7.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_recovered_noise_by_step"][0, 1], np.full((4, 3), 8.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_posterior_recovered_noise_samples"][0, 0], np.full((4, 3), 9.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_posterior_recovered_noise_samples"][0, 1], np.full((4, 3), 10.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_posterior_recovered_noise_mean"][0], np.full((4, 3), 9.5, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_posterior_recovered_noise_std"][0], np.full((4, 3), 0.5, dtype=np.float32))


def test_load_saved_inversion_rollout_round_trips_saved_payload(tmp_path):
    result = _script.online_eval.RolloutResult(
        telemetry=np.zeros((3, 7), dtype=np.float32),
        success=True,
        chunk_size=4,
        task_description="pick up the mug",
        steps=3,
        execution_segments=(
            _script.online_eval.ExecutionSegment(
                chunk_index=0,
                start_step=0,
                end_step=2,
                executed_steps=2,
            ),
        ),
        chunk_traces=(),
        executed_actions=np.ones((2, 7), dtype=np.float32),
    )
    trace = _script.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((4, 3), dtype=np.float32),
        recovered_noise=np.full((4, 3), 2.0, dtype=np.float32),
        injected_noise=np.full((4, 3), 3.0, dtype=np.float32),
        raw_actions=np.full((4, 7), 4.0, dtype=np.float32),
        selected=True,
        prompt="pick up the mug",
        observation_state=np.arange(7, dtype=np.float32),
        observation_image=np.full((2, 2, 3), 5, dtype=np.uint8),
        observation_wrist_image=np.full((2, 2, 3), 6, dtype=np.uint8),
        map_restart_recovered_noise=np.asarray(
            [
                np.full((4, 3), 6.0, dtype=np.float32),
                np.full((4, 3), 2.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        map_restart_energies=np.asarray([1.5, 0.5], dtype=np.float32),
        map_best_restart_index=1,
        recovered_noise_by_step={
            1: np.full((4, 3), 7.0, dtype=np.float32),
            2: np.full((4, 3), 8.0, dtype=np.float32),
        },
        posterior_recovered_noise_samples=np.asarray(
            [
                np.full((4, 3), 9.0, dtype=np.float32),
                np.full((4, 3), 10.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        posterior_recovered_noise_mean=np.full((4, 3), 9.5, dtype=np.float32),
        posterior_recovered_noise_std=np.full((4, 3), 0.5, dtype=np.float32),
    )
    args = SimpleNamespace(
        eval_mode="task_rollout",
        max_rollout_steps=None,
        detector="wmf",
        reference_mode="gaussian",
        probe_duration_sec=10.0,
        probe_pattern="axis_sweep",
        probe_amplitude=0.04,
        probe_axis_mode="xyz",
        probe_gripper_mode="hold_open",
        probe_replan_interval=5,
        probe_speed_scale=0.35,
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=1,
        chunk_selection_count=5,
        chunk_selection_total_slots=None,
        max_score_windows=None,
        window_aggregator="sum",
        score_step_scope="full_chunk",
        num_inversion_steps=8,
        inversion_method="reverse",
        refinement_steps=0,
        refinement_learning_rate=0.05,
        refinement_latent_l2=1e-4,
        refinement_init_l2=1e-3,
        task_suite_name="libero_spatial",
        fm_channel_inverse=False,
        fm_full_latent_map=True,
        fm_latent_map=False,
        fm_latent_posterior=False,
        map_num_starts=2,
        map_random_seed=0,
    )
    out_path = _script._save_inversion_rollout(
        save_dir=tmp_path,
        task_id=2,
        episode_idx=1,
        episode_nonce=200001,
        variant="plain",
        result=result,
        inversion_traces=[trace],
        args=args,
    )

    loaded_result, loaded_traces, loaded_nonce = _script._load_saved_inversion_rollout(out_path)

    assert loaded_nonce == 200001
    assert loaded_result.success is True
    assert loaded_result.chunk_size == 4
    np.testing.assert_allclose(loaded_result.executed_actions, np.ones((2, 7), dtype=np.float32))
    assert len(loaded_traces) == 1
    assert loaded_traces[0].prompt == "pick up the mug"
    np.testing.assert_allclose(loaded_traces[0].recovered_noise, np.full((4, 3), 2.0, dtype=np.float32))
    np.testing.assert_allclose(loaded_traces[0].map_restart_recovered_noise[0], np.full((4, 3), 6.0, dtype=np.float32))
    np.testing.assert_allclose(loaded_traces[0].map_restart_recovered_noise[1], np.full((4, 3), 2.0, dtype=np.float32))
    np.testing.assert_allclose(loaded_traces[0].map_restart_energies, np.asarray([1.5, 0.5], dtype=np.float32))
    assert loaded_traces[0].map_best_restart_index == 1
    np.testing.assert_allclose(loaded_traces[0].recovered_noise_by_step[1], np.full((4, 3), 7.0, dtype=np.float32))
    np.testing.assert_allclose(loaded_traces[0].recovered_noise_by_step[2], np.full((4, 3), 8.0, dtype=np.float32))
    np.testing.assert_allclose(
        loaded_traces[0].posterior_recovered_noise_samples[0],
        np.full((4, 3), 9.0, dtype=np.float32),
    )
    np.testing.assert_allclose(
        loaded_traces[0].posterior_recovered_noise_samples[1],
        np.full((4, 3), 10.0, dtype=np.float32),
    )
    np.testing.assert_allclose(loaded_traces[0].posterior_recovered_noise_mean, np.full((4, 3), 9.5, dtype=np.float32))
    np.testing.assert_allclose(loaded_traces[0].posterior_recovered_noise_std, np.full((4, 3), 0.5, dtype=np.float32))


def test_save_inversion_rollout_handles_unselected_zero_restart_traces(tmp_path):
    result = _script.online_eval.RolloutResult(
        telemetry=np.zeros((3, 7), dtype=np.float32),
        success=True,
        chunk_size=4,
        task_description="pick up the mug",
        steps=3,
        execution_segments=(),
        chunk_traces=(),
        executed_actions=np.ones((2, 7), dtype=np.float32),
    )
    selected_trace = _script.InversionChunkTrace(
        chunk_index=0,
        executed_steps=2,
        reference=np.ones((4, 3), dtype=np.float32),
        recovered_noise=np.full((4, 3), 2.0, dtype=np.float32),
        injected_noise=np.full((4, 3), 3.0, dtype=np.float32),
        raw_actions=np.full((4, 7), 4.0, dtype=np.float32),
        observed_actions=np.full((4, 7), 5.0, dtype=np.float32),
        selected=True,
        map_restart_recovered_noise=np.asarray(
            [
                np.full((4, 3), 6.0, dtype=np.float32),
                np.full((4, 3), 2.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        map_restart_energies=np.asarray([1.5, 0.5], dtype=np.float32),
        map_best_restart_index=1,
        posterior_recovered_noise_samples=np.asarray(
            [
                np.full((4, 3), 9.0, dtype=np.float32),
                np.full((4, 3), 10.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        posterior_recovered_noise_mean=np.full((4, 3), 9.5, dtype=np.float32),
        posterior_recovered_noise_std=np.full((4, 3), 0.5, dtype=np.float32),
    )
    unselected_trace = _script.InversionChunkTrace(
        chunk_index=1,
        executed_steps=0,
        reference=np.ones((4, 3), dtype=np.float32),
        recovered_noise=np.full((4, 3), -2.0, dtype=np.float32),
        injected_noise=np.full((4, 3), -3.0, dtype=np.float32),
        raw_actions=np.full((4, 7), -4.0, dtype=np.float32),
        observed_actions=np.full((4, 7), -5.0, dtype=np.float32),
        selected=False,
        map_restart_recovered_noise=np.zeros((0, 4, 3), dtype=np.float32),
        map_restart_energies=np.zeros((0,), dtype=np.float32),
        map_best_restart_index=-1,
        posterior_recovered_noise_samples=np.zeros((2, 4, 3), dtype=np.float32),
        posterior_recovered_noise_mean=np.zeros((4, 3), dtype=np.float32),
        posterior_recovered_noise_std=np.zeros((4, 3), dtype=np.float32),
    )
    args = SimpleNamespace(
        eval_mode="task_rollout",
        max_rollout_steps=None,
        detector="wmf",
        reference_mode="gaussian",
        probe_duration_sec=10.0,
        probe_pattern="axis_sweep",
        probe_amplitude=0.04,
        probe_axis_mode="xyz",
        probe_gripper_mode="hold_open",
        probe_replan_interval=5,
        probe_speed_scale=0.35,
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=1,
        chunk_selection_count=5,
        chunk_selection_total_slots=None,
        max_score_windows=None,
        window_aggregator="sum",
        score_step_scope="full_chunk",
        num_inversion_steps=8,
        inversion_method="reverse",
        refinement_steps=0,
        refinement_learning_rate=0.05,
        refinement_latent_l2=1e-4,
        refinement_init_l2=1e-3,
        task_suite_name="libero_spatial",
        fm_channel_inverse=False,
        fm_full_latent_map=False,
        fm_latent_map=False,
        fm_latent_posterior=True,
        map_num_starts=2,
        map_random_seed=0,
    )

    out_path = _script._save_inversion_rollout(
        save_dir=tmp_path,
        task_id=2,
        episode_idx=1,
        episode_nonce=200001,
        variant="watermarked",
        result=result,
        inversion_traces=[selected_trace, unselected_trace],
        args=args,
    )

    payload = np.load(out_path)

    assert payload["chunk_map_restart_recovered_noise"].shape == (2, 2, 4, 3)
    np.testing.assert_allclose(payload["chunk_map_restart_recovered_noise"][0, 0], np.full((4, 3), 6.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_map_restart_recovered_noise"][0, 1], np.full((4, 3), 2.0, dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_map_restart_recovered_noise"][1], np.zeros((2, 4, 3), dtype=np.float32))
    np.testing.assert_allclose(payload["chunk_map_restart_energies"][0], np.asarray([1.5, 0.5], dtype=np.float32))
    assert np.isnan(payload["chunk_map_restart_energies"][1]).all()
    np.testing.assert_array_equal(payload["chunk_map_best_restart_index"], np.asarray([1, -1], dtype=np.int32))
