from types import SimpleNamespace

import numpy as np

from scripts import eval_saved_libero_action_inversion as _saved
from scripts import eval_saved_libero_action_inversion_posterior as _script


def _make_trace() -> _saved._base.InversionChunkTrace:
    return _saved._base.InversionChunkTrace(
        chunk_index=3,
        executed_steps=4,
        reference=np.ones((4, 32), dtype=np.float32),
        recovered_noise=np.full((4, 32), 2.5, dtype=np.float32),
        injected_noise=np.full((4, 32), 0.5, dtype=np.float32),
        raw_actions=np.full((4, 32), 1.0, dtype=np.float32),
        observed_actions=np.full((4, 7), 1.5, dtype=np.float32),
        selected=True,
        prompt="pick up the mug",
        observation_state=np.arange(7, dtype=np.float32),
        observation_image=np.full((2, 2, 3), 5, dtype=np.uint8),
        observation_wrist_image=np.full((2, 2, 3), 6, dtype=np.uint8),
        map_restart_recovered_noise=np.asarray([np.full((4, 32), 9.0, dtype=np.float32)], dtype=np.float32),
        map_restart_energies=np.asarray([1.25], dtype=np.float32),
        map_best_restart_index=0,
        posterior_init_mode="map_from_old_reverse",
        posterior_chain_init="map_from_old_reverse",
    )


def test_parse_args_uses_saved_map_only_defaults(tmp_path):
    args = _script._parse_args(
        [
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--rollout-dir",
            str(tmp_path / "rollouts"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.posterior_burnin == 20
    assert args.posterior_thinning == 10
    assert args.posterior_num_samples == 8
    assert args.posterior_map_tether_weight == 1.0
    assert args.posterior_grad_clip_norm == 100.0
    assert not hasattr(args, "map_num_starts")
    assert not hasattr(args, "latent_map_iters")
    assert not hasattr(args, "latent_map_lr")


def test_augment_trace_with_saved_map_posterior_uses_saved_map_directly(monkeypatch):
    trace = _make_trace()
    seen = {}

    monkeypatch.setattr(
        _script._base,
        "_prepare_jax_channel_observation_context",
        lambda policy, *, obs, env_action_chunk: (
            "inputs",
            np.asarray(env_action_chunk, dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )

    def fake_run(policy, *, model_inputs, y_obs, time_grid, z_init, mode, args):
        del policy, model_inputs, y_obs, time_grid, mode, args
        seen["z_init"] = np.asarray(z_init, dtype=np.float32)
        samples = np.asarray([z_init, z_init + 1.0], dtype=np.float32)
        energies = np.asarray([3.0, 1.0], dtype=np.float32)
        return samples, energies

    monkeypatch.setattr(_script, "_run_jax_posterior_from_saved_map", fake_run)

    out = _script._augment_trace_with_saved_map_posterior(
        SimpleNamespace(_is_pytorch_model=False),
        trace=trace,
        episode_nonce=101,
        mode="channel",
        args=SimpleNamespace(
            posterior_num_samples=2,
            posterior_step_size=1e-3,
            posterior_burnin=20,
            posterior_thinning=10,
            posterior_map_tether_weight=1.0,
            posterior_grad_clip_norm=100.0,
            obs_sigma=1e-4,
            latent_prior_weight=1.0,
        ),
    )

    np.testing.assert_allclose(seen["z_init"], trace.recovered_noise)
    np.testing.assert_allclose(out.posterior_recovered_noise_samples[0], trace.recovered_noise)
    np.testing.assert_allclose(out.posterior_recovered_noise_samples[1], trace.recovered_noise + 1.0)
    np.testing.assert_allclose(out.posterior_recovered_noise_mean, trace.recovered_noise + 0.5)
    np.testing.assert_allclose(out.posterior_recovered_noise_std, np.full_like(trace.recovered_noise, 0.5))
    np.testing.assert_allclose(out.map_restart_recovered_noise, trace.map_restart_recovered_noise)
    np.testing.assert_allclose(out.map_restart_energies, trace.map_restart_energies)
    assert out.map_best_restart_index == trace.map_best_restart_index
    assert out.posterior_best_restart_index == 1
    assert out.posterior_best_energy == 1.0
    assert out.posterior_init_mode == "map_from_old_reverse"
    assert out.posterior_chain_init == "map_from_old_reverse"
