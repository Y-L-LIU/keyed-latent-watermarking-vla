import torch

from openpi.wm import channel_observation as _channel_observation
from openpi.wm import fm_latent_map_solver as _fm_latent_map_solver
from openpi.wm import fm_latent_posterior_sampler as _fm_latent_posterior_sampler


class _IdentityLatentModel:
    action_dim = 32

    def eval(self):
        return self

    def sample_actions_from_noise(self, model_inputs, z_init, time_grid):  # noqa: ANN001, ARG002
        return z_init


def test_fm_latent_map_solver_matches_observed_channels():
    torch.manual_seed(0)
    obs = _channel_observation.ChannelObservation(obs_sigma=0.1)
    solver = _fm_latent_map_solver.FMLatentMAPSolver(
        _IdentityLatentModel(),
        obs,
        _fm_latent_map_solver.FMLatentMAPConfig(
            num_iters=80,
            lr=0.2,
            obs_sigma=0.1,
            prior_weight=0.0,
        ),
    )
    y_obs = torch.randn(1, 3, 7, dtype=torch.float32)
    time_grid = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)

    out = solver.solve(model_inputs={"dummy": 1}, y_obs=y_obs, time_grid=time_grid)

    assert out["z_map"].shape == (1, 3, 32)
    assert out["a_map"].shape == (1, 3, 32)
    torch.testing.assert_close(out["a_map"][:, :, :7], y_obs, rtol=0.0, atol=5e-2)
    assert out["final_obs_mse"] < 5e-3


def test_fm_latent_posterior_sampler_returns_finite_samples():
    torch.manual_seed(0)
    obs = _channel_observation.ChannelObservation(obs_sigma=0.1)
    sampler = _fm_latent_posterior_sampler.FMLatentPosteriorSampler(
        _IdentityLatentModel(),
        obs,
        _fm_latent_posterior_sampler.FMLatentPosteriorConfig(
            obs_sigma=0.1,
            step_size=1e-2,
            burnin_steps=5,
            thinning=2,
            num_samples=3,
        ),
    )
    y_obs = torch.randn(1, 3, 7, dtype=torch.float32)
    time_grid = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)

    out = sampler.sample(model_inputs={"dummy": 1}, y_obs=y_obs, time_grid=time_grid)

    assert out["z_samples"].shape == (1, 3, 3, 32)
    assert torch.isfinite(out["z_samples"]).all()
