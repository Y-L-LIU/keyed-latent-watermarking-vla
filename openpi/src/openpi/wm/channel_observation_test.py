import torch

from openpi.wm import channel_observation as _channel_observation
from openpi.wm import fm_channel_solver as _fm_channel_solver


def test_channel_observation_apply_selects_requested_channels():
    obs = _channel_observation.ChannelObservation(channel_idx=(0, 2, 6))
    a_raw = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)

    result = obs.apply(a_raw)

    expected = a_raw[:, :, (0, 2, 6)]
    assert result.shape == (2, 3, 3)
    torch.testing.assert_close(result, expected)


def test_channel_observation_overwrite_only_replaces_observed_channels():
    obs = _channel_observation.ChannelObservation(channel_idx=(1, 3))
    a_raw = torch.arange(1 * 2 * 5, dtype=torch.float32).reshape(1, 2, 5)
    y_obs = torch.tensor([[[100.0, 200.0], [300.0, 400.0]]], dtype=torch.float32)

    result = obs.overwrite(a_raw, y_obs)

    expected = a_raw.clone()
    expected[:, :, (1, 3)] = y_obs
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result[:, :, (0, 2, 4)], a_raw[:, :, (0, 2, 4)])


class _IdentityEndpointModel:
    def eval(self):
        return self

    def predict_velocity_and_endpoint(self, model_inputs, x_t, t):  # noqa: ANN001, ARG002
        return torch.zeros_like(x_t), x_t


class _NaNHiddenEndpointModel:
    def eval(self):
        return self

    def predict_velocity_and_endpoint(self, model_inputs, x_t, t):  # noqa: ANN001, ARG002
        endpoint = x_t.clone()
        endpoint[:, :, 7:] = float("nan")
        return torch.zeros_like(x_t), endpoint


def test_fm_channel_solver_returns_full_actions_and_matches_observation():
    torch.manual_seed(0)
    obs = _channel_observation.ChannelObservation(obs_sigma=0.2)
    solver = _fm_channel_solver.FMChannelSolver(
        _IdentityEndpointModel(),
        obs,
        _fm_channel_solver.FMChannelSolverConfig(
            guide_scale=5.0,
            guide_schedule="const",
            hard_overwrite_final=True,
        ),
    )
    y_obs = torch.randn(1, 3, 7, dtype=torch.float32)
    time_grid = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)

    completed = solver.complete(model_inputs={"dummy": 1}, y_obs=y_obs, time_grid=time_grid)

    assert completed.shape == (1, 3, 32)
    torch.testing.assert_close(completed[:, :, :7], y_obs, rtol=0.0, atol=1e-6)
    assert torch.isfinite(completed).all()


def test_fm_channel_solver_sanitizes_nonfinite_hidden_channels():
    torch.manual_seed(0)
    obs = _channel_observation.ChannelObservation(obs_sigma=0.2)
    solver = _fm_channel_solver.FMChannelSolver(
        _NaNHiddenEndpointModel(),
        obs,
        _fm_channel_solver.FMChannelSolverConfig(
            guide_scale=5.0,
            guide_schedule="const",
            hard_overwrite_final=True,
        ),
    )
    y_obs = torch.randn(1, 3, 7, dtype=torch.float32)
    time_grid = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)

    completed = solver.complete(model_inputs={"dummy": 1}, y_obs=y_obs, time_grid=time_grid)

    assert completed.shape == (1, 3, 32)
    torch.testing.assert_close(completed[:, :, :7], y_obs, rtol=0.0, atol=1e-6)
    assert torch.isfinite(completed).all()
