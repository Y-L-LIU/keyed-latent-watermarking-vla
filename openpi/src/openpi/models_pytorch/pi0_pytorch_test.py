import torch

from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch


def test_sample_actions_from_noise_backpropagates_to_noise():
    model = object.__new__(_pi0_pytorch.PI0Pytorch)
    torch.nn.Module.__init__(model)

    def _predict_velocity_and_endpoint(model_inputs, x_t, timestep):  # noqa: ANN001, ARG001
        if timestep.ndim == 0:
            timestep = timestep.expand(x_t.shape[0])
        v_t = 2.0 * x_t + timestep[:, None, None]
        endpoint = x_t - timestep[:, None, None] * v_t
        return v_t, endpoint

    model.predict_velocity_and_endpoint = _predict_velocity_and_endpoint

    z_init = torch.randn(1, 3, 32, dtype=torch.float32, requires_grad=True)
    time_grid = torch.linspace(1.0, 0.0, 5, dtype=torch.float32)

    a_raw = model.sample_actions_from_noise(model_inputs={"dummy": 1}, z_init=z_init, time_grid=time_grid)
    loss = a_raw.square().mean()
    loss.backward()

    assert a_raw.shape == z_init.shape
    assert z_init.grad is not None
    assert torch.linalg.norm(z_init.grad) > 0
