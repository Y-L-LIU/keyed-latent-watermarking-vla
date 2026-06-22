from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FMLatentPosteriorConfig:
    obs_sigma: float = 1e-4
    step_size: float = 1e-3
    burnin_steps: int = 100
    thinning: int = 50
    num_samples: int = 8


class FMLatentPosteriorSampler:
    def __init__(self, model, obs_op, cfg: FMLatentPosteriorConfig):
        self.model = model.eval() if hasattr(model, "eval") else model
        self.obs_op = obs_op
        self.cfg = cfg

    def energy(self, model_inputs, z: torch.Tensor, y_obs: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        a_pred = self.model.sample_actions_from_noise(model_inputs, z, time_grid)
        pred_obs = self.obs_op.apply(a_pred)
        obs_loss = 0.5 * (((pred_obs - y_obs) / float(self.cfg.obs_sigma)) ** 2).mean()
        prior_loss = 0.5 * z.square().mean()
        return obs_loss + prior_loss

    def sample(self, model_inputs, y_obs: torch.Tensor, time_grid: torch.Tensor, z_init: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if y_obs.ndim != 3:
            raise ValueError(f"Expected y_obs with shape [B, T, C], got {tuple(y_obs.shape)}")

        batch_size, horizon, _ = y_obs.shape
        device = y_obs.device
        dtype = y_obs.dtype
        raw_dim = int(getattr(self.model, "action_dim", 32))
        time_grid = torch.as_tensor(time_grid, dtype=dtype, device=device)

        if z_init is None:
            z = torch.randn(batch_size, horizon, raw_dim, dtype=dtype, device=device)
        else:
            z = z_init.detach().clone().to(device=device, dtype=dtype)

        samples: list[torch.Tensor] = []
        total_steps = int(self.cfg.burnin_steps) + int(self.cfg.num_samples) * int(self.cfg.thinning)
        step_size = float(self.cfg.step_size)

        for step in range(total_steps):
            z = z.detach().requires_grad_(True)
            energy = self.energy(model_inputs, z, y_obs, time_grid)
            grad_z = torch.autograd.grad(energy, z)[0]
            noise = torch.randn_like(z)
            z = (z - 0.5 * step_size * grad_z + (step_size**0.5) * noise).detach()

            if step >= int(self.cfg.burnin_steps) and (step - int(self.cfg.burnin_steps)) % int(self.cfg.thinning) == 0:
                samples.append(z.clone())

        return {
            "z_samples": torch.stack(samples, dim=1),
        }
