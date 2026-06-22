from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FMLatentMAPConfig:
    num_iters: int = 100
    lr: float = 1e-1
    obs_sigma: float = 1e-4
    prior_weight: float = 1.0


class FMLatentMAPSolver:
    def __init__(self, model, obs_op, cfg: FMLatentMAPConfig):
        self.model = model.eval() if hasattr(model, "eval") else model
        self.obs_op = obs_op
        self.cfg = cfg

    def solve(self, model_inputs, y_obs: torch.Tensor, time_grid: torch.Tensor, z_init: torch.Tensor | None = None) -> dict[str, torch.Tensor | float]:
        if y_obs.ndim != 3:
            raise ValueError(f"Expected y_obs with shape [B, T, C], got {tuple(y_obs.shape)}")

        batch_size, horizon, _ = y_obs.shape
        device = y_obs.device
        dtype = y_obs.dtype
        raw_dim = int(getattr(self.model, "action_dim", 32))
        time_grid = torch.as_tensor(time_grid, dtype=dtype, device=device)

        if z_init is None:
            z = torch.randn(batch_size, horizon, raw_dim, dtype=dtype, device=device, requires_grad=True)
        else:
            z = z_init.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        opt = torch.optim.Adam([z], lr=float(self.cfg.lr))

        for _ in range(int(self.cfg.num_iters)):
            opt.zero_grad()
            a_pred = self.model.sample_actions_from_noise(model_inputs, z, time_grid)
            pred_obs = self.obs_op.apply(a_pred)
            obs_loss = 0.5 * (((pred_obs - y_obs) / float(self.cfg.obs_sigma)) ** 2).mean()
            prior_loss = 0.5 * float(self.cfg.prior_weight) * (z.square().mean())
            loss = obs_loss + prior_loss
            loss.backward()
            opt.step()

        with torch.no_grad():
            a_final = self.model.sample_actions_from_noise(model_inputs, z, time_grid)
            final_obs_mse = torch.mean(torch.square(self.obs_op.apply(a_final) - y_obs)).item()

        return {
            "z_map": z.detach(),
            "a_map": a_final.detach(),
            "final_obs_mse": float(final_obs_mse),
        }
