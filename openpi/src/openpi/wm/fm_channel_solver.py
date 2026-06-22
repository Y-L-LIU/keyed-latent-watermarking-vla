from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FMChannelSolverConfig:
    guide_scale: float = 0.5
    guide_schedule: str = "linear_decay"
    hard_overwrite_final: bool = True


class FMChannelSolver:
    def __init__(self, model, obs_op, cfg: FMChannelSolverConfig):
        self.model = model.eval() if hasattr(model, "eval") else model
        self.obs_op = obs_op
        self.cfg = cfg

    def _guide_weight(self, step_idx: int, step_count: int) -> float:
        if self.cfg.guide_schedule == "const":
            return float(self.cfg.guide_scale)
        if self.cfg.guide_schedule != "linear_decay":
            raise ValueError(f"Unsupported guide_schedule={self.cfg.guide_schedule!r}")
        frac = step_idx / max(step_count - 1, 1)
        return float(self.cfg.guide_scale) * (1.0 - frac)

    def complete(self, model_inputs, y_obs: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        if y_obs.ndim != 3:
            raise ValueError(f"Expected y_obs with shape [B, T, C], got {tuple(y_obs.shape)}")

        batch_size, horizon, _ = y_obs.shape
        device = y_obs.device
        dtype = y_obs.dtype
        raw_dim = int(getattr(self.model, "action_dim", 32))
        time_grid = torch.as_tensor(time_grid, dtype=dtype, device=device)

        x_t = torch.randn(batch_size, horizon, raw_dim, dtype=dtype, device=device)

        for step_idx in range(len(time_grid) - 1):
            time = time_grid[step_idx]
            next_time = time_grid[step_idx + 1]
            dt = next_time - time
            guide_weight = self._guide_weight(step_idx, len(time_grid) - 1)
            step_sign = float(torch.sign(dt).item()) if float(dt.item()) != 0.0 else 1.0

            x_t = x_t.detach().requires_grad_(True)
            with torch.enable_grad():
                v_t, a_hat_t = self.model.predict_velocity_and_endpoint(model_inputs, x_t, time)
                obs_loss = self.obs_op.loss(a_hat_t, y_obs)
                grad_x = torch.autograd.grad(obs_loss, x_t, only_inputs=True)[0]
                v_t = torch.nan_to_num(v_t)
                grad_x = torch.nan_to_num(grad_x)
                # The OpenPI sampler integrates from t=1 -> 0. Flip the correction sign for
                # reverse-time steps so the Euler update moves toward lower observation loss.
                v_corr = v_t - (step_sign * guide_weight) * grad_x
                v_corr = torch.nan_to_num(v_corr)
            x_next = x_t + dt * v_corr
            x_t = torch.where(torch.isfinite(x_next), x_next, x_t).detach()

        with torch.no_grad():
            _, a_final = self.model.predict_velocity_and_endpoint(model_inputs, x_t, time_grid[-1])
            a_final = torch.where(torch.isfinite(a_final), a_final, x_t)
            a_final = torch.nan_to_num(a_final)
            if self.cfg.hard_overwrite_final:
                a_final = self.obs_op.overwrite(a_final, y_obs)
        return a_final
