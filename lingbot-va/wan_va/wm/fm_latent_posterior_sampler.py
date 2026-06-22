"""ULA posterior sampler for latent noise recovery.

Warm-starts from z_map and samples from the posterior via Unadjusted Langevin.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FMLatentPosteriorConfig:
    obs_sigma: float = 1e-4
    prior_weight: float = 1.0
    step_size: float = 1e-3
    burnin_steps: int = 20
    thinning: int = 10
    num_samples: int = 8
    map_tether_weight: float = 1.0
    grad_clip_norm: float = 100.0


class FMLatentPosteriorSampler:
    """ULA sampler around z_map for posterior uncertainty estimation."""

    def __init__(self, decode_fn, obs_op, cfg: FMLatentPosteriorConfig):
        """
        Args:
            decode_fn: callable(z) -> a_pred [B, C, F, H, 1]
            obs_op: ChannelObservation instance.
            cfg: posterior sampling config.
        """
        self.decode_fn = decode_fn
        self.obs_op = obs_op
        self.cfg = cfg

    def energy(self, z: torch.Tensor, y_obs: torch.Tensor, z_map: torch.Tensor | None = None) -> torch.Tensor:
        a_pred = self.decode_fn(z)
        pred_obs = self.obs_op.apply(a_pred)
        obs_loss = 0.5 * (((pred_obs - y_obs) / float(self.cfg.obs_sigma)) ** 2).mean()
        prior_loss = 0.5 * float(self.cfg.prior_weight) * z.square().mean()
        total = obs_loss + prior_loss
        if z_map is not None and self.cfg.map_tether_weight > 0:
            tether_loss = 0.5 * float(self.cfg.map_tether_weight) * ((z - z_map) ** 2).mean()
            total = total + tether_loss
        return total

    def sample(
        self,
        y_obs: torch.Tensor,
        z_init: torch.Tensor,
        z_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run ULA from z_init (typically z_map), collect posterior samples.

        Args:
            y_obs: observed actions [B, C_obs, F, H, 1]
            z_init: starting point [B, C_total, F, H, 1]
            z_map: MAP estimate for tethering (optional)

        Returns:
            dict with z_samples: [B, num_samples, C_total, F, H, 1]
        """
        z = z_init.detach().clone()
        step_size = float(self.cfg.step_size)
        total_steps = int(self.cfg.burnin_steps) + int(self.cfg.num_samples) * int(self.cfg.thinning)

        samples: list[torch.Tensor] = []

        for step in range(total_steps):
            z = z.detach().requires_grad_(True)
            e = self.energy(z, y_obs, z_map)
            grad_z = torch.autograd.grad(e, z)[0]

            if self.cfg.grad_clip_norm > 0:
                grad_norm = grad_z.norm()
                if grad_norm > self.cfg.grad_clip_norm:
                    grad_z = grad_z * (self.cfg.grad_clip_norm / grad_norm)

            noise = torch.randn_like(z)
            z = (z - 0.5 * step_size * grad_z + (step_size ** 0.5) * noise).detach()

            past_burnin = step >= int(self.cfg.burnin_steps)
            at_thinning = (step - int(self.cfg.burnin_steps)) % int(self.cfg.thinning) == 0
            if past_burnin and at_thinning:
                samples.append(z.clone())

        return {
            "z_samples": torch.stack(samples, dim=1),
        }
