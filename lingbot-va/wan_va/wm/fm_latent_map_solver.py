"""Flow-matching latent MAP solver for LingBot-VA watermark inversion.

Given observed actions y_obs, recovers the initial noise z that best explains
the observation under the flow-matching action decoder.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FMLatentMAPConfig:
    num_iters: int = 100
    lr: float = 0.1
    obs_sigma: float = 1e-4
    prior_weight: float = 1.0
    optimizer: str = "sgd"
    max_grad_norm: float = 1.0   # clip z-gradient norm; obs_loss carries a ~1e6 (1/obs_sigma^2)
                                 # factor so on off-manifold inversions the raw grad explodes
                                 # (~1e5) and z -> NaN in one step. Clipping bounds it at the source.


class FMLatentMAPSolver:
    """MAP estimator for the action latent noise z.

    Solves: z_map = argmin_z E(z)
    where E(z) = obs_mismatch(decode(z), y_obs) + prior_weight * ||z||^2
    """

    def __init__(self, decode_fn, obs_op, cfg: FMLatentMAPConfig):
        """
        Args:
            decode_fn: callable(z) -> a_pred [B, C, F, H, 1]
                The differentiable action denoising function.
            obs_op: ChannelObservation instance.
            cfg: solver config.
        """
        self.decode_fn = decode_fn
        self.obs_op = obs_op
        self.cfg = cfg

    def solve(
        self,
        y_obs: torch.Tensor,
        z_init: torch.Tensor | None = None,
        z_shape: tuple | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Run MAP optimization.

        Args:
            y_obs: observed action channels [B, C_obs, F, H, 1]
            z_init: optional initial guess for z [B, C_total, F, H, 1]
            z_shape: shape of z if z_init is None

        Returns:
            dict with keys: z_map, a_map, final_obs_mse
        """
        device = y_obs.device
        dtype = y_obs.dtype

        if z_init is None:
            if z_shape is None:
                raise ValueError("Must provide z_init or z_shape")
            z = torch.randn(*z_shape, dtype=dtype, device=device)
        else:
            z = z_init.detach().clone().to(device=device, dtype=dtype)

        # obs_loss carries a 1/obs_sigma^2 factor (~1e6); when the residual cannot be driven to
        # zero (e.g. inverting an OFF-MANIFOLD distilled student's actions through the base),
        # SGD's z = z - lr*grad diverges to NaN. Adam normalizes by the running gradient
        # magnitude -> stable in that regime; we also keep the last FINITE z as a guard so one
        # bad step can never poison the result.
        max_gn = float(self.cfg.max_grad_norm)

        def _clip(g):
            # clip the z-gradient to max_gn by norm; zero it if non-finite (inf/NaN) so the step
            # is skipped rather than poisoning the optimizer state.
            if not torch.isfinite(g).all():
                return torch.zeros_like(g)
            gn = torch.linalg.vector_norm(g)
            if max_gn > 0 and gn > max_gn:
                g = g * (max_gn / (gn + 1e-12))
            return g

        z_best = z.detach().clone()
        if self.cfg.optimizer == "adam":
            z = z.detach().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=float(self.cfg.lr))
            for it in range(int(self.cfg.num_iters)):
                opt.zero_grad()
                a_pred = self.decode_fn(z)
                pred_obs = self.obs_op.apply(a_pred)
                obs_loss = 0.5 * (((pred_obs - y_obs) / float(self.cfg.obs_sigma)) ** 2).mean()
                prior_loss = 0.5 * float(self.cfg.prior_weight) * z.square().mean()
                loss = obs_loss + prior_loss
                if not torch.isfinite(loss):
                    break
                loss.backward()
                z.grad = _clip(z.grad)            # <-- gradient clipping (prevents the explosion)
                opt.step()
                with torch.no_grad():
                    if torch.isfinite(z).all():
                        z_best = z.detach().clone()
                    else:
                        break
            z = z_best
        else:
            for it in range(int(self.cfg.num_iters)):
                z_opt = z.detach().requires_grad_(True)
                a_pred = self.decode_fn(z_opt)
                pred_obs = self.obs_op.apply(a_pred)
                obs_loss = 0.5 * (((pred_obs - y_obs) / float(self.cfg.obs_sigma)) ** 2).mean()
                prior_loss = 0.5 * float(self.cfg.prior_weight) * z_opt.square().mean()
                loss = obs_loss + prior_loss
                if not torch.isfinite(loss):
                    break
                grad_z = _clip(torch.autograd.grad(loss, z_opt)[0])   # <-- gradient clipping
                with torch.no_grad():
                    z = z - float(self.cfg.lr) * grad_z
                    if torch.isfinite(z).all():
                        z_best = z.detach().clone()
                    else:
                        z = z_best
                        break
            z = z_best

        with torch.no_grad():
            a_final = self.decode_fn(z)
            final_obs_mse = torch.mean(torch.square(self.obs_op.apply(a_final) - y_obs)).item()

        return {
            "z_map": z.detach(),
            "a_map": a_final.detach(),
            "final_obs_mse": float(final_obs_mse),
        }


def run_map_restarts(
    decode_fn,
    obs_op,
    *,
    y_obs: torch.Tensor,
    z_shape: tuple,
    cfg: FMLatentMAPConfig,
    num_starts: int = 4,
    z_seed: torch.Tensor | None = None,
    rng_seed: int = 0,
) -> dict:
    """Multi-start MAP: run solver from several initializations, pick best.

    Returns:
        dict with keys: z_map, a_map, final_obs_mse, all_energies, best_index
    """
    solver = FMLatentMAPSolver(decode_fn, obs_op, cfg)
    device = y_obs.device
    dtype = y_obs.dtype

    def energy_fn(z: torch.Tensor) -> float:
        with torch.no_grad():
            a_pred = decode_fn(z)
            pred_obs = obs_op.apply(a_pred)
            obs_loss = 0.5 * (((pred_obs - y_obs) / float(cfg.obs_sigma)) ** 2).mean()
            prior_loss = 0.5 * float(cfg.prior_weight) * z.square().mean()
            return float((obs_loss + prior_loss).item())

    rng = torch.Generator(device='cpu').manual_seed(rng_seed)
    inits: list[torch.Tensor | None] = []

    if z_seed is not None:
        inits.append(z_seed.detach().clone())
        for _ in range(num_starts - 1):
            perturbation = torch.randn(*z_shape, generator=rng, dtype=dtype).to(device)
            inits.append(z_seed.detach().clone() + 0.5 * perturbation)
    else:
        for _ in range(num_starts):
            inits.append(torch.randn(*z_shape, generator=rng, dtype=dtype).to(device))

    results = []
    energies = []
    for z_init in inits:
        out = solver.solve(y_obs=y_obs, z_init=z_init)
        results.append(out)
        energies.append(energy_fn(out["z_map"]))

    best_idx = int(min(range(len(energies)), key=lambda i: energies[i]))
    best = results[best_idx]
    best["all_energies"] = energies
    best["best_index"] = best_idx
    return best
