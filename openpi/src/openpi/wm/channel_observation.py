from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelObservation:
    channel_idx: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    obs_sigma: float = 1e-4

    def apply(self, a_raw: torch.Tensor) -> torch.Tensor:
        return a_raw[:, :, self.channel_idx]

    def loss(self, a_raw: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        pred = self.apply(a_raw)
        return 0.5 * (((pred - y_obs) / float(self.obs_sigma)) ** 2).mean()

    def overwrite(self, a_raw: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        a_completed = a_raw.clone()
        a_completed[:, :, self.channel_idx] = y_obs
        return a_completed
