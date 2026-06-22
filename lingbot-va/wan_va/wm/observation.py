"""Channel observation operator for partial-observation watermark detection."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelObservation:
    """Selects active action channels from the full action tensor.

    LingBot-VA action tensor shape: [B, C_total, F, H, 1]
    After apply: [B, len(channel_idx), F, H, 1]
    """

    channel_idx: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    def apply(self, a_raw: torch.Tensor) -> torch.Tensor:
        return a_raw[:, list(self.channel_idx)]

    def loss(self, a_raw: torch.Tensor, y_obs: torch.Tensor, obs_sigma: float = 1e-4) -> torch.Tensor:
        pred = self.apply(a_raw)
        return 0.5 * (((pred - y_obs) / obs_sigma) ** 2).mean()
