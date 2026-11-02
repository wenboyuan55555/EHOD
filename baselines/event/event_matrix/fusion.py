from __future__ import annotations
from typing import Tuple
import torch
from torch import nn

class MatrixEventAdapter(nn.Module):
    """OD-PF-style three-channel event matrix with joint Hurdle calibration."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.activation_delta = nn.Linear(hidden, 1)
        self.magnitude_delta = nn.Linear(hidden, 1)
        nn.init.zeros_(self.activation_delta.weight)
        nn.init.zeros_(self.activation_delta.bias)
        nn.init.zeros_(self.magnitude_delta.weight)
        nn.init.zeros_(self.magnitude_delta.bias)

    def forward(
        self,
        event_features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_logit: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del origin, destination
        hidden = self.network(
            torch.cat(
                [event_features, base_logit[:, None], base_log_magnitude[:, None]],
                dim=-1,
            )
        )
        return (
            base_logit
            + 2.0 * torch.tanh(self.activation_delta(hidden).squeeze(-1)),
            torch.clamp(
                base_log_magnitude
                + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
                min=0.0,
            ),
        )
