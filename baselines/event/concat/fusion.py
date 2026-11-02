from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from baselines.event._shared.features import EVENT_FEATURE_DIM

class ConcatEventAdapter(nn.Module):
    """Unstructured feature-concatenation comparison."""

    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM + 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.activation = nn.Linear(hidden, 1)
        self.magnitude = nn.Linear(hidden, 1)

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
        return self.activation(hidden).squeeze(-1), torch.clamp(
            self.magnitude(hidden).squeeze(-1), min=0.0
        )
