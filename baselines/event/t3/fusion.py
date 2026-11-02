from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import EVENT_FEATURE_DIM

class T3DualPredictorEventAdapter(nn.Module):
    """T3-inspired projected concat with traffic and multimodal predictors."""

    def __init__(
        self, nodes: int = N_NODES, node_dim: int = 12, hidden: int = 96
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.event_projection = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.traffic_projection = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.multimodal_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.traffic_activation = nn.Linear(hidden, 1)
        self.traffic_magnitude = nn.Linear(hidden, 1)
        self.multimodal_activation = nn.Linear(hidden, 1)
        self.multimodal_magnitude = nn.Linear(hidden, 1)
        for layer in (
            self.traffic_activation,
            self.traffic_magnitude,
            self.multimodal_activation,
            self.multimodal_magnitude,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        event_features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_logit: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        event = self.event_projection(event_features)
        traffic = self.traffic_projection(
            torch.cat(
                [
                    base_logit[:, None],
                    base_log_magnitude[:, None],
                    self.origin(origin),
                    self.destination(destination),
                ],
                dim=-1,
            )
        )
        multimodal = self.multimodal_projection(torch.cat([event, traffic], dim=-1))
        activation_delta = (
            self.traffic_activation(traffic).squeeze(-1)
            + self.multimodal_activation(multimodal).squeeze(-1)
        )
        magnitude_delta = (
            self.traffic_magnitude(traffic).squeeze(-1)
            + self.multimodal_magnitude(multimodal).squeeze(-1)
        )
        return (
            base_logit + 2.5 * torch.tanh(activation_delta),
            torch.clamp(
                base_log_magnitude + 1.5 * torch.tanh(magnitude_delta), min=0.0
            ),
        )
