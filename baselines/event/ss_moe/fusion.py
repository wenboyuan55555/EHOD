from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import EVENT_FEATURE_DIM

class SemanticSpatialMoEEventAdapter(nn.Module):
    """SS-MoE-inspired semantic, spatial, and general event experts."""

    def __init__(
        self, nodes: int = N_NODES, node_dim: int = 16, hidden: int = 96
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.semantic_expert = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.spatial_expert = nn.Sequential(
            nn.Linear(2 * node_dim + 2, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.general_expert = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM + 2 * node_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.router = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM + 2 * node_dim + 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
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
        spatial_input = torch.cat(
            [
                base_logit[:, None],
                base_log_magnitude[:, None],
                self.origin(origin),
                self.destination(destination),
            ],
            dim=-1,
        )
        joint = torch.cat([event_features, spatial_input], dim=-1)
        experts = torch.stack(
            [
                self.semantic_expert(event_features),
                self.spatial_expert(spatial_input),
                self.general_expert(joint),
            ],
            dim=1,
        )
        weights = torch.softmax(self.router(joint), dim=-1)
        hidden = torch.sum(experts * weights[:, :, None], dim=1)
        return (
            base_logit
            + 2.5 * torch.tanh(self.activation_delta(hidden).squeeze(-1)),
            torch.clamp(
                base_log_magnitude
                + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
                min=0.0,
            ),
        )
