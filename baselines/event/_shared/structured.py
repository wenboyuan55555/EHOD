from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from ehod.data.od import ACTIVE_CITY, N_NODES
from baselines.event._shared.features import N_EVENT_TYPES, N_VENUES, STRUCTURED_DIM, VENUE_ROLE_DIM, TIME_DIM, EVENT_FEATURE_DIM, event_feature_tokens

class StructuredEventAdapter(nn.Module):
    """Backbone-agnostic joint activation/magnitude event calibrator."""

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 12,
        hidden: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.structured_gate = nn.Parameter(torch.zeros(STRUCTURED_DIM))
        scalar_dim = EVENT_FEATURE_DIM + 2
        self.encoder = nn.Sequential(
            nn.Linear(scalar_dim + 2 * node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
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
        structured = event_features[:, :STRUCTURED_DIM]
        gated = structured * torch.sigmoid(self.structured_gate)[None, :]
        event_features = torch.cat(
            [gated, event_features[:, STRUCTURED_DIM:]], dim=-1
        )
        hidden = self.encoder(
            torch.cat(
                [
                    event_features,
                    base_logit[:, None],
                    base_log_magnitude[:, None],
                    self.origin(origin),
                    self.destination(destination),
                ],
                dim=-1,
            )
        )
        logit = base_logit + 2.0 * torch.tanh(
            self.activation_delta(hidden).squeeze(-1)
        )
        magnitude_log = torch.clamp(
            base_log_magnitude
            + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
            min=0.0,
        )
        return logit, magnitude_log
