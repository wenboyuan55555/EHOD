from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import N_EVENT_TYPES, N_VENUES, TIME_DIM, event_feature_tokens

class FuseCrossAttentionEventAdapter(nn.Module):
    """FUSE-Traffic-inspired traffic-query/event-key cross attention."""

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 12,
        hidden: int = 96,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.query_projection = nn.Linear(2 + 2 * node_dim, hidden)
        self.event_projection = nn.Linear(
            N_EVENT_TYPES + N_VENUES + TIME_DIM, hidden
        )
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
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
        query = self.query_projection(
            torch.cat(
                [
                    base_logit[:, None],
                    base_log_magnitude[:, None],
                    self.origin(origin),
                    self.destination(destination),
                ],
                dim=-1,
            )
        )[:, None, :]
        event = self.event_projection(event_feature_tokens(event_features))
        attended, _ = self.attention(query, event, event, need_weights=False)
        hidden = self.norm1(query + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))[:, 0]
        return (
            base_logit
            + 2.5 * torch.tanh(self.activation_delta(hidden).squeeze(-1)),
            torch.clamp(
                base_log_magnitude
                + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
                min=0.0,
            ),
        )
