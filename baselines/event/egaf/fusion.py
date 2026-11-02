from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import N_EVENT_TYPES, N_VENUES, TIME_DIM, event_feature_tokens

class LegacyAttentionEventAdapter(nn.Module):
    """Comparable version of the original post-Hurdle attention adapter."""

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 12,
        hidden: int = 96,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(N_EVENT_TYPES + N_VENUES + TIME_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
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
        query = self.query_encoder(
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
        tokens = self.token_encoder(event_feature_tokens(event_features))
        attended, _ = self.attention(
            query[:, None, :], tokens, tokens, need_weights=False
        )
        hidden = self.fusion(torch.cat([query, attended[:, 0]], dim=-1))
        return (
            base_logit
            + 3.0 * torch.tanh(self.activation_delta(hidden).squeeze(-1)),
            torch.clamp(
                base_log_magnitude
                + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
                min=0.0,
            ),
        )
