from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import N_EVENT_TYPES, N_VENUES, TIME_DIM, EVENT_FEATURE_DIM, event_feature_tokens

class QAQueryEventAdapter(nn.Module):
    """QA-STGACN-inspired query attention, convolution, and gated fusion."""

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 16,
        hidden: int = 96,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.state = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        token_dim = N_EVENT_TYPES + N_VENUES + TIME_DIM
        self.query_token = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.query_attention = nn.MultiheadAttention(
            hidden, heads, dropout=0.1, batch_first=True
        )
        self.query_convolution = nn.Conv1d(
            hidden, hidden, kernel_size=3, padding=1
        )
        self.pattern_transform = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.fusion_gate = nn.Linear(hidden * 2, hidden)
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
        state = self.state(
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
        tokens = self.query_token(event_feature_tokens(event_features))
        attended, _ = self.query_attention(
            state[:, None, :], tokens, tokens, need_weights=False
        )
        convolved = self.query_convolution(tokens.transpose(1, 2)).mean(dim=-1)
        pattern = self.pattern_transform(event_features)
        candidate = self.candidate(
            torch.cat([attended[:, 0], convolved, pattern], dim=-1)
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat([state, candidate], dim=-1)))
        hidden = state + gate * candidate
        return (
            base_logit
            + 2.5 * torch.tanh(self.activation_delta(hidden).squeeze(-1)),
            torch.clamp(
                base_log_magnitude
                + 1.5 * torch.tanh(self.magnitude_delta(hidden).squeeze(-1)),
                min=0.0,
            ),
        )
