from __future__ import annotations

import math

import torch
from torch import nn


class PrototypeBranch(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, prototypes: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden)
        self.micro_query = nn.Linear(hidden, hidden, bias=False)
        self.micro_key = nn.Linear(hidden, hidden, bias=False)
        self.prototype_assignment = nn.Parameter(torch.randn(hidden, prototypes) * 0.02)
        self.prototype_graph = nn.Parameter(torch.eye(prototypes))
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.LayerNorm(hidden)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value)
        query = self.micro_query(hidden)
        key = self.micro_key(hidden)
        global_key = key.mean(dim=1, keepdim=True)
        micro = hidden + torch.sigmoid(
            (query * global_key).sum(dim=-1, keepdim=True) / math.sqrt(hidden.shape[-1])
        ) * global_key
        assignment = torch.softmax(micro @ self.prototype_assignment, dim=-1)
        normalizer = assignment.sum(dim=1).clamp_min(1.0e-6)
        prototype = torch.einsum("bsp,bsh->bph", assignment, micro)
        prototype = prototype / normalizer.unsqueeze(-1)
        graph = torch.softmax(self.prototype_graph, dim=-1)
        prototype = torch.einsum("pq,bqh->bph", graph, prototype)
        mapped = torch.einsum("bsp,bph->bsh", assignment, prototype)
        return self.fusion(torch.cat([micro, mapped], dim=-1))


class STProOD(nn.Module):
    """Paper-faithful scalable reproduction of STPro for 263-zone OD."""

    def __init__(
        self,
        history_hours: int,
        nodes: int = 263,
        hidden: int = 24,
        prototypes: int = 23,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_hours = history_hours
        self.nodes = nodes
        self.origin = PrototypeBranch(nodes, hidden, prototypes, dropout)
        self.destination = PrototypeBranch(nodes, hidden, prototypes, dropout)
        self.o_query = nn.Linear(hidden, hidden, bias=False)
        self.o_key = nn.Linear(hidden, hidden, bias=False)
        self.o_value = nn.Linear(hidden, hidden, bias=False)
        self.d_query = nn.Linear(hidden, hidden, bias=False)
        self.d_key = nn.Linear(hidden, hidden, bias=False)
        self.d_value = nn.Linear(hidden, hidden, bias=False)
        self.temporal_score = nn.Linear(2 * hidden + 4, 1)
        self.decoder = nn.Linear(2 * hidden, nodes)

    def forward(self, history: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = history.shape
        origin_input = history.reshape(batch, steps * nodes, nodes)
        destination_input = history.transpose(2, 3).reshape(batch, steps * nodes, nodes)
        origin = self.origin(origin_input).reshape(batch, steps, nodes, -1)
        destination = self.destination(destination_input).reshape(
            batch, steps, nodes, -1
        )
        temporal_fused = torch.cat([origin, destination], dim=-1)
        calendar_expanded = calendar[:, None, None, :].expand(
            batch, steps, nodes, calendar.shape[-1]
        )
        temporal_weight = torch.softmax(
            self.temporal_score(
                torch.cat([temporal_fused, calendar_expanded], dim=-1)
            ),
            dim=1,
        )
        origin = (origin * temporal_weight).sum(dim=1)
        destination = (destination * temporal_weight).sum(dim=1)
        scale = math.sqrt(origin.shape[-1])
        attention_od = torch.softmax(
            self.o_query(origin) @ self.d_key(destination).transpose(1, 2) / scale,
            dim=-1,
        )
        attention_do = torch.softmax(
            self.d_query(destination) @ self.o_key(origin).transpose(1, 2) / scale,
            dim=-1,
        )
        origin = origin + attention_od @ self.d_value(destination)
        destination = destination + attention_do @ self.o_value(origin)
        node = torch.cat([origin, destination], dim=-1)
        return self.decoder(node)
