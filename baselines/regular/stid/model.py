from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class STIDOD(nn.Module):
    """STID-style history MLP with origin, hour and weekday identities."""

    def __init__(
        self,
        num_nodes: int,
        input_length: int,
        horizon: int,
        hidden_dim: int,
        node_embedding_dim: int,
        time_embedding_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.horizon = int(horizon)
        self.history_projection = nn.Linear(input_length * num_nodes, hidden_dim)
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        self.hour_embedding = nn.Embedding(24, time_embedding_dim)
        self.weekday_embedding = nn.Embedding(7, time_embedding_dim)
        total_width = hidden_dim + node_embedding_dim + 2 * time_embedding_dim
        self.input_fusion = nn.Linear(total_width, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLP(hidden_dim, dropout) for _ in range(layers)]
        )
        self.head = nn.Linear(hidden_dim, num_nodes)

    def forward(self, history: torch.Tensor, target_starts: torch.Tensor) -> torch.Tensor:
        batch, destination, origin, length = history.shape
        node_history = history.permute(0, 2, 3, 1).reshape(
            batch, origin, length * destination
        )
        history_embedding = self.history_projection(node_history)
        node_ids = torch.arange(origin, device=history.device)
        spatial = self.node_embedding(node_ids)[None, :, :].expand(batch, -1, -1)

        outputs = []
        starts = target_starts.long()
        for step in range(self.horizon):
            target_index = starts + step
            hour = torch.remainder(target_index, 24)
            weekday = torch.remainder(torch.div(target_index, 24, rounding_mode="floor") + 6, 7)
            hour_embedding = self.hour_embedding(hour)[:, None, :].expand(-1, origin, -1)
            weekday_embedding = self.weekday_embedding(weekday)[:, None, :].expand(
                -1, origin, -1
            )
            x = torch.cat(
                [history_embedding, spatial, hour_embedding, weekday_embedding], dim=-1
            )
            x = self.input_fusion(x)
            for block in self.blocks:
                x = block(x)
            outputs.append(self.head(x))
        return torch.stack(outputs, dim=1)


def build_model(
    config: Dict[str, Any], num_nodes: int, supports: Sequence[torch.Tensor]
) -> nn.Module:
    model = config["model"]
    return STIDOD(
        num_nodes=num_nodes,
        input_length=int(config["data"]["input_length"]),
        horizon=int(config["data"]["horizon"]),
        hidden_dim=int(model["hidden_dim"]),
        node_embedding_dim=int(model["node_embedding_dim"]),
        time_embedding_dim=int(model["time_embedding_dim"]),
        layers=int(model["layers"]),
        dropout=float(model["dropout"]),
    )

