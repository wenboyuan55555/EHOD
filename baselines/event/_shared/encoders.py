from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualMLP(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)

class STIDEncoder(nn.Module):
    """STID encoder matching the official T3 implementation's identities."""

    def __init__(self, history: int, nodes: int, hidden: int = 128) -> None:
        super().__init__()
        self.history = history
        self.nodes = nodes
        self.node = nn.Embedding(nodes, 32)
        self.hour = nn.Embedding(24, 32)
        self.weekday = nn.Embedding(7, 32)
        self.traffic = nn.Linear(history * 2, 32)
        self.encoder = nn.Sequential(
            ResidualMLP(hidden), ResidualMLP(hidden), ResidualMLP(hidden)
        )

    def forward(
        self, history: torch.Tensor, hour: torch.Tensor, weekday: torch.Tensor
    ) -> torch.Tensor:
        batch, _, nodes, _ = history.shape
        traffic = self.traffic(history.transpose(1, 2).reshape(batch, nodes, -1))
        node = self.node.weight.unsqueeze(0).expand(batch, -1, -1)
        tod = self.hour(hour).unsqueeze(1).expand(-1, nodes, -1)
        dow = self.weekday(weekday).unsqueeze(1).expand(-1, nodes, -1)
        return self.encoder(torch.cat((traffic, node, tod, dow), dim=-1))
