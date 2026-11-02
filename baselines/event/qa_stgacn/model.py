from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class QASTGACN(nn.Module):
    """Query-driven spatiotemporal graph attention with exogenous events."""

    def __init__(self, nodes: int, event_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(2, hidden, kernel_size=3, padding=1)
        self.node = nn.Embedding(nodes, hidden)
        self.event = nn.Linear(6, hidden)
        self.query = nn.Linear(hidden * 3, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.scale = hidden ** -0.5

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        size, steps, nodes, _ = batch["history"].shape
        sequence = batch["history"].permute(0, 2, 3, 1).reshape(size * nodes, 2, steps)
        temporal = self.temporal(sequence).mean(-1).reshape(size, nodes, -1)
        identity = self.node.weight.unsqueeze(0).expand(size, -1, -1)
        event = self.event(batch["event_node"])
        query = self.query(torch.cat((temporal, identity, event), dim=-1))
        key, value = self.key(temporal + identity), self.value(temporal)
        score = torch.einsum("bnd,bmd->bnm", query, key) * self.scale
        score = score.masked_fill(~(batch["adjacency"] > 0).unsqueeze(0), -1.0e4)
        graph = torch.einsum("bnm,bmd->bnd", torch.softmax(score, -1), value)
        return self.head(torch.cat((query, graph), dim=-1))
