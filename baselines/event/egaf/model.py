from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.event._shared.encoders import ResidualMLP

class EGAFNet(nn.Module):
    """Event-guided temporal and spatial structured attention."""

    def __init__(self, nodes: int, event_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.traffic = nn.Linear(2, hidden)
        self.event_node = nn.Linear(6, hidden)
        self.temporal_score = nn.Linear(hidden, 1)
        self.spatial_q = nn.Linear(hidden, hidden)
        self.spatial_k = nn.Linear(hidden, hidden)
        self.spatial_v = nn.Linear(hidden, hidden)
        self.global_event = nn.Linear(event_dim, hidden)
        self.head = nn.Sequential(ResidualMLP(hidden), nn.Linear(hidden, 2))
        self.scale = hidden ** -0.5

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = self.traffic(batch["history"])
        event = self.event_node(batch["event_node"])
        score = self.temporal_score(torch.tanh(sequence + event.unsqueeze(1)))
        temporal = (torch.softmax(score, dim=1) * sequence).sum(dim=1)
        enriched = temporal + event + self.global_event(batch["event_global"]).unsqueeze(1)
        q, k, v = self.spatial_q(enriched), self.spatial_k(enriched), self.spatial_v(enriched)
        attention = torch.einsum("bnd,bmd->bnm", q, k) * self.scale
        graph_mask = batch["adjacency"] > 0
        attention = attention.masked_fill(~graph_mask.unsqueeze(0), -1.0e4)
        spatial = torch.einsum("bnm,bmd->bnd", torch.softmax(attention, -1), v)
        return self.head(enriched + spatial)
