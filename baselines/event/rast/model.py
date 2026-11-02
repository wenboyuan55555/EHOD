from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAST(nn.Module):
    """Decoupled temporal/spatial query with dual retrieval and attention fusion."""

    def __init__(self, history: int, nodes: int, event_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.temporal = nn.Linear(history * 2, 64)
        self.spatial = nn.Embedding(nodes, 64)
        self.query = nn.Linear(128, 64)
        self.retrieval = nn.Linear(2, 64)
        self.event = nn.Linear(event_dim, 64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        size, _, nodes, _ = batch["history"].shape
        temporal = self.temporal(batch["history"].transpose(1, 2).reshape(size, nodes, -1))
        spatial = self.spatial.weight.unsqueeze(0).expand(size, -1, -1)
        query = self.query(torch.cat((temporal, spatial), dim=-1))
        retrieved = self.retrieval(batch["retrieved"])
        event = self.event(batch["event_global"]).unsqueeze(1).expand(-1, nodes, -1)
        documents = torch.stack((retrieved, event), dim=2).reshape(size, nodes * 2, -1)
        fused, _ = self.attention(query, documents, documents, need_weights=False)
        return self.head(torch.cat((query, fused), dim=-1))
