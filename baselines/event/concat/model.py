from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class FUSEConcatD2STGNN(nn.Module):
    """FUSE-Traffic concatenation ablation with the same D2STGNN bridge."""

    def __init__(self, nodes: int, event_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.temporal = nn.GRU(2, hidden, batch_first=True)
        self.node_u = nn.Parameter(torch.empty(nodes, hidden))
        self.node_d = nn.Parameter(torch.empty(nodes, hidden))
        self.diffusion = nn.Linear(hidden, hidden)
        self.inherent = nn.Linear(hidden, hidden)
        self.decouple_gate = nn.Linear(hidden * 3, hidden)
        self.event_projection = nn.Linear(event_dim, hidden)
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 2))
        nn.init.xavier_uniform_(self.node_u)
        nn.init.xavier_uniform_(self.node_d)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history"]
        size, steps, nodes, _ = history.shape
        sequence = history.transpose(1, 2).reshape(size * nodes, steps, 2)
        inherent = self.temporal(sequence)[1][-1].reshape(size, nodes, -1)
        adaptive = torch.softmax(F.relu(self.node_u @ self.node_d.T), dim=-1)
        diffused = torch.einsum("nm,bmd->bnd", adaptive, inherent)
        static = torch.einsum("nm,bmd->bnd", batch["adjacency"], inherent)
        gate = torch.sigmoid(
            self.decouple_gate(torch.cat((inherent, diffused, static), dim=-1))
        )
        traffic = gate * self.diffusion(diffused + static) + (1.0 - gate) * self.inherent(
            inherent
        )
        tokens = self.event_projection(batch["event_tokens"])
        mask = batch["event_token_mask"].to(tokens.dtype).unsqueeze(-1)
        event = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        event = event.unsqueeze(1).expand(-1, nodes, -1)
        return self.head(self.concat_fusion(torch.cat((traffic, event), dim=-1)))
