from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMoE(nn.Module):
    """Periodic/trend/instant traffic state with spatial, semantic and general experts."""

    def __init__(self, history: int, nodes: int, event_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.history = history
        self.node = nn.Embedding(nodes, 16)
        self.time = nn.Linear(4, 16)
        self.state = nn.Linear(2 * 3 + 32, hidden)
        self.spatial_expert = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU())
        self.semantic_projection = nn.Linear(event_dim, hidden)
        self.semantic_expert = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU())
        self.general_expert = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.router = nn.Linear(hidden, 3)
        self.head = nn.Linear(hidden, 2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history"]
        instant = history[:, -1]
        trend = history[:, -1] - history.mean(dim=1)
        spectrum = torch.fft.rfft(history, dim=1).abs()
        periodic = spectrum[:, 1 : min(4, spectrum.shape[1])].mean(dim=1)
        if periodic.shape[-1] != 2:
            periodic = instant.new_zeros(instant.shape)
        calendar = torch.stack(
            (
                torch.sin(2 * torch.pi * batch["hour"].float() / 24),
                torch.cos(2 * torch.pi * batch["hour"].float() / 24),
                torch.sin(2 * torch.pi * batch["weekday"].float() / 7),
                torch.cos(2 * torch.pi * batch["weekday"].float() / 7),
            ),
            dim=-1,
        )
        time = self.time(calendar).unsqueeze(1).expand(-1, instant.shape[1], -1)
        node = self.node.weight.unsqueeze(0).expand(instant.shape[0], -1, -1)
        state = self.state(torch.cat((instant, trend, periodic, node, time), dim=-1))
        spatial_context = torch.einsum("nm,bmd->bnd", batch["adjacency"], state)
        spatial = self.spatial_expert(torch.cat((state, spatial_context), dim=-1))
        semantic_context = self.semantic_projection(batch["event_global"])
        semantic_context = semantic_context.unsqueeze(1).expand(-1, state.shape[1], -1)
        semantic = self.semantic_expert(torch.cat((state, semantic_context), dim=-1))
        general = self.general_expert(state)
        weight = torch.softmax(self.router(state), dim=-1)
        mixture = (
            weight[..., 0:1] * spatial
            + weight[..., 1:2] * semantic
            + weight[..., 2:3] * general
        )
        return self.head(mixture)
