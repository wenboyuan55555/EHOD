from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class CDGON(nn.Module):
    """One-step physics-informed graph ODE adaptation of CDGON."""

    def __init__(self, nodes: int, event_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.node = nn.Embedding(nodes, hidden)
        self.state = nn.Linear(2, hidden)
        self.normal = nn.Linear(2, hidden)
        self.event = nn.Linear(6, hidden)
        self.edge_message = nn.Linear(hidden, hidden)
        self.decay = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.residual = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_raw = batch["history"][:, -1]
        normal_raw = batch["seasonal_marginal"]
        size = current_raw.shape[0]
        identity = self.node.weight.unsqueeze(0).expand(size, -1, -1)
        current = self.state(current_raw)
        normal = self.normal(normal_raw)
        event = self.event(batch["event_node"])
        message = torch.einsum("nm,bmd->bnd", batch["adjacency"], current)
        message = self.edge_message(message)
        physics_rate = F.softplus(self.decay(torch.cat((current, normal, event), dim=-1)))
        physical = physics_rate * (normal_raw - current_raw)
        learned = self.residual(torch.cat((current, message, identity + event), dim=-1))
        severity = batch["event_node"][..., 0:3].sum(dim=-1, keepdim=True).clamp(0, 1)
        derivative = physical + severity * learned
        return current_raw + derivative
