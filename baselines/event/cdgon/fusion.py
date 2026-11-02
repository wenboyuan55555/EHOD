from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from ehod.data.od import N_NODES
from baselines.event._shared.features import STRUCTURED_DIM, EVENT_FEATURE_DIM

class CDGONEventAdapter(nn.Module):
    """CDGON-inspired one-step recovery-to-normal event dynamics for HCF."""

    def __init__(
        self, nodes: int = N_NODES, node_dim: int = 16, hidden: int = 96
    ) -> None:
        super().__init__()
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.state = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.event = nn.Sequential(
            nn.Linear(EVENT_FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.edge_interaction = nn.Sequential(
            nn.Linear(2 * node_dim, hidden), nn.Tanh()
        )
        self.event_perturbation = nn.Linear(hidden * 3, 2)
        self.recovery_rate = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 2)
        )
        self.learned_derivative = nn.Linear(hidden * 3, 2)
        nn.init.zeros_(self.event_perturbation.weight)
        nn.init.zeros_(self.event_perturbation.bias)
        nn.init.zeros_(self.learned_derivative.weight)
        nn.init.zeros_(self.learned_derivative.bias)

    def forward(
        self,
        event_features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_logit: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        origin_embedding = self.origin(origin)
        destination_embedding = self.destination(destination)
        normal = torch.stack([base_logit, base_log_magnitude], dim=-1)
        state = self.state(
            torch.cat(
                [normal, origin_embedding, destination_embedding], dim=-1
            )
        )
        event = self.event(event_features)
        interaction = self.edge_interaction(
            torch.cat([origin_embedding, destination_embedding], dim=-1)
        )
        joint = torch.cat([state, event, interaction], dim=-1)
        severity = event_features[:, :STRUCTURED_DIM].abs().amax(
            dim=-1, keepdim=True
        ).clamp(0.0, 1.0)
        perturbation = severity * 1.5 * torch.tanh(self.event_perturbation(joint))
        current = normal + perturbation
        recovery = torch.nn.functional.softplus(self.recovery_rate(joint))
        learned = severity * 1.5 * torch.tanh(self.learned_derivative(joint))
        derivative = recovery * (normal - current) + learned
        updated = current + 0.5 * derivative
        return updated[:, 0], torch.clamp(updated[:, 1], min=0.0)
