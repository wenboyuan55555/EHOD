from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.event._shared.encoders import STIDEncoder

class T3STID(nn.Module):
    """Official T3 pattern: frozen STID + text head + traffic-only head mean."""

    def __init__(self, history: int, nodes: int, event_dim: int) -> None:
        super().__init__()
        self.encoder = STIDEncoder(history, nodes)
        self.event_projection = nn.Linear(event_dim, 256)
        self.multimodal_head = nn.Linear(128 + 256, 2)
        self.traffic_head = nn.Linear(128, 2)

    def freeze_traffic_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def traffic_only(
        self, history: torch.Tensor, hour: torch.Tensor, weekday: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(history, hour, weekday)
        return self.traffic_head(hidden)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(batch["history"], batch["hour"], batch["weekday"])
        event = self.event_projection(batch["event_global"])
        event = event.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        multimodal = self.multimodal_head(torch.cat((hidden, event), dim=-1))
        traffic = self.traffic_head(hidden)
        return 0.5 * (multimodal + traffic)
