from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ODInteraction(nn.Module):
    def __init__(self, nodes: int, hidden: int, dropout: float):
        super().__init__()
        self.channel = FeedForward(hidden, 2 * hidden, hidden, dropout)
        self.origin = FeedForward(nodes, 2 * hidden, nodes, dropout)
        self.destination = FeedForward(nodes, 2 * hidden, nodes, dropout)
        self.channel_norm = nn.LayerNorm(hidden)
        self.spatial_norm = nn.LayerNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.channel_norm(value + self.channel(value))
        origin = self.origin(value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        destination = self.destination(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.spatial_norm(value + origin + destination)


class TrendInteraction(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.current_gate = FeedForward(hidden, hidden, hidden, dropout)
        self.previous_gate = FeedForward(hidden, hidden, hidden, dropout)
        self.current_value = FeedForward(hidden, hidden, hidden, dropout)
        self.previous_value = FeedForward(hidden, hidden, hidden, dropout)

    def forward(
        self, current: torch.Tensor, previous: torch.Tensor
    ) -> tuple:
        current_new = current + self.current_value(current) * torch.sigmoid(
            self.current_gate(0.5 * (current + previous))
        )
        previous_new = previous + self.previous_value(previous) * torch.sigmoid(
            self.previous_gate(0.5 * (current + previous))
        )
        return current_new, previous_new


class ODMixerOD(nn.Module):
    """Strict-window adaptation of the released ODMixer."""

    def __init__(
        self,
        history_hours: int,
        nodes: int = 263,
        hidden: int = 16,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_hours = history_hours
        self.embedding = nn.Linear(history_hours, hidden)
        self.interactions = nn.ModuleList(
            [ODInteraction(nodes, hidden, dropout) for _ in range(layers)]
        )
        self.trends = nn.ModuleList(
            [TrendInteraction(hidden, dropout) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 4)),
            nn.PReLU(),
            nn.Linear(max(hidden // 2, 4), 1),
        )

    def forward(self, history: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        del calendar
        current_raw = history.permute(0, 2, 3, 1)
        if self.history_hours == 1:
            previous_raw = current_raw
        else:
            shifted = torch.cat([history[:, :1], history[:, :-1]], dim=1)
            previous_raw = shifted.permute(0, 2, 3, 1)
        current = self.embedding(current_raw)
        previous = self.embedding(previous_raw)
        for interaction, trend in zip(self.interactions, self.trends):
            current = interaction(current)
            previous = interaction(previous)
            current, previous = trend(current, previous)
        return self.output(current).squeeze(-1)

