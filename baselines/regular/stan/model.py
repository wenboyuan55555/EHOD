from __future__ import annotations

import torch
from torch import nn


class TimePrompter(nn.Module):
    def __init__(self, calendar_dim: int, hidden: int, edge_dim: int):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(calendar_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.origin_scale = nn.Linear(hidden, edge_dim)
        self.destination_scale = nn.Linear(hidden, edge_dim)
        self.ffn_scale = nn.Linear(hidden, edge_dim)

    def forward(self, calendar: torch.Tensor) -> tuple:
        core = self.core(calendar)
        return (
            1.0 + 0.25 * torch.tanh(self.origin_scale(core)),
            1.0 + 0.25 * torch.tanh(self.destination_scale(core)),
            1.0 + 0.25 * torch.tanh(self.ffn_scale(core)),
        )


class AxialEdgeBlock(nn.Module):
    def __init__(self, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        self.origin_attention = nn.MultiheadAttention(
            edge_dim, heads, dropout=dropout, batch_first=True
        )
        self.destination_attention = nn.MultiheadAttention(
            edge_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(edge_dim)
        self.norm2 = nn.LayerNorm(edge_dim)
        self.norm3 = nn.LayerNorm(edge_dim)
        self.ffn = nn.Sequential(
            nn.Linear(edge_dim, 2 * edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * edge_dim, edge_dim),
        )

    def forward(self, edge: torch.Tensor, scales: tuple) -> torch.Tensor:
        batch, nodes, _, channels = edge.shape
        origin_input = edge.reshape(batch * nodes, nodes, channels)
        origin, _ = self.origin_attention(origin_input, origin_input, origin_input)
        origin = origin.reshape(batch, nodes, nodes, channels)
        edge = self.norm1(edge + origin * scales[0][:, None, None, :])
        destination_input = edge.transpose(1, 2).reshape(batch * nodes, nodes, channels)
        destination, _ = self.destination_attention(
            destination_input, destination_input, destination_input
        )
        destination = destination.reshape(batch, nodes, nodes, channels).transpose(1, 2)
        edge = self.norm2(edge + destination * scales[1][:, None, None, :])
        return self.norm3(edge + self.ffn(edge) * scales[2][:, None, None, :])


class STANOD(nn.Module):
    """Scalable strict-window reproduction of STAN's edge Transformer."""

    def __init__(
        self,
        history_hours: int,
        static_dim: int = 4,
        edge_dim: int = 16,
        global_dim: int = 16,
        heads: int = 4,
        blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_projector = nn.Linear(history_hours, edge_dim // 2)
        self.static_projector = nn.Linear(static_dim, edge_dim // 2)
        self.prompter = TimePrompter(4, global_dim, edge_dim)
        self.blocks = nn.ModuleList(
            [AxialEdgeBlock(edge_dim, heads, dropout) for _ in range(blocks)]
        )
        self.output = nn.Linear(edge_dim, 1)

    def forward(
        self, history: torch.Tensor, calendar: torch.Tensor, static_edge: torch.Tensor
    ) -> torch.Tensor:
        history_edge = self.history_projector(history.permute(0, 2, 3, 1))
        static = self.static_projector(static_edge).unsqueeze(0).expand(
            history.shape[0], -1, -1, -1
        )
        edge = torch.cat([history_edge, static], dim=-1)
        scales = self.prompter(calendar)
        for block in self.blocks:
            edge = block(edge, scales)
        return self.output(edge).squeeze(-1)

