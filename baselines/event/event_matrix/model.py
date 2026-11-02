from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, value: torch.Tensor, state: tuple) -> tuple:
        hidden, cell = state
        gates = self.gates(torch.cat([value, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ODPF(nn.Module):
    """OD-PF reproduction adapted from metro tap-ins to taxi origin totals."""

    def __init__(
        self,
        history_hours: int,
        nodes: int = 263,
        hidden: int = 8,
        graph_hidden: int = 16,
        event_channels: int = 3,
    ) -> None:
        super().__init__()
        self.history_hours = history_hours
        self.nodes = nodes
        self.cell = ConvLSTMCell(1, hidden)
        self.temporal_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.hod_decoder = nn.Conv2d(hidden, 1, kernel_size=1)
        self.origin_temporal = nn.GRU(1, graph_hidden, batch_first=True)
        self.origin_output = nn.Linear(graph_hidden, 1)
        self.external = nn.Sequential(
            nn.Linear(4, graph_hidden), nn.SiLU(), nn.Linear(graph_hidden, nodes)
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_channels, graph_hidden),
            nn.SiLU(),
            nn.Linear(graph_hidden, 1),
        )
        self.origin_destination_gate = nn.Parameter(torch.zeros(nodes))
        self.hod_weight = nn.Parameter(torch.tensor(1.0))
        self.feature_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        history: torch.Tensor,
        calendar: torch.Tensor,
        adjacency: torch.Tensor,
        event_matrix: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = history.shape
        hidden = history.new_zeros(
            (batch, self.cell.hidden_channels, nodes, nodes)
        )
        cell = torch.zeros_like(hidden)
        hidden_sequence = []
        for step in range(steps):
            hidden, cell = self.cell(history[:, step : step + 1], (hidden, cell))
            hidden_sequence.append(hidden)
        sequence = torch.stack(hidden_sequence, dim=1)
        scores = torch.einsum("bthij,h->btij", sequence, self.temporal_query)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)
        hod = self.hod_decoder((sequence * weights).sum(dim=1)).squeeze(1)

        origin_total = history.sum(dim=3).permute(0, 2, 1).unsqueeze(-1)
        origin_encoded, _ = self.origin_temporal(origin_total.reshape(batch * nodes, steps, 1))
        origin = self.origin_output(origin_encoded[:, -1]).reshape(batch, nodes)
        origin = torch.einsum("ij,bj->bi", adjacency, origin)
        destination_gate = torch.softmax(self.origin_destination_gate, dim=0)
        origin_edge = origin[:, :, None] * destination_gate[None, None, :]
        time_edge = self.external(calendar)[:, :, None] * destination_gate[None, None, :]
        event_edge = self.event_encoder(event_matrix).squeeze(-1)
        return self.hod_weight * hod + self.feature_weight * (
            origin_edge + time_edge + event_edge
        )

