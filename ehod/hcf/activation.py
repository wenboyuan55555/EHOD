from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn


GATE_FEATURE_DIM = 9


class StrictActivationGate(nn.Module):
    """Shared activation model using only the declared history window."""

    def __init__(
        self, nodes: int = 263, embedding_dim: int = 8, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.origin_embedding = nn.Embedding(nodes, embedding_dim)
        self.destination_embedding = nn.Embedding(nodes, embedding_dim)
        self.calendar_encoder = nn.Sequential(
            nn.Linear(4, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        input_dim = GATE_FEATURE_DIM + 3 * embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        if calendar.ndim == 2:
            calendar = calendar[:, None, :].expand(
                features.shape[0], features.shape[1], calendar.shape[-1]
            )
        context = self.calendar_encoder(calendar)
        combined = torch.cat(
            [
                features,
                self.origin_embedding(origin),
                self.destination_embedding(destination),
                context,
            ],
            dim=-1,
        )
        return self.network(combined).squeeze(-1)


def selected_gate_features(
    history: torch.Tensor,
    starts: torch.Tensor,
    origin: torch.Tensor,
    destination: torch.Tensor,
    static: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build strict features for selected edges.

    history: [B,H,N,N], origin/destination: [B,E].
    """
    batch, steps = history.shape[:2]
    batch_index = torch.arange(batch, device=history.device)[:, None]
    values = history[
        batch_index[:, :, None],
        torch.arange(steps, device=history.device)[None, None, :],
        origin[:, :, None],
        destination[:, :, None],
    ]
    log_values = torch.log1p(values)
    last = log_values[:, :, -1]
    mean = log_values.mean(dim=-1)
    maximum = log_values.max(dim=-1).values
    nonzero_fraction = (values > 0).float().mean(dim=-1)
    last_matrix = history[:, -1]
    origin_total = torch.log1p(last_matrix.sum(dim=2))[batch_index, origin]
    destination_total = torch.log1p(last_matrix.sum(dim=1))[batch_index, destination]
    edge_frequency = static["edge_nonzero_frequency"][origin, destination]
    positive_mean = torch.log1p(static["edge_positive_mean"][origin, destination])
    slot = (starts % 168).long()
    seasonal = static["seasonal_probability"][
        slot[:, None], origin, destination
    ].float()
    return torch.stack(
        [
            last,
            mean,
            maximum,
            nonzero_fraction,
            origin_total,
            destination_total,
            edge_frequency,
            positive_mean,
            seasonal,
        ],
        dim=-1,
    )


@torch.no_grad()
def predict_gate_full(
    model: StrictActivationGate,
    history: torch.Tensor,
    starts: torch.Tensor,
    calendar: torch.Tensor,
    static: Dict[str, torch.Tensor],
    edge_chunk: int = 32768,
) -> np.ndarray:
    model.eval()
    batch, _, nodes, _ = history.shape
    output = np.empty((batch, nodes * nodes), dtype=np.float32)
    all_origin = torch.arange(nodes, device=history.device).repeat_interleave(nodes)
    all_destination = torch.arange(nodes, device=history.device).repeat(nodes)
    for begin in range(0, nodes * nodes, edge_chunk):
        end = min(begin + edge_chunk, nodes * nodes)
        origin = all_origin[begin:end][None, :].expand(batch, -1)
        destination = all_destination[begin:end][None, :].expand(batch, -1)
        features = selected_gate_features(
            history, starts, origin, destination, static
        )
        logits = model(features, origin, destination, calendar)
        output[:, begin:end] = torch.sigmoid(logits).float().cpu().numpy()
    return output.reshape(batch, nodes, nodes)

