"""OD-matrix adapter retaining STIDGCN's interleaved dynamic-GCN tree."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


from ehod.common.upstream import load_snapshot
PAPER_CODE = Path(__file__).resolve().parent / "upstream"
_upstream = load_snapshot(PAPER_CODE, "model.py", "_ehod_upstream_stidgcn")
GLU, IDGCN_Tree = _upstream.GLU, _upstream.IDGCN_Tree


class ODTemporalEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.hour = nn.Parameter(torch.empty(24, channels))
        self.weekday = nn.Parameter(torch.empty(7, channels))
        nn.init.xavier_uniform_(self.hour)
        nn.init.xavier_uniform_(self.weekday)

    def forward(self, target_starts: torch.Tensor, nodes: int, steps: int = 12):
        offsets = torch.arange(-steps, 0, device=target_starts.device)
        history = target_starts.long()[:, None] + offsets[None, :]
        hour = torch.remainder(history, 24)
        weekday = torch.remainder(
            torch.div(history, 24, rounding_mode="floor") + 6, 7
        )
        embedding = self.hour[hour] + self.weekday[weekday]
        return embedding.permute(0, 2, 1)[:, :, None, :].expand(
            -1, -1, nodes, -1
        )


class STIDGCNOD(nn.Module):
    """Treat origins as graph nodes and destination flows as node features."""

    def __init__(
        self,
        device: torch.device,
        num_nodes: int = 263,
        channels: int = 32,
        dropout: float = 0.1,
        native_steps: int = 12,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.native_steps = int(native_steps)
        if self.native_steps != 12:
            raise ValueError("The released STIDGCN tree requires 12 native steps")
        self.temporal_embedding = ODTemporalEmbedding(channels)
        self.start_conv = nn.Conv2d(num_nodes, channels, kernel_size=(1, 1))
        self.tree = IDGCN_Tree(
            device=device,
            channels=channels * 2,
            diffusion_step=1,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        self.glu = GLU(channels * 2, dropout)
        self.regression_layer = nn.Conv2d(
            channels * 2, num_nodes, kernel_size=(1, self.native_steps)
        )

    def pad_short_history(
        self, history: torch.Tensor, temporal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad with the earliest visible step without reading older hours."""
        steps = history.shape[-1]
        if steps > self.native_steps:
            raise ValueError(
                f"Expected at most {self.native_steps} history steps, got {steps}"
            )
        if steps == self.native_steps:
            return history, temporal
        padding = self.native_steps - steps
        history_prefix = history[..., :1].expand(*history.shape[:-1], padding)
        temporal_prefix = temporal[..., :1].expand(*temporal.shape[:-1], padding)
        return (
            torch.cat([history_prefix, history], dim=-1),
            torch.cat([temporal_prefix, temporal], dim=-1),
        )

    def encode_features(
        self, history: torch.Tensor, target_starts: torch.Tensor
    ) -> torch.Tensor:
        # history: [batch, destination-feature, origin-node, visible steps]
        temporal = self.temporal_embedding(
            target_starts, nodes=history.shape[2], steps=history.shape[3]
        )
        history, temporal = self.pad_short_history(history, temporal)
        x = torch.cat([self.start_conv(history), temporal], dim=1)
        x = self.tree(x)
        return self.glu(x) + x

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.regression_layer(F.relu(features)).squeeze(-1)
        # [batch, destination, origin] -> [batch, origin, destination]
        return prediction.permute(0, 2, 1).contiguous()

    def forward(self, history: torch.Tensor, target_starts: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(history, target_starts))
