from __future__ import annotations

import torch
from baselines.regular.stidgcn.model import STIDGCNOD
from baselines.regular.stid.model import STIDOD


def create_stidgcn(
    history_hours: int, device: torch.device, nodes: int = 263
) -> STIDGCNOD:
    del history_hours
    return STIDGCNOD(
        device=device,
        num_nodes=nodes,
        channels=32,
        dropout=0.1,
        native_steps=12,
    )


def create_stid(history_hours: int, nodes: int = 263) -> STIDOD:
    return STIDOD(
        num_nodes=nodes,
        input_length=history_hours,
        horizon=1,
        hidden_dim=128,
        node_embedding_dim=32,
        time_embedding_dim=16,
        layers=2,
        dropout=0.1,
    )


def legacy_history(history: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,O,D] to the released adapters' [B,D,O,H]."""
    return history.permute(0, 3, 2, 1).contiguous()


__all__ = ["create_stid", "create_stidgcn", "legacy_history"]
