from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from ehod.data.od import N_NODES

from baselines.regular.odmixer.model import ODMixerOD
from baselines.event.event_matrix.model import ODPF
from baselines.regular.stan.model import STANOD
from baselines.regular.stpro.model import STProOD
from baselines.regular._shared.discrete import create_stid, create_stidgcn, legacy_history


MODEL_NAMES = ("stidgcn", "stid", "stpro", "odmixer", "odpf", "stan")


def create_model(
    name: str, history_hours: int, device: torch.device = None
) -> nn.Module:
    if name == "stidgcn":
        if device is None:
            raise ValueError("STIDGCN construction requires the selected device")
        return create_stidgcn(history_hours, device, nodes=N_NODES)
    if name == "stid":
        return create_stid(history_hours, nodes=N_NODES)
    if name == "stpro":
        return STProOD(
            history_hours=history_hours,
            nodes=N_NODES,
            hidden=16,
            prototypes=23,
            dropout=0.1,
        )
    if name == "odmixer":
        return ODMixerOD(
            history_hours=history_hours,
            nodes=N_NODES,
            hidden=16,
            layers=3,
            dropout=0.1,
        )
    if name == "odpf":
        return ODPF(
            history_hours=history_hours,
            nodes=N_NODES,
            hidden=8,
            graph_hidden=16,
        )
    if name == "stan":
        return STANOD(
            history_hours=history_hours,
            edge_dim=16,
            global_dim=16,
            heads=4,
            blocks=2,
            dropout=0.1,
        )
    raise KeyError(name)


def static_tensors(static: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for key, value in static.items()
        if key in ("distance", "same_borough", "adjacency", "affinity")
    }
    result["stan_edge"] = torch.stack(
        [
            result["distance"],
            result["same_borough"],
            result["adjacency"],
            result["affinity"],
        ],
        dim=-1,
    )
    return result


def forward_model(
    name: str,
    model: nn.Module,
    history: torch.Tensor,
    calendar: torch.Tensor,
    static: Dict[str, torch.Tensor],
    event_matrix: torch.Tensor = None,
    target_starts: torch.Tensor = None,
) -> torch.Tensor:
    if name == "stidgcn":
        if target_starts is None:
            raise ValueError("STIDGCN requires exact target indices")
        return model(legacy_history(history), target_starts)
    if name == "stid":
        if target_starts is None:
            raise ValueError("STID requires exact target indices")
        return model(legacy_history(history), target_starts)[:, 0]
    if name in ("stpro", "odmixer"):
        return model(history, calendar)
    if name == "stan":
        return model(history, calendar, static["stan_edge"])
    if name == "odpf":
        if event_matrix is None:
            event_matrix = history.new_zeros(
                (history.shape[0], history.shape[2], history.shape[3], 3)
            )
        return model(history, calendar, static["adjacency"], event_matrix)
    raise KeyError(name)
