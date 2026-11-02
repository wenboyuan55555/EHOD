from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Tuple
from ehod.hcf.features import ContinuousFeatureStore
from ehod.data.od import N_NODES

def torch_static(payload, device) -> Dict[str, torch.Tensor]:
    names = (
        "edge_nonzero_frequency",
        "edge_positive_mean",
        "seasonal_probability",
        "seasonal_positive_mean",
        "distance",
        "same_borough",
        "affinity",
    )
    return {
        key: torch.from_numpy(np.asarray(payload[key], dtype=np.float32)).to(device)
        for key in names
    }

def sample_edges(
    store: ContinuousFeatureStore,
    starts: np.ndarray,
    uniform_edges: int,
    positive_edges: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = len(starts)
    total = uniform_edges + positive_edges
    origin = np.empty((batch, total), dtype=np.int64)
    destination = np.empty_like(origin)
    for b, start in enumerate(starts):
        uniform_flat = rng.integers(0, N_NODES * N_NODES, size=uniform_edges)
        target = np.asarray(store.od[start]).reshape(-1)
        positive_flat = np.flatnonzero(target > 0)
        if len(positive_flat) == 0:
            chosen = rng.integers(0, N_NODES * N_NODES, size=positive_edges)
        else:
            chosen = rng.choice(
                positive_flat, size=positive_edges, replace=len(positive_flat) < positive_edges
            )
        flat = np.concatenate([uniform_flat, chosen])
        origin[b] = flat // N_NODES
        destination[b] = flat % N_NODES
    return origin, destination

def to_torch(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)
