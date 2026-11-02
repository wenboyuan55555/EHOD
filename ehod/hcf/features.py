from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_HOURS, N_NODES, load_od
from ehod.common.paths import CHICAGO_DATA_ROOT, CONTINUOUS_CACHE_ROOT


CONTINUOUS_CACHE = (
    CHICAGO_DATA_ROOT / "continuous_cache"
    if ACTIVE_CITY == "chicago"
    else CONTINUOUS_CACHE_ROOT
)
NODE_TOTAL_CACHE = EXPERIMENT_ROOT / "cache" / "node_hour_totals.npz"


def build_node_total_cache() -> Path:
    if NODE_TOTAL_CACHE.exists():
        return NODE_TOTAL_CACHE
    od = load_od()
    origin = np.empty((N_HOURS, N_NODES), dtype=np.float32)
    destination = np.empty_like(origin)
    for begin in range(0, N_HOURS, 24):
        end = min(begin + 24, N_HOURS)
        block = np.asarray(od[begin:end], dtype=np.float32)
        origin[begin:end] = block.sum(axis=2)
        destination[begin:end] = block.sum(axis=1)
    NODE_TOTAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(NODE_TOTAL_CACHE.parent), suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            origin=origin,
            destination=destination,
        )
        os.replace(str(temporary), str(NODE_TOTAL_CACHE))
    finally:
        if temporary.exists():
            temporary.unlink()
    return NODE_TOTAL_CACHE


class ContinuousFeatureStore:
    def __init__(self) -> None:
        self.od = load_od()
        self.hour_weight = np.load(
            CONTINUOUS_CACHE / "continuous_hour_exp_weights.npy", mmap_mode="r"
        )
        self.origin_last = np.load(
            CONTINUOUS_CACHE / "continuous_hour_last_pickup_second.npy",
            mmap_mode="r",
        )
        self.destination_last = np.load(
            CONTINUOUS_CACHE / "continuous_hour_destination_last_pickup_second.npy",
            mmap_mode="r",
        )
        self.quarter_edge = None
        self.quarter_origin = None
        self.quarter_destination = None
        quarter_edge_path = CONTINUOUS_CACHE / "continuous_hour_quarter_counts.npy"
        quarter_origin_path = CONTINUOUS_CACHE / "continuous_hour_quarter_origin.npy"
        quarter_destination_path = (
            CONTINUOUS_CACHE / "continuous_hour_quarter_destination.npy"
        )
        if all(
            path.exists()
            for path in (
                quarter_edge_path,
                quarter_origin_path,
                quarter_destination_path,
            )
        ):
            self.quarter_edge = np.load(quarter_edge_path, mmap_mode="r")
            self.quarter_origin = np.load(quarter_origin_path, mmap_mode="r")
            self.quarter_destination = np.load(
                quarter_destination_path, mmap_mode="r"
            )
        self.bin15_weight = None
        self.bin15_origin_last = None
        self.bin15_destination_last = None
        bin15_weight_path = CONTINUOUS_CACHE / "continuous_hour_bin15_exp_weights.npy"
        bin15_origin_path = (
            CONTINUOUS_CACHE / "continuous_hour_bin15_last_pickup_second.npy"
        )
        bin15_destination_path = (
            CONTINUOUS_CACHE
            / "continuous_hour_bin15_destination_last_pickup_second.npy"
        )
        if all(
            path.exists()
            for path in (bin15_weight_path, bin15_origin_path, bin15_destination_path)
        ):
            self.bin15_weight = np.load(bin15_weight_path, mmap_mode="r")
            self.bin15_origin_last = np.load(bin15_origin_path, mmap_mode="r")
            self.bin15_destination_last = np.load(
                bin15_destination_path, mmap_mode="r"
            )
        totals = np.load(build_node_total_cache(), mmap_mode="r")
        self.origin_total = totals["origin"]
        self.destination_total = totals["destination"]

    def selected(
        self,
        starts: np.ndarray,
        origin: np.ndarray,
        destination: np.ndarray,
        history_hours: int,
        continuous_mode: str = "exact",
    ) -> Dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int32)
        origin = np.asarray(origin, dtype=np.int64)
        destination = np.asarray(destination, dtype=np.int64)
        batch, edges = origin.shape
        offsets = np.arange(-history_hours, 0, dtype=np.int32)
        edge_history = np.empty((batch, edges, history_hours), dtype=np.float32)
        origin_history = np.empty_like(edge_history)
        destination_history = np.empty_like(edge_history)
        intensity = np.zeros((batch, edges), dtype=np.float32)
        origin_recency = np.zeros_like(intensity)
        destination_recency = np.zeros_like(intensity)
        target = np.empty((batch, edges), dtype=np.float32)
        quarter_edge = None
        quarter_origin = None
        quarter_destination = None
        decay = np.exp(-np.arange(history_hours - 1, -1, -1)).astype(np.float32)
        if continuous_mode == "quantized_15min":
            if self.bin15_weight is None:
                raise RuntimeError(
                    "Missing true 15-minute cache; run "
                    "python -m ehod.data.prepare_quarter_hours first"
                )
            hour_weight = self.bin15_weight
            origin_last_source = self.bin15_origin_last
            destination_last_source = self.bin15_destination_last
        else:
            hour_weight = self.hour_weight
            origin_last_source = self.origin_last
            destination_last_source = self.destination_last
        if continuous_mode == "hierarchical":
            if (
                self.quarter_edge is None
                or self.quarter_origin is None
                or self.quarter_destination is None
            ):
                raise RuntimeError(
                    "Missing quarter-hour cache; run "
                    "python -m ehod.data.prepare_quarter_hours first"
                )
            previous = starts[:, None] - 1
            quarter_edge = np.asarray(
                self.quarter_edge[previous, origin, destination], dtype=np.float32
            )
            quarter_origin = np.asarray(
                self.quarter_origin[previous, origin], dtype=np.float32
            )
            quarter_destination = np.asarray(
                self.quarter_destination[previous, destination], dtype=np.float32
            )
        for b, start in enumerate(starts):
            hours = start + offsets
            o = origin[b]
            d = destination[b]
            for k, hour in enumerate(hours):
                edge_history[b, :, k] = self.od[hour, o, d]
                origin_history[b, :, k] = self.origin_total[hour, o]
                destination_history[b, :, k] = self.destination_total[hour, d]
                intensity[b] += hour_weight[hour, o, d] * decay[k]
            o_last = np.max(np.asarray(origin_last_source[hours]), axis=0)[o]
            d_last = np.max(np.asarray(destination_last_source[hours]), axis=0)[d]
            o_active = o_last >= 0
            d_active = d_last >= 0
            origin_recency[b, o_active] = np.exp(
                (o_last[o_active] - start * 3600.0) / 3600.0
            ).astype(np.float32)
            destination_recency[b, d_active] = np.exp(
                (d_last[d_active] - start * 3600.0) / 3600.0
            ).astype(np.float32)
            target[b] = self.od[start, o, d]
        return {
            "edge_history": edge_history,
            "origin_history": origin_history,
            "destination_history": destination_history,
            "continuous_intensity": intensity,
            "origin_recency": origin_recency,
            "destination_recency": destination_recency,
            "quarter_edge": quarter_edge,
            "quarter_origin": quarter_origin,
            "quarter_destination": quarter_destination,
            "target": target,
        }


def full_edge_indices(batch: int) -> Tuple[np.ndarray, np.ndarray]:
    origin = np.arange(N_NODES, dtype=np.int64).repeat(N_NODES)
    destination = np.tile(np.arange(N_NODES, dtype=np.int64), N_NODES)
    return (
        np.broadcast_to(origin[None, :], (batch, len(origin))),
        np.broadcast_to(destination[None, :], (batch, len(destination))),
    )


__all__ = ["ContinuousFeatureStore", "full_edge_indices"]
