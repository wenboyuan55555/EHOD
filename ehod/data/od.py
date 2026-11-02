from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from ehod.common.paths import ACTIVE_CITY, BUNDLE_ROOT, CHICAGO_DATA_ROOT, CHICAGO_EXPERIMENT_ROOT, EXPERIMENT_ROOT, NYC_CORE_ROOT, CITY_EXPERIMENT_ROOT

PROJECT_SOURCE_ROOT = EXPERIMENT_ROOT
EXPERIMENT_ROOT = CITY_EXPERIMENT_ROOT
PROJECT_ROOT = BUNDLE_ROOT
DATA_ROOT = (
    CHICAGO_DATA_ROOT / "core_od" if ACTIVE_CITY == "chicago" else NYC_CORE_ROOT
)
OD_PATH = DATA_ROOT / "od_matrix.npy"
NORMALIZED_PATH = DATA_ROOT / "normalized/od_log1p_zscore.npy"
SCALER_PATH = DATA_ROOT / "normalized/scaler.npz"
ZONE_PATH = DATA_ROOT / "zone_index.csv"
STATIC_CACHE = EXPERIMENT_ROOT / "cache/static_features.npz"

N_HOURS = 8760
N_NODES = 77 if ACTIVE_CITY == "chicago" else 263
TRAIN_START = 12
TRAIN_END = 6720
VAL_END = 7728
TEST_END = 8760
HISTORY_HOURS = (1, 3, 6, 12)
INVALID_TARGET_HOURS = (
    frozenset((1682, 8757, 8758, 8759))
    if ACTIVE_CITY == "chicago"
    else frozenset()
)


def split_bounds(split: str) -> Tuple[int, int]:
    if split == "train":
        return TRAIN_START, TRAIN_END
    if split == "validation":
        return TRAIN_END, VAL_END
    if split == "test":
        return VAL_END, TEST_END
    raise KeyError(split)


def target_hours(split: str) -> np.ndarray:
    start, end = split_bounds(split)
    hours = np.arange(start, end, dtype=np.int32)
    if INVALID_TARGET_HOURS:
        hours = hours[~np.isin(hours, np.asarray(sorted(INVALID_TARGET_HOURS)))]
    return hours


def iter_hour_batches(
    split: str,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    limit: Optional[int] = None,
) -> Iterator[np.ndarray]:
    hours = target_hours(split)
    if shuffle:
        np.random.default_rng(seed).shuffle(hours)
    if limit is not None:
        hours = hours[: int(limit)]
    for begin in range(0, len(hours), batch_size):
        part = hours[begin : begin + batch_size]
        if len(part):
            yield part


def load_od() -> np.ndarray:
    return np.load(OD_PATH, mmap_mode="r")


def load_normalized() -> np.ndarray:
    return np.load(NORMALIZED_PATH, mmap_mode="r")


def load_scaler() -> Tuple[float, float]:
    payload = np.load(SCALER_PATH)
    return float(payload["mean"]), float(payload["std"])


def history_indices(starts: np.ndarray, history_hours: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int32)
    if history_hours not in HISTORY_HOURS:
        raise ValueError("history_hours must be one of %s" % (HISTORY_HOURS,))
    return starts[:, None] + np.arange(-history_hours, 0, dtype=np.int32)[None, :]


def history_batch(
    source: np.ndarray, starts: np.ndarray, history_hours: int
) -> np.ndarray:
    return np.asarray(source[history_indices(starts, history_hours)], dtype=np.float32)


def calendar_features(starts: np.ndarray) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    hour = starts % 24
    weekday = (starts // 24) % 7
    return np.stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
        ],
        axis=1,
    ).astype(np.float32)


def inverse_log_standardize(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.expm1(value.float() * std + mean).clamp_min(0.0)


def _haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude.astype(np.float64))
    lon = np.radians(longitude.astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(
        dlon / 2.0
    ) ** 2
    return (6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(
        np.float32
    )


def build_static_cache(overwrite: bool = False, top_k: int = 12) -> Path:
    if STATIC_CACHE.exists() and not overwrite:
        return STATIC_CACHE
    STATIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    od = load_od()
    nonzero = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    positive_sum = np.zeros_like(nonzero)
    total_sum = np.zeros_like(nonzero)
    seasonal_nonzero = np.zeros((168, N_NODES, N_NODES), dtype=np.uint16)
    seasonal_sum = np.zeros((168, N_NODES, N_NODES), dtype=np.float64)
    seasonal_count = np.zeros(168, dtype=np.uint16)
    for begin in range(TRAIN_START, TRAIN_END, 168):
        end = min(begin + 168, TRAIN_END)
        hours = np.arange(begin, end, dtype=np.int32)
        if INVALID_TARGET_HOURS:
            hours = hours[~np.isin(hours, np.asarray(sorted(INVALID_TARGET_HOURS)))]
        if not len(hours):
            continue
        block = np.asarray(od[hours], dtype=np.float32)
        nonzero += (block > 0).sum(axis=0)
        positive_sum += block.sum(axis=0)
        total_sum += block.sum(axis=0)
        for local, hour in enumerate(hours):
            slot = hour % 168
            seasonal_nonzero[slot] += (block[local] > 0).astype(np.uint16)
            seasonal_sum[slot] += block[local]
            seasonal_count[slot] += 1
    train_hours = float(len(target_hours("train")))
    edge_nonzero_frequency = (nonzero / train_hours).astype(np.float32)
    edge_positive_mean = (positive_sum / np.maximum(nonzero, 1.0)).astype(np.float32)
    origin_total = total_sum.sum(axis=1).astype(np.float32)
    destination_total = total_sum.sum(axis=0).astype(np.float32)
    origin_feature = np.log1p(origin_total)
    destination_feature = np.log1p(destination_total)
    origin_feature = (origin_feature - origin_feature.mean()) / max(
        float(origin_feature.std()), 1.0e-6
    )
    destination_feature = (destination_feature - destination_feature.mean()) / max(
        float(destination_feature.std()), 1.0e-6
    )
    seasonal_probability = seasonal_nonzero.astype(np.float32) / np.maximum(
        seasonal_count[:, None, None].astype(np.float32), 1.0
    )
    seasonal_mean = seasonal_sum.astype(np.float32) / np.maximum(
        seasonal_count[:, None, None].astype(np.float32), 1.0
    )
    seasonal_positive_mean = seasonal_sum.astype(np.float32) / np.maximum(
        seasonal_nonzero.astype(np.float32), 1.0
    )

    zones = np.genfromtxt(
        ZONE_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    latitude = np.asarray(zones["centroid_lat"], dtype=np.float32)
    longitude = np.asarray(zones["centroid_lon"], dtype=np.float32)
    if "borough" in zones.dtype.names:
        region = np.asarray(zones["borough"]).astype(str)
    else:
        # Chicago Community Areas have no borough-level equivalent.  Retain
        # the same static channel as an exact same-area relation rather than
        # inventing a post-hoc geographic grouping.
        region = np.asarray(zones["community_name"]).astype(str)
    distance = _haversine_matrix(latitude, longitude)
    same_borough = (region[:, None] == region[None, :]).astype(np.float32)
    order = np.argsort(distance, axis=1)
    adjacency = np.zeros_like(distance, dtype=np.float32)
    for node in range(N_NODES):
        neighbors = order[node, 1 : top_k + 1]
        sigma = max(float(distance[node, neighbors[-1]]), 1.0)
        adjacency[node, neighbors] = np.exp(-distance[node, neighbors] / sigma)
    adjacency += np.eye(N_NODES, dtype=np.float32)
    adjacency /= adjacency.sum(axis=1, keepdims=True).clip(min=1.0e-6)
    affinity = np.log1p(total_sum.astype(np.float32))
    affinity /= max(float(affinity.max()), 1.0)
    distance_normalized = np.log1p(distance) / max(float(np.log1p(distance).max()), 1.0)
    np.savez_compressed(
        STATIC_CACHE,
        edge_nonzero_frequency=edge_nonzero_frequency,
        edge_positive_mean=edge_positive_mean,
        seasonal_probability=seasonal_probability.astype(np.float16),
        seasonal_mean=seasonal_mean.astype(np.float16),
        seasonal_positive_mean=seasonal_positive_mean.astype(np.float16),
        origin_feature=origin_feature.astype(np.float32),
        destination_feature=destination_feature.astype(np.float32),
        distance=distance_normalized.astype(np.float32),
        same_borough=same_borough,
        adjacency=adjacency,
        affinity=affinity.astype(np.float32),
    )
    metadata = {
        "city": ACTIVE_CITY,
        "train_target_hours": [TRAIN_START, TRAIN_END],
        "invalid_target_hours_excluded": sorted(INVALID_TARGET_HOURS),
        "top_k": top_k,
        "uses_test_data": False,
    }
    (STATIC_CACHE.with_suffix(".json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return STATIC_CACHE


def load_static() -> Dict[str, np.ndarray]:
    path = build_static_cache()
    payload = np.load(path, mmap_mode="r")
    return {key: payload[key] for key in payload.files}
