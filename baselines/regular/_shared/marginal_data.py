"""Leakage-safe shared data contract for the seven added baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_HOURS, N_NODES, TRAIN_END, TRAIN_START, load_od


CACHE_ROOT = EXPERIMENT_ROOT / "runs" / "seven_baselines" / "shared_cache"
PROTOCOL_VERSION = "eadt-seven-baselines-v1"


def _existing_original_cache() -> Path:
    return EXPERIMENT_ROOT / "cache" / "original_event_baselines"


def _cache_complete(root: Path) -> bool:
    return all(
        (root / name).exists()
        for name in ("marginal_raw.npy", "marginal_normalized.npy", "marginal_scaler.npz")
    )


def build_marginal_cache(overwrite: bool = False) -> Path:
    """Build only from OD observations whose scaler fit is restricted to train."""

    if not overwrite and _cache_complete(CACHE_ROOT):
        return CACHE_ROOT
    # NYC already has a byte-compatible audited marginal cache.  Reuse it to
    # avoid duplicating tens of MiB on the nearly full experiment filesystem.
    existing = _existing_original_cache()
    if not overwrite and _cache_complete(existing):
        return existing

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = CACHE_ROOT / "marginal_raw.npy"
    normalized_path = CACHE_ROOT / "marginal_normalized.npy"
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(N_HOURS, N_NODES, 2)
    )
    od = load_od()
    for begin in range(0, N_HOURS, 168):
        end = min(begin + 168, N_HOURS)
        block = np.asarray(od[begin:end], dtype=np.float32)
        raw[begin:end, :, 0] = block.sum(axis=2)
        raw[begin:end, :, 1] = block.sum(axis=1)
    raw.flush()

    train_log = np.log1p(np.asarray(raw[TRAIN_START:TRAIN_END], dtype=np.float32))
    mean = train_log.mean(axis=(0, 1)).astype(np.float32)
    std = train_log.std(axis=(0, 1)).clip(min=1.0e-6).astype(np.float32)
    normalized = np.lib.format.open_memmap(
        normalized_path, mode="w+", dtype=np.float32, shape=raw.shape
    )
    for begin in range(0, N_HOURS, 168):
        end = min(begin + 168, N_HOURS)
        normalized[begin:end] = (
            np.log1p(np.asarray(raw[begin:end], dtype=np.float32)) - mean
        ) / std
    normalized.flush()
    np.savez(CACHE_ROOT / "marginal_scaler.npz", mean=mean, std=std)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "city": ACTIVE_CITY,
        "shape": [N_HOURS, N_NODES, 2],
        "channels": ["origin_outflow", "destination_inflow"],
        "transform": "per-channel log1p z-score",
        "scaler_fit_target_hours": [TRAIN_START, TRAIN_END],
        "validation_or_test_used_for_scaler": False,
    }
    (CACHE_ROOT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return CACHE_ROOT


def load_marginal_cache() -> Dict[str, np.ndarray]:
    root = build_marginal_cache()
    scaler = np.load(root / "marginal_scaler.npz")
    return {
        "root": root,
        "raw": np.load(root / "marginal_raw.npy", mmap_mode="r"),
        "normalized": np.load(root / "marginal_normalized.npy", mmap_mode="r"),
        "mean": np.asarray(scaler["mean"], dtype=np.float32),
        "std": np.asarray(scaler["std"], dtype=np.float32),
    }


def inverse_marginal(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return np.maximum(np.expm1(array * std + mean), 0.0).astype(np.float32)


__all__ = [
    "CACHE_ROOT",
    "PROTOCOL_VERSION",
    "build_marginal_cache",
    "inverse_marginal",
    "load_marginal_cache",
]
