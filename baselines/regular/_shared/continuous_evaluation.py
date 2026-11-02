"""Shared strict-window hurdle evaluation for continuous OD baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, TRAIN_END, VAL_END, load_od, target_hours
from ehod.evaluation.metrics import StreamingMetrics


# Share threshold selection without importing architecture-specific trainers.
THRESHOLDS = np.asarray(
    [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ],
    dtype=np.float32,
)


def update_threshold_bins(probability, magnitude, actual, bins):
    p = probability.reshape(-1)
    m = np.maximum(magnitude.reshape(-1), 0.0).astype(np.float64, copy=False)
    y = actual.reshape(-1).astype(np.float64, copy=False)
    base = np.abs(y)
    delta = np.abs(m - y) - base
    index = np.searchsorted(THRESHOLDS, p, side="right")
    bins += np.bincount(index, weights=delta, minlength=len(THRESHOLDS) + 1)
    return float(base.sum())


def evaluate_hurdle_model(
    model_name: str,
    predict_magnitude: Callable[[np.ndarray], np.ndarray],
    output_dir: Path,
    device: torch.device,
    hour_batch: int = 4,
    edge_batch: int = 131072,
    history_hours: Optional[int] = None,
) -> dict:
    """Tune the shared activation threshold on validation, then test once."""
    del device, edge_batch
    if history_hours is None:
        raise ValueError("history_hours is required for portable activation caches")
    od = load_od()
    gate_dir = EXPERIMENT_ROOT / "runs" / "activation" / (
        "h%d_seed0" % int(history_hours)
    )
    validation_probability = np.load(
        gate_dir / "validation_probability.npy", mmap_mode="r"
    )
    test_probability = np.load(gate_dir / "test_probability.npy", mmap_mode="r")

    bins = np.zeros(len(THRESHOLDS) + 1, dtype=np.float64)
    base_error = 0.0
    validation_n = 0
    validation_hours = target_hours("validation")
    for begin in range(0, len(validation_hours), hour_batch):
        starts = validation_hours[begin : begin + hour_batch]
        actual = np.asarray(od[starts], dtype=np.float32)
        probability = np.asarray(
            validation_probability[starts - TRAIN_END], dtype=np.float32
        )
        magnitude = predict_magnitude(starts)
        base_error += update_threshold_bins(probability, magnitude, actual, bins)
        validation_n += int(actual.size)
    active_delta = np.cumsum(bins[:0:-1])[::-1]
    validation_mae = (base_error + active_delta) / validation_n
    best_index = int(np.argmin(validation_mae))
    threshold = float(THRESHOLDS[best_index])

    metrics = StreamingMetrics()
    test_hours = target_hours("test")
    for begin in range(0, len(test_hours), hour_batch):
        starts = test_hours[begin : begin + hour_batch]
        actual = np.asarray(od[starts], dtype=np.float32)
        probability = np.asarray(
            test_probability[starts - VAL_END], dtype=np.float32
        )
        magnitude = predict_magnitude(starts)
        prediction = np.where(probability >= threshold, magnitude, 0.0).astype(
            np.float32
        )
        metrics.update(torch.from_numpy(prediction), torch.from_numpy(actual))
    test = metrics.compute()
    result = {
        "model": model_name,
        "zero_treatment": "shared strict activation classifier + positive magnitude",
        "history_hours": int(history_hours),
        "threshold": threshold,
        "validation_mae": float(validation_mae[best_index]),
        "validation_n": int(validation_n),
        "test": test,
        "test_target_hours": [int(test_hours[0]), int(test_hours[-1])],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hurdle_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
