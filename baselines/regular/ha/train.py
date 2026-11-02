#!/usr/bin/env python3
"""HA training and evaluation for the shared hourly OD task."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np


from baselines.regular._shared.simple_data import CityConfig, AggregateTables, write_json, evaluate_hurdle, cache_event_magnitude, base_payload

def run_ha(
    config: CityConfig,
    od: np.ndarray,
    tables: AggregateTables,
    validation_hours: np.ndarray,
    test_hours: np.ndarray,
    history_hours: int,
) -> Dict[str, object]:
    nodes = config.nodes
    edges = nodes * nodes
    started = time.perf_counter()
    seasonal = np.empty((168, edges), dtype=np.float32)
    edge_mean = tables.edge_positive_mean
    for slot in range(168):
        count = tables.seasonal_positive_count[slot]
        value = tables.seasonal_sum[slot] / np.maximum(count, 1)
        seasonal[slot] = np.where(count > 0, value, edge_mean)
    training_seconds = time.perf_counter() - started

    def predict_full(hours: np.ndarray) -> np.ndarray:
        return seasonal[hours % 168].reshape(len(hours), nodes, nodes)

    def predict_samples(hours: np.ndarray, edge: np.ndarray) -> np.ndarray:
        return seasonal[hours % 168, edge]

    evaluation, inference_seconds = evaluate_hurdle(
        config, od, validation_hours, test_hours, history_hours, predict_full, 32
    )
    payload = base_payload(
        config,
        "Historical Average",
        history_hours,
        training_seconds,
        inference_seconds,
        evaluation,
    )
    payload.update(
        {
            "category": "traditional statistical",
            "definition": (
                "per-OD training-only positive mean at the same hour-of-week; "
                "per-OD positive mean fallback; shared activation gate"
            ),
            "parameter_count": 0,
            "deterministic": True,
            "magnitude_training_target": "positive cells only",
        }
    )
    run_dir = config.output_root / "ha" / ("h%d_seed0" % history_hours)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "model.npz", seasonal_positive_mean=seasonal)
    write_json(run_dir / "result.json", payload)
    write_json(run_dir / "config.json", {k: v for k, v in payload.items() if k != "evaluation"})
    if history_hours == 12:
        cache_event_magnitude(config, "supplemental_ha", predict_samples, payload)
    return payload
