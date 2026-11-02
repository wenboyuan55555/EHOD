#!/usr/bin/env python3
"""Run HA, XGBoost, and flattened-OD LSTM under the frozen Hurdle protocol."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


from ehod.common.paths import EXPERIMENT_ROOT as ROOT
import os
DATA_ROOT = Path(os.environ.get("EHOD_DATA_ROOT", str(ROOT / "data")))
OUTPUT_ROOT = Path(os.environ.get("EHOD_OUTPUT_ROOT", str(ROOT / "outputs")))
TRAIN_START = 12
TRAIN_END = 6720
VAL_END = 7728
TEST_END = 8760
THRESHOLDS = np.arange(0.05, 1.0, 0.05, dtype=np.float32)


@dataclass(frozen=True)
class CityConfig:
    name: str
    label: str
    nodes: int
    od_path: Path
    normalized_path: Path
    scaler_path: Path
    experiment_root: Path
    event_index_path: Path
    output_root: Path
    summary_path: Path
    invalid_hours: Tuple[int, ...]
    device: str
    lstm_micro_batch: int


def city_config(name: str, device: Optional[str] = None) -> CityConfig:
    if name not in {"nyc", "chicago"}:
        raise ValueError("Unknown city %r" % name)
    core = DATA_ROOT / name / "core_od"
    output = OUTPUT_ROOT / name
    return CityConfig(
        name=name, label="NYC" if name == "nyc" else "Chicago",
        nodes=263 if name == "nyc" else 77,
        od_path=core / "od_matrix.npy",
        normalized_path=core / "normalized/od_log1p_zscore.npy",
        scaler_path=core / "normalized/scaler.npz",
        experiment_root=output,
        event_index_path=DATA_ROOT / name / "events/event_edges_h12.npz",
        output_root=output / "runs/simple_baselines",
        summary_path=output / "simple_baselines.json",
        invalid_hours=() if name == "nyc" else (1682, 8757, 8758, 8759),
        device=device or "cuda:0", lstm_micro_batch=4 if name == "nyc" else 32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("nyc", "chicago"), required=True)
    parser.add_argument("--models", default="ha,xgboost,lstm")
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xgb-rounds", type=int, default=300)
    parser.add_argument("--xgb-patience", type=int, default=20)
    parser.add_argument("--xgb-threads", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-epochs", type=int, default=40)
    parser.add_argument("--lstm-patience", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def target_hours(config: CityConfig, split: str, smoke: bool = False) -> np.ndarray:
    bounds = {
        "train": (TRAIN_START, TRAIN_END),
        "validation": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }
    start, end = bounds[split]
    hours = np.arange(start, end, dtype=np.int32)
    if config.invalid_hours:
        hours = hours[~np.isin(hours, np.asarray(config.invalid_hours))]
    if smoke:
        hours = hours[: {"train": 48, "validation": 24, "test": 24}[split]]
    return hours


def calendar_features(hours: np.ndarray) -> np.ndarray:
    hour = hours % 24
    weekday = (hours // 24) % 7
    return np.stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
        ],
        axis=1,
    ).astype(np.float32)


class StreamingMetrics:
    FIELDS = (
        "n",
        "absolute",
        "square",
        "sum_y",
        "sum_p",
        "sum_y2",
        "sum_p2",
        "sum_yp",
        "target_total",
        "prediction_total",
        "target_zero",
        "predicted_zero",
        "positive_absolute",
        "positive_n",
        "tp",
        "fp",
        "fn",
    )

    def __init__(self) -> None:
        for field in self.FIELDS:
            setattr(self, field, 0)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        error = p - y
        self.n += int(y.size)
        self.absolute += float(np.abs(error).sum())
        self.square += float(np.square(error).sum())
        self.sum_y += float(y.sum())
        self.sum_p += float(p.sum())
        self.sum_y2 += float(np.square(y).sum())
        self.sum_p2 += float(np.square(p).sum())
        self.sum_yp += float((y * p).sum())
        self.target_total += float(y.sum())
        self.prediction_total += float(p.sum())
        target_positive = y >= 0.5
        predicted_positive = p >= 0.5
        self.target_zero += int((~target_positive).sum())
        self.predicted_zero += int((~predicted_positive).sum())
        self.positive_absolute += float(np.abs(error[target_positive]).sum())
        self.positive_n += int(target_positive.sum())
        self.tp += int((target_positive & predicted_positive).sum())
        self.fp += int((~target_positive & predicted_positive).sum())
        self.fn += int((target_positive & ~predicted_positive).sum())

    def sufficient_statistics(self) -> Dict[str, float]:
        return {field: getattr(self, field) for field in self.FIELDS}

    def compute(self) -> Dict[str, float]:
        n = max(int(self.n), 1)
        covariance = self.sum_yp - self.sum_y * self.sum_p / n
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / n
        variance_p = self.sum_p2 - self.sum_p * self.sum_p / n
        pcc = covariance / math.sqrt(max(variance_y * variance_p, 1.0e-12))
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "n": int(self.n),
            "mae": self.absolute / n,
            "rmse": math.sqrt(self.square / n),
            "pcc": pcc,
            "activation_f1": f1,
            "false_activation_rate": self.fp / max(self.target_zero, 1),
            "miss_activation_rate": self.fn / max(self.positive_n, 1),
            "positive_mae": self.positive_absolute / max(self.positive_n, 1),
            "target_zero_rate": self.target_zero / n,
            "predicted_zero_rate": self.predicted_zero / n,
            "target_total": self.target_total,
            "prediction_total": self.prediction_total,
            "total_flow_absolute_percentage_error": 100.0
            * abs(self.prediction_total - self.target_total)
            / max(self.target_total, 1.0e-12),
        }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@dataclass
class AggregateTables:
    seasonal_sum: np.ndarray
    seasonal_positive_count: np.ndarray
    edge_sum: np.ndarray
    edge_positive_count: np.ndarray
    train_hour_count: int

    @property
    def edge_positive_mean(self) -> np.ndarray:
        return (
            self.edge_sum / np.maximum(self.edge_positive_count, 1)
        ).astype(np.float32)

    def seasonal_positive_mean(self, slots: np.ndarray, edges: np.ndarray) -> np.ndarray:
        count = self.seasonal_positive_count[slots, edges]
        value = self.seasonal_sum[slots, edges] / np.maximum(count, 1)
        fallback = self.edge_sum[edges] / np.maximum(self.edge_positive_count[edges], 1)
        return np.where(count > 0, value, fallback).astype(np.float32)


def build_aggregate_tables(
    od: np.ndarray, train_hours: np.ndarray, nodes: int
) -> AggregateTables:
    edges = nodes * nodes
    seasonal_sum = np.zeros((168, edges), dtype=np.float64)
    seasonal_count = np.zeros((168, edges), dtype=np.int32)
    edge_sum = np.zeros(edges, dtype=np.float64)
    edge_count = np.zeros(edges, dtype=np.int32)
    for begin in range(0, len(train_hours), 48):
        part = train_hours[begin : begin + 48]
        block = np.asarray(od[part], dtype=np.float32).reshape(len(part), edges)
        positive = block > 0
        edge_sum += block.sum(axis=0, dtype=np.float64)
        edge_count += positive.sum(axis=0, dtype=np.int32)
        for local, hour in enumerate(part):
            slot = int(hour % 168)
            seasonal_sum[slot] += block[local]
            seasonal_count[slot] += positive[local]
    return AggregateTables(
        seasonal_sum=seasonal_sum,
        seasonal_positive_count=seasonal_count,
        edge_sum=edge_sum,
        edge_positive_count=edge_count,
        train_hour_count=len(train_hours),
    )








def update_threshold_bins(
    probability: np.ndarray,
    magnitude: np.ndarray,
    actual: np.ndarray,
    bins: np.ndarray,
) -> float:
    p = probability.reshape(-1)
    m = np.maximum(magnitude.reshape(-1), 0.0).astype(np.float64, copy=False)
    y = actual.reshape(-1).astype(np.float64, copy=False)
    base = np.abs(y)
    delta = np.abs(m - y) - base
    index = np.searchsorted(THRESHOLDS, p, side="right")
    bins += np.bincount(index, weights=delta, minlength=len(THRESHOLDS) + 1)
    return float(base.sum())


def evaluate_hurdle(
    config: CityConfig,
    od: np.ndarray,
    validation_hours: np.ndarray,
    test_hours: np.ndarray,
    history_hours: int,
    predict_full: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> Tuple[Dict[str, object], float]:
    gate_dir = config.experiment_root / "runs/activation" / (
        "h%d_seed0" % history_hours
    )
    validation_probability = np.load(
        gate_dir / "validation_probability.npy", mmap_mode="r"
    )
    test_probability = np.load(gate_dir / "test_probability.npy", mmap_mode="r")
    bins = np.zeros(len(THRESHOLDS) + 1, dtype=np.float64)
    base_error = 0.0
    count = 0
    for begin in range(0, len(validation_hours), batch_size):
        part = validation_hours[begin : begin + batch_size]
        magnitude = predict_full(part)
        actual = np.asarray(od[part], dtype=np.float32)
        probability = np.asarray(
            validation_probability[part - TRAIN_END], dtype=np.float32
        )
        base_error += update_threshold_bins(probability, magnitude, actual, bins)
        count += actual.size
    active_delta = np.cumsum(bins[:0:-1])[::-1]
    validation_mae = (base_error + active_delta) / max(count, 1)
    best_index = int(np.argmin(validation_mae))
    threshold = float(THRESHOLDS[best_index])
    metrics = StreamingMetrics()
    started = time.perf_counter()
    for begin in range(0, len(test_hours), batch_size):
        part = test_hours[begin : begin + batch_size]
        magnitude = predict_full(part)
        probability = np.asarray(test_probability[part - VAL_END], dtype=np.float32)
        prediction = np.where(probability >= threshold, magnitude, 0.0).astype(
            np.float32
        )
        metrics.update(prediction, np.asarray(od[part], dtype=np.float32))
    inference_seconds = time.perf_counter() - started
    return (
        {
            "threshold": threshold,
            "validation_mae": float(validation_mae[best_index]),
            "test": metrics.compute(),
            "test_sufficient_statistics": metrics.sufficient_statistics(),
        },
        inference_seconds,
    )


def cache_event_magnitude(
    config: CityConfig,
    cache_key: str,
    predict_samples: Callable[[np.ndarray, np.ndarray], np.ndarray],
    metadata: Dict[str, object],
) -> None:
    index = np.load(config.event_index_path, mmap_mode="r")
    hours = np.asarray(index["sample_hour"], dtype=np.int32)
    edges = np.asarray(index["sample_edge"], dtype=np.int32)
    cache_dir = config.experiment_root / "cache/event_backbones/h12"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (cache_key + "_magnitude.npy")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(hours),))
    for begin in range(0, len(hours), 100_000):
        end = min(begin + 100_000, len(hours))
        output[begin:end] = np.maximum(
            predict_samples(hours[begin:end], edges[begin:end]), 0.0
        )
    output.flush()
    write_json(
        cache_dir / (cache_key + "_magnitude.json"),
        {
            **metadata,
            "samples": len(hours),
            "event_index_path": str(config.event_index_path),
            "uses_event_features": False,
            "dtype": "float32",
        },
    )


def base_payload(
    config: CityConfig,
    model: str,
    history_hours: int,
    training_seconds: float,
    inference_seconds: float,
    evaluation: Dict[str, object],
    seed: int = 0,
) -> Dict[str, object]:
    return {
        "model": model,
        "city": config.label,
        "task": "H%d-to-1 complete %dx%d OD forecasting"
        % (history_hours, config.nodes, config.nodes),
        "history_hours": history_hours,
        "seed": seed,
        "variant": "hurdle",
        "train_target_hours": [TRAIN_START, TRAIN_END],
        "validation_target_hours": [TRAIN_END, VAL_END],
        "test_target_hours": [VAL_END, TEST_END],
        "invalid_target_hours_excluded": list(config.invalid_hours),
        "activation_gate": "shared frozen StrictActivationGate",
        "activation_threshold_grid": THRESHOLDS.tolist(),
        "uses_event_features": False,
        "test_used_for_model_selection": False,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "evaluation": evaluation,
    }








def collect_summary(config: CityConfig) -> Dict[str, object]:
    result: Dict[str, object] = {
        "city": config.label,
        "protocol": "frozen Hurdle magnitude plus shared StrictActivationGate",
        "models": {},
    }
    models = result["models"]
    assert isinstance(models, dict)
    for model in ("ha", "xgboost", "lstm"):
        values = {}
        for history_hours in (1, 3, 6, 12):
            path = config.output_root / model / ("h%d_seed0/result.json" % history_hours)
            if path.exists():
                values["h%d" % history_hours] = json.loads(path.read_text(encoding="utf-8"))
        if values:
            models[model] = values
    write_json(config.summary_path, result)
    return result


def main() -> None:
    from baselines.regular.ha.train import run_ha
    from baselines.regular.xgboost.train import run_xgboost
    from baselines.regular.lstm.train import run_lstm
    args = parse_args()
    config = city_config(args.city, args.device)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    if set(models) - {"ha", "xgboost", "lstm"}:
        raise ValueError("--models must be a subset of ha,xgboost,lstm")
    if set(horizons) - {1, 3, 6, 12}:
        raise ValueError("--horizons must be a subset of 1,3,6,12")
    set_seed(args.seed)
    od = np.load(config.od_path, mmap_mode="r")
    if od.shape != (8760, config.nodes, config.nodes):
        raise ValueError("Unexpected OD shape %r" % (od.shape,))
    train = target_hours(config, "train", args.smoke)
    validation = target_hours(config, "validation", args.smoke)
    test = target_hours(config, "test", args.smoke)
    table_started = time.perf_counter()
    tables = build_aggregate_tables(od, train, config.nodes)
    print(
        json.dumps(
            {
                "city": config.name,
                "aggregate_table_seconds": time.perf_counter() - table_started,
                "positive_training_cells": int(tables.edge_positive_count.sum()),
            }
        ),
        flush=True,
    )
    normalized = None
    scaler = None
    if "lstm" in models:
        normalized = np.load(config.normalized_path, mmap_mode="r")
        scaler = np.load(config.scaler_path)
    for history_hours in horizons:
        print(json.dumps({"city": config.name, "history_hours": history_hours}), flush=True)
        if "ha" in models:
            run_ha(config, od, tables, validation, test, history_hours)
        if "xgboost" in models:
            run_xgboost(
                config,
                od,
                tables,
                train,
                validation,
                test,
                history_hours,
                10 if args.smoke else args.xgb_rounds,
                3 if args.smoke else args.xgb_patience,
                args.xgb_threads,
                args.seed,
            )
        if "lstm" in models:
            assert normalized is not None and scaler is not None
            run_lstm(
                config,
                od,
                normalized,
                float(scaler["mean"]),
                float(scaler["std"]),
                train,
                validation,
                test,
                history_hours,
                args.lstm_hidden,
                2 if args.smoke else args.lstm_epochs,
                2 if args.smoke else args.lstm_patience,
                args.seed,
            )
        if not args.smoke and args.seed == 0:
            collect_summary(config)
    if not args.smoke and args.seed == 0:
        summary = collect_summary(config)
        print(
            json.dumps(
                {
                    "city": summary["city"],
                    "summary_path": str(config.summary_path),
                    "completed_models": sorted(summary["models"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elif args.smoke:
        print(json.dumps({"smoke": True}), flush=True)
    else:
        print(
            json.dumps(
                {
                    "city": config.label,
                    "seed": args.seed,
                    "completed_models": models,
                    "completed_histories": horizons,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
