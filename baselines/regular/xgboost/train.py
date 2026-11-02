#!/usr/bin/env python3
"""XGBOOST training and evaluation for the shared hourly OD task."""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


from baselines.regular._shared.simple_data import CityConfig, AggregateTables, calendar_features, write_json, evaluate_hurdle, cache_event_magnitude, base_payload

def xgb_feature_names(history_hours: int) -> List[str]:
    names = ["lag_%02d_log1p" % lag for lag in range(1, history_hours + 1)]
    names += ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    names += [
        "seasonal_positive_mean_log1p",
        "edge_positive_mean_log1p",
        "edge_nonzero_frequency",
        "origin_index",
        "destination_index",
    ]
    return names

def make_xgb_sample_features(
    od: np.ndarray,
    hours: np.ndarray,
    edge: np.ndarray,
    history_hours: int,
    tables: AggregateTables,
    nodes: int,
    target: Optional[np.ndarray] = None,
    leave_one_out: bool = False,
) -> np.ndarray:
    hours = np.asarray(hours, dtype=np.int32)
    edge = np.asarray(edge, dtype=np.int32)
    origin = edge // nodes
    destination = edge % nodes
    features = np.empty((len(hours), history_hours + 9), dtype=np.float32)
    for column, lag in enumerate(range(1, history_hours + 1)):
        features[:, column] = np.log1p(
            np.asarray(od[hours - lag, origin, destination], dtype=np.float32)
        )
    features[:, history_hours : history_hours + 4] = calendar_features(hours)
    slot = hours % 168
    if leave_one_out:
        if target is None:
            raise ValueError("leave-one-out features require the row target")
        y = np.asarray(target, dtype=np.float64)
        seasonal_count = tables.seasonal_positive_count[slot, edge] - 1
        edge_count = tables.edge_positive_count[edge] - 1
        edge_mean = (tables.edge_sum[edge] - y) / np.maximum(edge_count, 1)
        edge_mean = np.where(edge_count > 0, edge_mean, 0.0)
        seasonal_mean = (tables.seasonal_sum[slot, edge] - y) / np.maximum(
            seasonal_count, 1
        )
        seasonal_mean = np.where(seasonal_count > 0, seasonal_mean, edge_mean)
        frequency = np.maximum(edge_count, 0) / max(tables.train_hour_count - 1, 1)
    else:
        seasonal_mean = tables.seasonal_positive_mean(slot, edge)
        edge_mean = tables.edge_sum[edge] / np.maximum(
            tables.edge_positive_count[edge], 1
        )
        frequency = tables.edge_positive_count[edge] / max(tables.train_hour_count, 1)
    offset = history_hours + 4
    features[:, offset] = np.log1p(np.maximum(seasonal_mean, 0.0))
    features[:, offset + 1] = np.log1p(np.maximum(edge_mean, 0.0))
    features[:, offset + 2] = frequency
    denominator = max(nodes - 1, 1)
    features[:, offset + 3] = origin / denominator
    features[:, offset + 4] = destination / denominator
    return features

def positive_xgb_arrays(
    od: np.ndarray,
    hours: np.ndarray,
    history_hours: int,
    tables: AggregateTables,
    nodes: int,
    leave_one_out: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = nodes * nodes
    if leave_one_out:
        total = int(tables.edge_positive_count.sum())
    else:
        total = 0
        for begin in range(0, len(hours), 96):
            total += int((np.asarray(od[hours[begin : begin + 96]]) > 0).sum())
    features = np.empty((total, history_hours + 9), dtype=np.float32)
    labels = np.empty(total, dtype=np.float32)
    cursor = 0
    for begin in range(0, len(hours), 48):
        part = hours[begin : begin + 48]
        block = np.asarray(od[part], dtype=np.float32).reshape(len(part), edges)
        row, edge = np.nonzero(block > 0)
        size = len(edge)
        sample_hours = part[row]
        target = block[row, edge]
        features[cursor : cursor + size] = make_xgb_sample_features(
            od,
            sample_hours,
            edge,
            history_hours,
            tables,
            nodes,
            target=target,
            leave_one_out=leave_one_out,
        )
        labels[cursor : cursor + size] = np.log1p(target)
        cursor += size
    if cursor != total:
        raise RuntimeError("positive-row count changed while building XGBoost data")
    return features, labels

def run_xgboost(
    config: CityConfig,
    od: np.ndarray,
    tables: AggregateTables,
    train_hours: np.ndarray,
    validation_hours: np.ndarray,
    test_hours: np.ndarray,
    history_hours: int,
    rounds: int,
    patience: int,
    threads: int,
    seed: int,
) -> Dict[str, object]:
    import xgboost as xgb

    names = xgb_feature_names(history_hours)
    started = time.perf_counter()
    train_x, train_y = positive_xgb_arrays(
        od, train_hours, history_hours, tables, config.nodes, leave_one_out=True
    )
    validation_x, validation_y = positive_xgb_arrays(
        od,
        validation_hours,
        history_hours,
        tables,
        config.nodes,
        leave_one_out=False,
    )
    dtrain = xgb.QuantileDMatrix(
        train_x, label=train_y, feature_names=names, max_bin=128
    )
    dvalidation = xgb.QuantileDMatrix(
        validation_x,
        label=validation_y,
        feature_names=names,
        max_bin=128,
        ref=dtrain,
    )
    training_samples = len(train_y)
    validation_samples = len(validation_y)
    del train_x, train_y, validation_x, validation_y
    gc.collect()

    def raw_scale_mae(prediction: np.ndarray, matrix) -> Tuple[str, float]:
        actual = np.expm1(matrix.get_label())
        predicted = np.maximum(np.expm1(prediction), 0.0)
        return "raw_scale_positive_mae", float(np.abs(predicted - actual).mean())

    params = {
        "objective": "reg:squarederror",
        "disable_default_eval_metric": 1,
        "tree_method": "hist",
        "max_bin": 128,
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 20.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": seed,
        "nthread": threads,
    }
    history: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dvalidation, "validation")],
        custom_metric=raw_scale_mae,
        evals_result=history,
        early_stopping_rounds=patience,
        verbose_eval=10,
    )
    training_seconds = time.perf_counter() - started
    best_iteration = int(booster.best_iteration)

    def predict_samples(hours: np.ndarray, edge: np.ndarray) -> np.ndarray:
        features = make_xgb_sample_features(
            od, hours, edge, history_hours, tables, config.nodes
        )
        prediction = booster.inplace_predict(
            features, iteration_range=(0, best_iteration + 1)
        )
        return np.maximum(np.expm1(prediction), 0.0).astype(np.float32)

    def predict_full(hours: np.ndarray) -> np.ndarray:
        edge = np.tile(np.arange(config.nodes * config.nodes, dtype=np.int32), len(hours))
        sample_hours = np.repeat(hours, config.nodes * config.nodes)
        return predict_samples(sample_hours, edge).reshape(
            len(hours), config.nodes, config.nodes
        )

    evaluation, inference_seconds = evaluate_hurdle(
        config, od, validation_hours, test_hours, history_hours, predict_full, 8
    )
    payload = base_payload(
        config,
        "XGBoost",
        history_hours,
        training_seconds,
        inference_seconds,
        evaluation,
        seed,
    )
    payload.update(
        {
            "category": "machine learning",
            "device": "CPU",
            "implementation": "one pooled positive-magnitude regressor shared by all OD pairs",
            "target_transform": "log1p",
            "features": names,
            "training_samples": training_samples,
            "validation_positive_samples": validation_samples,
            "best_iteration": best_iteration,
            "best_validation_raw_scale_positive_mae": float(booster.best_score),
            "validation_selection_metric": "raw-scale positive-cell MAE",
            "aggregate_feature_protocol": (
                "training rows use exact leave-one-out seasonal/OD means and "
                "nonzero frequency; validation/test use fixed train-only aggregates"
            ),
            "parameters": params,
            "library_version": xgb.__version__,
        }
    )
    run_dir = config.output_root / "xgboost" / (
        "h%d_seed%d" % (history_hours, seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(run_dir / "best_model.ubj")
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "result.json", payload)
    write_json(run_dir / "config.json", {k: v for k, v in payload.items() if k != "evaluation"})
    if history_hours == 12:
        cache_key = "supplemental_xgboost" + ("" if seed == 0 else "_seed%d" % seed)
        cache_event_magnitude(config, cache_key, predict_samples, payload)
    dvalidation = dtrain = booster = None
    gc.collect()
    return payload
