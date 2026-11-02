from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, load_od
from ehod.common.device import require_cuda
from ehod.data.events import ScheduledEventStore
from ehod.evaluation.metrics import StreamingMetrics
from ehod.common.paths import EVENT_INDEX_PATH


INDEX_PATH = EVENT_INDEX_PATH
FEATURE_PATH = EXPERIMENT_ROOT / "cache" / "event_structured_features.npy"
STRENGTHS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float32)
RETRIEVAL_K = (3, 5, 10, 20, 40)
RETRIEVAL_TEMPERATURE = (0.25, 0.5, 1.0, 2.0)
RETRIEVAL_THRESHOLDS = np.arange(0.05, 1.0, 0.05, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument(
        "--method", choices=["posthoc", "retrieval", "event_matrix"], required=True
    )
    parser.add_argument("--base-threshold", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Optional event-backbone cache prefix; defaults to --backbone.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ridge", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser.parse_args()


def metric_dict(prediction: np.ndarray, target: np.ndarray):
    metrics = StreamingMetrics()
    metrics.update(
        torch.from_numpy(np.asarray(prediction, dtype=np.float32)),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
    )
    return metrics.compute()


def choose_strength(base, correction, target, indices):
    best = None
    for strength in STRENGTHS:
        prediction = np.maximum(
            base[indices] + float(strength) * correction[indices], 0.0
        )
        mae = float(np.mean(np.abs(prediction - target[indices])))
        row = (mae, float(strength))
        if best is None or row < best:
            best = row
    return best


def fit_ridge(features, residual, indices, ridge, batch_size):
    width = int(features.shape[1]) + 1
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros(width, dtype=np.float64)
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        x = np.empty((len(selected), width), dtype=np.float64)
        x[:, 0] = 1.0
        x[:, 1:] = np.asarray(features[selected], dtype=np.float64)
        y = residual[selected].astype(np.float64)
        xtx += x.T @ x
        xty += x.T @ y
    penalty = np.eye(width, dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty).astype(np.float32)


def ridge_correction(features, coefficient, length, batch_size):
    output = np.empty(length, dtype=np.float32)
    for begin in range(0, length, batch_size):
        end = min(begin + batch_size, length)
        x = np.asarray(features[begin:end], dtype=np.float32)
        output[begin:end] = coefficient[0] + x @ coefficient[1:]
    return output


def event_phase(hour, event, start_hour, end_hour):
    value = np.full(len(hour), 4, dtype=np.int8)
    begin = start_hour[event]
    finish = end_hour[event]
    value[(hour >= begin - 3) & (hour <= begin - 2)] = 0
    value[hour == begin - 1] = 1
    value[(hour >= begin) & (hour < finish)] = 2
    value[(hour >= finish) & (hour <= finish + 1)] = 3
    return value


def event_embedding(index, train_event):
    """Forecast-safe event key used to retrieve historical event cases.

    Besides schedule metadata, the key contains a compact traffic context from
    the 24 hours ending before the event influence window starts.  Therefore a
    validation/test event never uses traffic observed during or after itself.
    """
    event_type = np.asarray(index["event_type"], dtype=np.int64)
    venue = np.asarray(index["event_venue"], dtype=np.int64)
    venue_node = np.asarray(index["event_venue_node"], dtype=np.int64)
    start_min = np.asarray(index["event_start_min"], dtype=np.float32)
    end_min = np.asarray(index["event_end_min"], dtype=np.float32)
    scale = np.asarray(index["event_scale"], dtype=np.float32)
    start_hour = np.floor(start_min / 60.0).astype(np.int32)
    duration = np.maximum(end_min - start_min, 30.0)
    hour_of_day = np.mod(start_min / 60.0, 24.0)
    weekday = np.mod(start_hour // 24, 7).astype(np.float32)
    od = load_od()
    traffic = np.zeros((len(event_type), 8), dtype=np.float32)
    for event, (hour, node) in enumerate(zip(start_hour, venue_node)):
        cutoff = max(int(hour) - 3, 1)
        begin = max(0, cutoff - 24)
        block = np.asarray(od[begin:cutoff], dtype=np.float32)
        if not len(block):
            continue
        arrival = block[:, :, int(node)].sum(axis=1)
        departure = block[:, int(node), :].sum(axis=1)
        recent = min(6, len(block))
        traffic[event] = np.asarray(
            [
                np.log1p(arrival.mean()),
                np.log1p(departure.mean()),
                np.log1p(arrival[-recent:].mean()),
                np.log1p(departure[-recent:].mean()),
                np.log1p(block.sum(axis=(1, 2)).mean()),
                float((block > 0).mean()),
                np.log1p(arrival[-1]),
                np.log1p(departure[-1]),
            ],
            dtype=np.float32,
        )
    scalar = np.column_stack(
        [
            np.log1p(np.maximum(scale, 0.0)),
            np.log1p(duration),
            np.sin(2.0 * np.pi * hour_of_day / 24.0),
            np.cos(2.0 * np.pi * hour_of_day / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            traffic,
        ]
    ).astype(np.float32)
    mean = scalar[train_event].mean(axis=0)
    std = scalar[train_event].std(axis=0)
    scalar = (scalar - mean) / np.maximum(std, 1.0e-5)
    type_count = int(event_type.max()) + 1
    venue_count = int(venue.max()) + 1
    type_one_hot = np.eye(type_count, dtype=np.float32)[event_type] * 2.0
    venue_one_hot = np.eye(venue_count, dtype=np.float32)[venue] * 0.8
    return np.concatenate([type_one_hot, venue_one_hot, scalar], axis=1)


def event_case_profiles(index, probability, magnitude, target):
    """Build training-only Hurdle response profiles for every historical event."""
    sample_hour = np.asarray(index["sample_hour"], dtype=np.int32)
    sample_edge = np.asarray(index["sample_edge"], dtype=np.int32)
    pair_ptr = np.asarray(index["pair_ptr"], dtype=np.int64)
    pair_event = np.asarray(index["pair_event"], dtype=np.int32)
    pair_role = np.asarray(index["pair_role"], dtype=np.int8)
    start_hour = np.floor(
        np.asarray(index["event_start_min"], dtype=np.float32) / 60.0
    ).astype(np.int32)
    end_hour = np.floor(
        np.asarray(index["event_end_min"], dtype=np.float32) / 60.0
    ).astype(np.int32)
    train_event = (start_hour - 3 >= 12) & (end_hour + 3 < 6720)
    pair_sample = np.repeat(
        np.arange(len(sample_hour), dtype=np.int32), np.diff(pair_ptr)
    )
    pair_hour = sample_hour[pair_sample]
    phase = event_phase(pair_hour, pair_event, start_hour, end_hour)
    origin = sample_edge[pair_sample] // N_NODES
    destination = sample_edge[pair_sample] % N_NODES
    counterpart = np.where(pair_role == 0, origin, destination).astype(np.int32)
    valid = (pair_hour < 6720) & train_event[pair_event]
    event_count = len(start_hour)
    width = event_count * 5 * 3 * N_NODES
    key = (
        ((pair_event[valid].astype(np.int64) * 5 + phase[valid]) * 3
        + pair_role[valid].astype(np.int64))
        * N_NODES
        + counterpart[valid]
    )
    sample = pair_sample[valid]
    activation_residual = (
        (target[sample] > 0).astype(np.float32) - probability[sample]
    )
    activation_sum = np.bincount(
        key, weights=activation_residual, minlength=width
    ).reshape(event_count, 5, 3, N_NODES)
    activation_count = np.bincount(key, minlength=width).reshape(
        event_count, 5, 3, N_NODES
    )
    positive = target[sample] > 0
    magnitude_sum = np.bincount(
        key[positive],
        weights=(
            np.log1p(target[sample[positive]])
            - np.log1p(np.maximum(magnitude[sample[positive]], 0.0))
        ),
        minlength=width,
    ).reshape(event_count, 5, 3, N_NODES)
    magnitude_count = np.bincount(
        key[positive], minlength=width
    ).reshape(event_count, 5, 3, N_NODES)
    event_type = np.asarray(index["event_type"], dtype=np.int64)
    activation_profile = np.zeros_like(activation_sum, dtype=np.float32)
    magnitude_profile = np.zeros_like(magnitude_sum, dtype=np.float32)
    for kind in range(int(event_type.max()) + 1):
        selected = train_event & (event_type == kind)
        if not bool(selected.any()):
            selected = train_event
        a_sum = activation_sum[selected].sum(axis=0)
        a_count = activation_count[selected].sum(axis=0)
        m_sum = magnitude_sum[selected].sum(axis=0)
        m_count = magnitude_count[selected].sum(axis=0)
        a_prior = a_sum / np.maximum(a_count, 1.0)
        m_prior = m_sum / np.maximum(m_count, 1.0)
        members = np.flatnonzero(event_type == kind)
        activation_profile[members] = (
            activation_sum[members] + 2.0 * a_prior[None]
        ) / (activation_count[members] + 2.0)
        magnitude_profile[members] = (
            magnitude_sum[members] + 2.0 * m_prior[None]
        ) / (magnitude_count[members] + 2.0)
    return {
        "activation": activation_profile,
        "magnitude": magnitude_profile,
        "train_event": train_event,
        "pair_sample": pair_sample,
        "pair_event": pair_event,
        "pair_phase": phase,
        "pair_role": pair_role,
        "pair_counterpart": counterpart,
        "embedding": event_embedding(index, train_event),
    }


def retrieved_event_profiles(case, neighbors, distances, k, temperature):
    event_count = len(case["train_event"])
    activation = np.zeros((event_count, 5, 3, N_NODES), dtype=np.float32)
    magnitude = np.zeros_like(activation)
    selected_neighbors = neighbors[:, :k]
    selected_distance = distances[:, :k]
    weight = np.exp(-selected_distance / float(temperature)).astype(np.float32)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1.0e-8)
    for event in range(event_count):
        activation[event] = np.tensordot(
            weight[event], case["activation"][selected_neighbors[event]], axes=(0, 0)
        )
        magnitude[event] = np.tensordot(
            weight[event], case["magnitude"][selected_neighbors[event]], axes=(0, 0)
        )
    return activation, magnitude


def sample_retrieval_correction(case, event_activation, event_magnitude, length):
    pair_activation = event_activation[
        case["pair_event"],
        case["pair_phase"],
        case["pair_role"],
        case["pair_counterpart"],
    ]
    pair_magnitude = event_magnitude[
        case["pair_event"],
        case["pair_phase"],
        case["pair_role"],
        case["pair_counterpart"],
    ]
    count = np.maximum(
        np.bincount(case["pair_sample"], minlength=length), 1
    ).astype(np.float32)
    activation = np.bincount(
        case["pair_sample"], weights=pair_activation, minlength=length
    ) / count
    magnitude = np.bincount(
        case["pair_sample"], weights=pair_magnitude, minlength=length
    ) / count
    return activation.astype(np.float32), magnitude.astype(np.float32)


def retrieval_validation(
    base_probability,
    base_magnitude,
    activation_delta,
    magnitude_delta,
    target,
    validation_index,
):
    best = None
    base_logmag = np.log1p(np.maximum(base_magnitude[validation_index], 0.0))
    for strength in STRENGTHS:
        probability = np.clip(
            base_probability[validation_index]
            + float(strength) * activation_delta[validation_index],
            1.0e-5,
            1.0 - 1.0e-5,
        )
        magnitude = np.expm1(
            np.maximum(
                base_logmag + float(strength) * magnitude_delta[validation_index],
                0.0,
            )
        )
        for threshold in RETRIEVAL_THRESHOLDS:
            prediction = np.where(probability >= threshold, magnitude, 0.0)
            mae = float(np.mean(np.abs(prediction - target[validation_index])))
            row = (mae, float(strength), float(threshold))
            if best is None or row < best:
                best = row
    return best


def event_matrix_features(index, cache_path: Path):
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    sample_hour = index["sample_hour"]
    sample_edge = index["sample_edge"]
    hours, starts, counts = np.unique(
        sample_hour, return_index=True, return_counts=True
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=(len(sample_hour), 3)
    )
    store = ScheduledEventStore()
    for group, hour in enumerate(hours):
        begin = int(starts[group])
        end = begin + int(counts[group])
        matrix = store.batch_matrix(np.asarray([hour], dtype=np.int32))[0].reshape(-1, 3)
        output[begin:end] = matrix[sample_edge[begin:end]].astype(np.float16)
    output.flush()
    return np.load(cache_path, mmap_mode="r")


def main():
    args = parse_args()
    device = require_cuda(args.device)
    np.random.seed(args.seed)
    started = time.time()
    index = np.load(INDEX_PATH, mmap_mode="r")
    hour = np.asarray(index["sample_hour"], dtype=np.int32)
    target = np.asarray(index["sample_target"], dtype=np.float32)
    train_index = np.flatnonzero(hour < 6720)
    validation_index = np.flatnonzero((hour >= 6720) & (hour < 7728))
    test_index = np.flatnonzero(hour >= 7728)
    cache_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    cache_key = args.cache_key or args.backbone
    probability_path = cache_dir / (cache_key + "_probability.npy")
    if not probability_path.exists():
        probability_path = cache_dir / "strict_activation_probability.npy"
    probability = np.asarray(np.load(probability_path, mmap_mode="r"), dtype=np.float32)
    magnitude = np.asarray(
        np.load(cache_dir / (cache_key + "_magnitude.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    base = np.where(probability >= args.base_threshold, magnitude, 0.0).astype(
        np.float32
    )
    residual = target - base
    details = {}
    if args.method == "posthoc":
        features = np.load(FEATURE_PATH, mmap_mode="r")
        coefficient = fit_ridge(
            features, residual, train_index, args.ridge, args.batch_size
        )
        correction = ridge_correction(
            features, coefficient, len(target), args.batch_size
        )
        details = {
            "design": "ridge additive residual after final Hurdle prediction",
            "feature_dim": int(features.shape[1]),
            "ridge": args.ridge,
        }
    elif args.method == "retrieval":
        case = event_case_profiles(index, probability, magnitude, target)
        train_events = np.flatnonzero(case["train_event"])
        embedding = case["embedding"]
        squared_distance = np.mean(
            np.square(embedding[:, None, :] - embedding[train_events][None, :, :]),
            axis=2,
        )
        for row, event in enumerate(train_events):
            squared_distance[event, row] = np.inf
        order = np.argsort(squared_distance, axis=1)
        neighbors = train_events[order]
        distances = np.take_along_axis(squared_distance, order, axis=1)
        best = None
        best_activation = None
        best_magnitude = None
        for k in RETRIEVAL_K:
            available = min(int(k), len(train_events) - 1)
            for temperature in RETRIEVAL_TEMPERATURE:
                event_activation, event_magnitude = retrieved_event_profiles(
                    case, neighbors, distances, available, temperature
                )
                activation_delta, magnitude_delta = sample_retrieval_correction(
                    case, event_activation, event_magnitude, len(target)
                )
                selection = retrieval_validation(
                    probability,
                    magnitude,
                    activation_delta,
                    magnitude_delta,
                    target,
                    validation_index,
                )
                candidate = (
                    float(selection[0]),
                    int(available),
                    float(temperature),
                    float(selection[1]),
                    float(selection[2]),
                )
                print(
                    json.dumps(
                        {
                            "retrieval_k": available,
                            "temperature": temperature,
                            "validation_direct_mae": selection[0],
                            "strength": selection[1],
                            "threshold": selection[2],
                        }
                    ),
                    flush=True,
                )
                if best is None or candidate < best:
                    best = candidate
                    best_activation = activation_delta
                    best_magnitude = magnitude_delta
        selection_mae, selected_k, selected_temperature, strength, event_threshold = best
        probability_event = np.clip(
            probability[test_index]
            + strength * best_activation[test_index],
            1.0e-5,
            1.0 - 1.0e-5,
        )
        magnitude_event = np.expm1(
            np.maximum(
                np.log1p(np.maximum(magnitude[test_index], 0.0))
                + strength * best_magnitude[test_index],
                0.0,
            )
        )
        event_prediction = np.where(
            probability_event >= event_threshold, magnitude_event, 0.0
        ).astype(np.float32)
        details = {
            "design": (
                "training-only event case library; retrieve complete historical "
                "five-phase OD impact profiles and inject separate activation/"
                "positive-magnitude corrections into Hurdle"
            ),
            "retrieval_key": (
                "event type, venue, scale, start time, duration, weekday, and "
                "24-hour pre-influence traffic context"
            ),
            "training_events": int(len(train_events)),
            "selected_k": int(selected_k),
            "selected_temperature": float(selected_temperature),
            "selected_event_threshold": float(event_threshold),
            "leakage_control": (
                "only events whose complete influence window ends before hour 6720 "
                "are stored; context ends before start-3h"
            ),
        }
    else:
        features = event_matrix_features(
            index, EXPERIMENT_ROOT / "cache" / "event_matrix_features.npy"
        )
        coefficient = fit_ridge(
            features, residual, train_index, args.ridge, args.batch_size
        )
        correction = ridge_correction(
            features, coefficient, len(target), args.batch_size
        )
        details = {
            "design": "OD-PF-style arrival/departure/active event matrix residual",
            "feature_dim": 3,
            "ridge": args.ridge,
        }
    if args.method != "retrieval":
        selection_mae, strength = choose_strength(
            base, correction, target, validation_index
        )
        event_prediction = np.maximum(
            base[test_index] + strength * correction[test_index], 0.0
        ).astype(np.float32)
    base_prediction = base[test_index]
    base_metrics = metric_dict(base_prediction, target[test_index])
    event_metrics = metric_dict(event_prediction, target[test_index])
    result = {
        "backbone": args.backbone,
        "backbone_cache_key": cache_key,
        "history_hours": args.history_hours,
        "method": args.method,
        "seed": args.seed,
        "device": str(device),
        "adapter_strength": strength,
        "validation_direct_mae": selection_mae,
        "base_threshold": args.base_threshold,
        "training_seconds": time.time() - started,
        "test_direct_samples": int(len(test_index)),
        "details": details,
        "base": base_metrics,
        "event": event_metrics,
        "change": {
            "mae_percent": 100.0
            * (base_metrics["mae"] - event_metrics["mae"])
            / base_metrics["mae"],
            "rmse_percent": 100.0
            * (base_metrics["rmse"] - event_metrics["rmse"])
            / base_metrics["rmse"],
            "pcc_delta": event_metrics["pcc"] - base_metrics["pcc"],
        },
    }
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "event_adapters"
        / ("h%d" % args.history_hours)
        / args.backbone
        / ((args.run_name or args.method) + "_seed%d" % args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
