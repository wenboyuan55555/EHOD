from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ehod.hcf.features import ContinuousFeatureStore, full_edge_indices
from ehod.data.od import EXPERIMENT_ROOT, N_NODES, VAL_END, calendar_features, history_batch, inverse_log_standardize, iter_hour_batches, load_normalized, load_scaler, load_static
from ehod.common.device import require_cuda
from ehod.evaluation.metrics import StreamingMetrics
from baselines.regular._shared.factory import create_model, forward_model, static_tensors
from ehod.hcf.model import CONTEXT_ABLATIONS, CONTINUOUS_MODES, FEATURE_ABLATIONS, ContinuousHybridOD, feature_dimension, hybrid_features
from ehod.common.paths import EVENT_INDEX_PATH
from ehod.hcf.training_utils import sample_edges, to_torch, torch_static


THRESHOLDS = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
BLENDS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hour-batch", type=int, default=4)
    parser.add_argument("--uniform-edges", type=int, default=8192)
    parser.add_argument("--positive-edges", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument(
        "--magnitude-loss", choices=["log", "raw", "decision"], default="raw"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-hours", type=int, default=None)
    parser.add_argument("--max-eval-hours", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continuous-mode", choices=CONTINUOUS_MODES, default="exact")
    parser.add_argument(
        "--feature-ablation",
        choices=FEATURE_ABLATIONS,
        default="none",
        help="Mask one documented Count-Feature input group without changing capacity.",
    )
    parser.add_argument(
        "--context-ablation",
        choices=CONTEXT_ABLATIONS,
        default="none",
        help="Zero an embedding context group while retaining identical capacity.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("hurdle", "direct"),
        default="hurdle",
        help=(
            "hurdle predicts activation and positive magnitude; direct predicts "
            "the non-negative count without an activation gate or threshold"
        ),
    )
    parser.add_argument(
        "--backbone-variant",
        choices=("hurdle", "plain"),
        default=None,
        help=(
            "STIDGCN checkpoint used as the frozen base. Defaults to hurdle for "
            "hurdle prediction and plain for direct prediction."
        ),
    )
    parser.add_argument("--joint-backbone", action="store_true")
    parser.add_argument("--no-zero-init", action="store_true")
    parser.add_argument("--fixed-activation-blend", type=float, default=None)
    parser.add_argument("--fixed-residual-blend", type=float, default=None)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument(
        "--evaluation-selection",
        choices=("balanced", "mae"),
        default="mae",
        help=(
            "Validation-only operating-point rule. mae is the reported legacy "
            "protocol; balanced is an auditable three-metric sensitivity check."
        ),
    )
    parser.add_argument("--evaluation-only-checkpoint", type=Path, default=None)
    return parser.parse_args()


def discrete_prediction(
    backbone,
    normalized,
    starts,
    history_hours,
    discrete_static,
    device,
    mean,
    std,
) -> torch.Tensor:
    history = torch.from_numpy(
        history_batch(normalized, starts, history_hours)
    ).to(device)
    calendar = torch.from_numpy(calendar_features(starts)).to(device)
    target_starts = torch.from_numpy(starts.astype(np.int64)).to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        scaled = forward_model(
            "stidgcn",
            backbone,
            history,
            calendar,
            discrete_static,
            target_starts=target_starts,
        )
    return inverse_log_standardize(scaled, mean, std)


def continuous_inputs(
    store,
    static,
    starts,
    origin,
    destination,
    history_hours,
    device,
    continuous_mode="exact",
    feature_ablation="none",
):
    arrays = store.selected(
        starts,
        origin,
        destination,
        history_hours,
        continuous_mode=continuous_mode,
    )
    origin_tensor = to_torch(origin, device).long()
    destination_tensor = to_torch(destination, device).long()
    starts_tensor = to_torch(starts.astype(np.int64), device).long()
    quarter_edge = arrays.get("quarter_edge")
    quarter_origin = arrays.get("quarter_origin")
    quarter_destination = arrays.get("quarter_destination")
    features, _ = hybrid_features(
        to_torch(arrays["edge_history"], device),
        to_torch(arrays["origin_history"], device),
        to_torch(arrays["destination_history"], device),
        to_torch(arrays["continuous_intensity"], device),
        to_torch(arrays["origin_recency"], device),
        to_torch(arrays["destination_recency"], device),
        starts_tensor,
        origin_tensor,
        destination_tensor,
        static,
        continuous_mode=continuous_mode,
        feature_ablation=feature_ablation,
        quarter_edge=(
            None if quarter_edge is None else to_torch(quarter_edge, device)
        ),
        quarter_origin=(
            None if quarter_origin is None else to_torch(quarter_origin, device)
        ),
        quarter_destination=(
            None
            if quarter_destination is None
            else to_torch(quarter_destination, device)
        ),
    )
    return (
        features,
        origin_tensor,
        destination_tensor,
        to_torch(calendar_features(starts), device),
        to_torch(arrays["target"], device),
    )


def model_base_log(base_selected, continuous_mode):
    if continuous_mode == "continuous_only":
        return torch.zeros_like(base_selected)
    return torch.log1p(base_selected)


def selected_base(base_matrix, origin, destination, device):
    origin_tensor = to_torch(origin, device).long()
    destination_tensor = to_torch(destination, device).long()
    batch_index = torch.arange(len(origin), device=device)[:, None]
    return base_matrix[batch_index, origin_tensor, destination_tensor]


@torch.no_grad()
def validation_loss(
    model,
    backbone,
    store,
    normalized,
    static,
    discrete_static,
    args,
    device,
    mean,
    std,
):
    model.eval()
    backbone.eval()
    rng = np.random.default_rng(args.seed + 7001)
    total = 0.0
    n = 0
    limit = 8 if args.smoke else min(args.max_eval_hours or 256, 256)
    for starts in iter_hour_batches("validation", args.hour_batch, limit=limit):
        origin, destination = sample_edges(
            store,
            starts,
            min(args.uniform_edges, 4096),
            min(args.positive_edges, 2048),
            rng,
        )
        base = discrete_prediction(
            backbone,
            normalized,
            starts,
            args.history_hours,
            discrete_static,
            device,
            mean,
            std,
        )
        features, o, d, calendar, target = continuous_inputs(
            store,
            static,
            starts,
            origin,
            destination,
            args.history_hours,
            device,
            args.continuous_mode,
            args.feature_ablation,
        )
        base_selected = selected_base(base, origin, destination, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, magnitude_log = model(
                features,
                o,
                d,
                calendar,
                model_base_log(base_selected, args.continuous_mode),
            )
            uniform = min(args.uniform_edges, 4096)
            if args.prediction_mode == "direct":
                direct_count = torch.expm1(magnitude_log[:, :uniform])
                loss = torch.abs(direct_count - target[:, :uniform]).mean()
            elif args.magnitude_loss == "decision":
                activation = F.binary_cross_entropy_with_logits(
                    logits[:, :uniform], (target[:, :uniform] > 0).float()
                )
                probability = torch.sigmoid(logits[:, :uniform])
                raw_magnitude = torch.expm1(magnitude_log[:, :uniform])
                uniform_target = target[:, :uniform]
                expected_absolute = (
                    probability * torch.abs(raw_magnitude - uniform_target)
                    + (1.0 - probability) * torch.abs(uniform_target)
                ).mean()
                positive = target > 0
                magnitude = F.smooth_l1_loss(
                    torch.expm1(magnitude_log[positive]), target[positive], beta=1.0
                )
                loss = expected_absolute + 0.1 * activation + 0.5 * magnitude
            elif args.magnitude_loss == "raw":
                activation = F.binary_cross_entropy_with_logits(
                    logits[:, :uniform], (target[:, :uniform] > 0).float()
                )
                positive = target > 0
                magnitude = F.smooth_l1_loss(
                    torch.expm1(magnitude_log[positive]), target[positive], beta=1.0
                )
                loss = 0.3 * activation + magnitude
            else:
                activation = F.binary_cross_entropy_with_logits(
                    logits[:, :uniform], (target[:, :uniform] > 0).float()
                )
                positive = target > 0
                magnitude = F.smooth_l1_loss(
                    magnitude_log[positive], torch.log1p(target[positive])
                )
                loss = activation + magnitude
        total += float(loss.item()) * len(starts)
        n += len(starts)
    return total / max(n, 1)


@torch.no_grad()
def fusion_prediction(
    model,
    backbone,
    store,
    normalized,
    static,
    discrete_static,
    starts,
    history_hours,
    device,
    mean,
    std,
    continuous_mode="exact",
    feature_ablation="none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = discrete_prediction(
        backbone,
        normalized,
        starts,
        history_hours,
        discrete_static,
        device,
        mean,
        std,
    )
    base_flat = base.reshape(len(starts), -1)
    origin, destination = full_edge_indices(len(starts))
    probability_parts = []
    residual_parts = []
    chunk = 16384
    for begin in range(0, N_NODES * N_NODES, chunk):
        end = min(begin + chunk, N_NODES * N_NODES)
        features, o, d, calendar, _ = continuous_inputs(
            store,
            static,
            starts,
            origin[:, begin:end],
            destination[:, begin:end],
            history_hours,
            device,
            continuous_mode,
            feature_ablation,
        )
        base_log = model_base_log(base_flat[:, begin:end], continuous_mode)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, fused_log = model(features, o, d, calendar, base_log)
        probability_parts.append(torch.sigmoid(logits).float().cpu().numpy())
        residual_parts.append((fused_log.float() - base_log).cpu().numpy())
    probability = np.concatenate(probability_parts, axis=1).reshape(
        len(starts), N_NODES, N_NODES
    )
    residual = np.concatenate(residual_parts, axis=1).reshape(
        len(starts), N_NODES, N_NODES
    )
    model_base = (
        np.zeros_like(base.float().cpu().numpy(), dtype=np.float32)
        if continuous_mode == "continuous_only"
        else base.float().cpu().numpy().astype(np.float32)
    )
    return (
        probability.astype(np.float32),
        model_base,
        residual.astype(np.float32),
    )


def update_metric_bins(probability, magnitude, target, bins, thresholds):
    """Accumulate sufficient statistics for every threshold in one pass."""
    p = probability.reshape(-1)
    y = target.reshape(-1).astype(np.float64, copy=False)
    m = magnitude.reshape(-1).astype(np.float64, copy=False)
    index = np.searchsorted(thresholds, p, side="right")
    length = len(thresholds) + 1
    bins["absolute_delta"] += np.bincount(
        index,
        weights=np.abs(m - y) - np.abs(y),
        minlength=length,
    )
    bins["squared_delta"] += np.bincount(
        index,
        weights=np.square(m - y) - np.square(y),
        minlength=length,
    )
    bins["prediction"] += np.bincount(index, weights=m, minlength=length)
    bins["prediction_squared"] += np.bincount(
        index, weights=np.square(m), minlength=length
    )
    bins["prediction_target"] += np.bincount(
        index, weights=m * y, minlength=length
    )


def active_threshold_sums(bins):
    """Convert probability bins into sums for prediction >= each threshold."""
    return np.cumsum(bins[..., :0:-1], axis=-1)[..., ::-1]


def validation_grid_metrics(metric_bins, target_sums, n):
    """Recover MAE, RMSE, and PCC for a thresholded prediction grid."""
    absolute = target_sums["absolute"] + active_threshold_sums(
        metric_bins["absolute_delta"]
    )
    squared = target_sums["squared"] + active_threshold_sums(
        metric_bins["squared_delta"]
    )
    prediction = active_threshold_sums(metric_bins["prediction"])
    prediction_squared = active_threshold_sums(metric_bins["prediction_squared"])
    prediction_target = active_threshold_sums(metric_bins["prediction_target"])
    denominator_n = max(n, 1)
    mae = absolute / denominator_n
    rmse = np.sqrt(np.maximum(squared / denominator_n, 0.0))
    prediction_centered = prediction_squared - np.square(prediction) / denominator_n
    target_centered = (
        target_sums["squared"]
        - target_sums["sum"] * target_sums["sum"] / denominator_n
    )
    covariance = (
        prediction_target - prediction * target_sums["sum"] / denominator_n
    )
    pcc_denominator = np.sqrt(
        np.maximum(prediction_centered, 0.0) * max(target_centered, 0.0)
    )
    pcc = np.divide(
        covariance,
        pcc_denominator,
        out=np.zeros_like(covariance),
        where=pcc_denominator > 0.0,
    )
    return mae, rmse, pcc


def event_test_edges_by_hour():
    """Return the formal H12 event-direct OD edges grouped by test hour."""
    if not EVENT_INDEX_PATH.exists():
        return {}
    index = np.load(EVENT_INDEX_PATH, mmap_mode="r")
    hour = np.asarray(index["sample_hour"], dtype=np.int32)
    edge = np.asarray(index["sample_edge"], dtype=np.int32)
    keep = hour >= VAL_END
    hour = hour[keep]
    edge = edge[keep]
    lookup = {}
    if len(hour):
        unique, starts, counts = np.unique(hour, return_index=True, return_counts=True)
        for value, begin, count in zip(unique, starts, counts):
            lookup[int(value)] = edge[int(begin) : int(begin + count)]
    return lookup


@torch.no_grad()
def evaluate_direct_count(
    model,
    backbone,
    store,
    normalized,
    static,
    discrete_static,
    args,
    device,
    mean,
    std,
):
    """Evaluate one-stage count output without activation probabilities."""
    residual_values = (
        np.asarray([args.fixed_residual_blend], dtype=np.float32)
        if args.fixed_residual_blend is not None
        else (
            np.asarray([1.0], dtype=np.float32)
            if args.continuous_mode == "continuous_only"
            else BLENDS
        )
    )
    validation_absolute = np.zeros(len(residual_values), dtype=np.float64)
    validation_n = 0
    validation_hours = 1008 if args.max_eval_hours is None else args.max_eval_hours
    for starts in iter_hour_batches("validation", 2, limit=validation_hours):
        _, base, residual = fusion_prediction(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            starts,
            args.history_hours,
            device,
            mean,
            std,
            args.continuous_mode,
            args.feature_ablation,
        )
        target = np.asarray(store.od[starts], dtype=np.float32)
        base_log = np.log1p(base)
        for blend_index, residual_blend in enumerate(residual_values):
            prediction = np.expm1(base_log + residual_blend * residual)
            validation_absolute[blend_index] += float(
                np.abs(prediction.astype(np.float64) - target).sum()
            )
        validation_n += target.size
    validation_mae = validation_absolute / max(validation_n, 1)
    best_index = int(np.argmin(validation_mae))
    residual_blend = float(residual_values[best_index])

    metrics = StreamingMetrics()
    event_metrics = StreamingMetrics()
    low_flow_metrics = StreamingMetrics()
    high_flow_metrics = StreamingMetrics()
    event_edges = event_test_edges_by_hour() if args.history_hours == 12 else {}
    test_hour_stats = []
    test_hours = 1032 if args.max_eval_hours is None else args.max_eval_hours
    for starts in iter_hour_batches("test", 2, limit=test_hours):
        _, base, residual = fusion_prediction(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            starts,
            args.history_hours,
            device,
            mean,
            std,
            args.continuous_mode,
            args.feature_ablation,
        )
        prediction = np.expm1(
            np.log1p(base) + residual_blend * residual
        ).astype(np.float32)
        target = np.asarray(store.od[starts], dtype=np.float32)
        metrics.update(torch.from_numpy(prediction), torch.from_numpy(target))
        low_flow = (target > 0) & (target < 5)
        high_flow = target >= 5
        if bool(low_flow.any()):
            low_flow_metrics.update(
                torch.from_numpy(prediction[low_flow]),
                torch.from_numpy(target[low_flow]),
            )
        if bool(high_flow.any()):
            high_flow_metrics.update(
                torch.from_numpy(prediction[high_flow]),
                torch.from_numpy(target[high_flow]),
            )
        error = prediction.astype(np.float64) - target.astype(np.float64)
        for local_index, hour in enumerate(starts):
            selected_edges = event_edges.get(int(hour))
            if selected_edges is not None and len(selected_edges):
                flat_prediction = prediction[local_index].reshape(-1)
                flat_target = target[local_index].reshape(-1)
                event_metrics.update(
                    torch.from_numpy(flat_prediction[selected_edges]),
                    torch.from_numpy(flat_target[selected_edges]),
                )
            hourly_error = error[local_index].reshape(-1)
            test_hour_stats.append(
                (
                    int(hour),
                    int(hourly_error.size),
                    float(np.abs(hourly_error).sum()),
                    float(np.square(hourly_error).sum()),
                )
            )
    return {
        "activation_blend": None,
        "continuous_residual_blend": residual_blend,
        "threshold": None,
        "validation_mae": float(validation_mae[best_index]),
        "test": metrics.compute(),
        "event_direct_test": event_metrics.compute() if event_metrics.n else None,
        "flow_strata": {
            "low_positive_1_to_4": low_flow_metrics.compute(),
            "high_positive_ge_5": high_flow_metrics.compute(),
        },
        "test_hour_stats": test_hour_stats,
    }


@torch.no_grad()
def evaluate(
    model,
    backbone,
    store,
    normalized,
    static,
    discrete_static,
    args,
    device,
    mean,
    std,
):
    if args.prediction_mode == "direct":
        return evaluate_direct_count(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            args,
            device,
            mean,
            std,
        )
    gate_dir = EXPERIMENT_ROOT / "runs" / "activation" / (
        "h%d_seed0" % args.history_hours
    )
    shared_validation = np.load(
        gate_dir / "validation_probability.npy", mmap_mode="r"
    )
    shared_test = np.load(gate_dir / "test_probability.npy", mmap_mode="r")
    activation_values = (
        np.asarray([args.fixed_activation_blend], dtype=np.float32)
        if args.fixed_activation_blend is not None
        else (
            np.asarray([1.0], dtype=np.float32)
            if args.continuous_mode == "continuous_only"
            else BLENDS
        )
    )
    residual_values = (
        np.asarray([args.fixed_residual_blend], dtype=np.float32)
        if args.fixed_residual_blend is not None
        else (
            np.asarray([1.0], dtype=np.float32)
            if args.continuous_mode == "continuous_only"
            else BLENDS
        )
    )
    threshold_values = (
        np.asarray([args.fixed_threshold], dtype=np.float32)
        if args.fixed_threshold is not None
        else THRESHOLDS
    )
    grid_shape = (
        len(activation_values),
        len(residual_values),
        len(threshold_values) + 1,
    )
    metric_bins = {
        name: np.zeros(grid_shape, dtype=np.float64)
        for name in (
            "absolute_delta",
            "squared_delta",
            "prediction",
            "prediction_squared",
            "prediction_target",
        )
    }
    target_sums = {"absolute": 0.0, "squared": 0.0, "sum": 0.0}
    n = 0
    validation_hours = 1008 if args.max_eval_hours is None else args.max_eval_hours
    for starts in iter_hour_batches("validation", 2, limit=validation_hours):
        continuous_probability, base, residual = fusion_prediction(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            starts,
            args.history_hours,
            device,
            mean,
            std,
            args.continuous_mode,
            args.feature_ablation,
        )
        shared = np.asarray(shared_validation[starts - 6720], dtype=np.float32)
        target = np.asarray(store.od[starts], dtype=np.float32)
        target64 = target.astype(np.float64, copy=False)
        target_sums["absolute"] += float(np.abs(target64).sum())
        target_sums["squared"] += float(np.square(target64).sum())
        target_sums["sum"] += float(target64.sum())
        n += target.size
        base_log = np.log1p(base)
        for activation_index, activation_blend in enumerate(activation_values):
            probability = (
                (1.0 - activation_blend) * shared
                + activation_blend * continuous_probability
            )
            for residual_index, residual_blend in enumerate(residual_values):
                magnitude = np.expm1(base_log + residual_blend * residual)
                update_metric_bins(
                    probability,
                    magnitude,
                    target,
                    {
                        name: values[activation_index, residual_index]
                        for name, values in metric_bins.items()
                    },
                    threshold_values,
                )
    validation_mae, validation_rmse, validation_pcc = validation_grid_metrics(
        metric_bins, target_sums, n
    )
    if args.evaluation_selection == "balanced" and validation_mae.size > 1:
        activation_reference = int(np.argmin(np.abs(activation_values)))
        residual_reference = int(np.argmin(np.abs(residual_values)))
        reference_candidates = validation_mae[
            activation_reference, residual_reference
        ]
        threshold_reference = int(np.argmin(reference_candidates))
        reference = {
            "activation_blend": float(activation_values[activation_reference]),
            "residual_blend": float(residual_values[residual_reference]),
            "threshold": float(threshold_values[threshold_reference]),
            "mae": float(
                validation_mae[
                    activation_reference, residual_reference, threshold_reference
                ]
            ),
            "rmse": float(
                validation_rmse[
                    activation_reference, residual_reference, threshold_reference
                ]
            ),
            "pcc": float(
                validation_pcc[
                    activation_reference, residual_reference, threshold_reference
                ]
            ),
        }
        if min(reference["mae"], reference["rmse"], reference["pcc"]) <= 0.0:
            raise ValueError("Balanced validation selection requires positive references")
        validation_score = (
            validation_mae / reference["mae"]
            + validation_rmse / reference["rmse"]
            - validation_pcc / reference["pcc"]
        )
        ordering = np.lexsort(
            (
                -validation_pcc.reshape(-1),
                validation_rmse.reshape(-1),
                validation_mae.reshape(-1),
                validation_score.reshape(-1),
            )
        )
        best = np.unravel_index(int(ordering[0]), validation_mae.shape)
        selection_score = float(validation_score[best])
        selection_protocol = "equal normalized validation MAE/RMSE/PCC score"
    else:
        ordering = np.lexsort(
            (
                -validation_pcc.reshape(-1),
                validation_rmse.reshape(-1),
                validation_mae.reshape(-1),
            )
        )
        best = np.unravel_index(int(ordering[0]), validation_mae.shape)
        reference = None
        selection_score = float(validation_mae[best])
        selection_protocol = "minimum validation MAE; RMSE/PCC break exact ties"
    activation_blend = float(activation_values[best[0]])
    residual_blend = float(residual_values[best[1]])
    threshold = float(threshold_values[best[2]])
    metrics = StreamingMetrics()
    low_flow_metrics = StreamingMetrics()
    high_flow_metrics = StreamingMetrics()
    test_hour_stats = []
    test_hours = 1032 if args.max_eval_hours is None else args.max_eval_hours
    for starts in iter_hour_batches("test", 2, limit=test_hours):
        continuous_probability, base, residual = fusion_prediction(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            starts,
            args.history_hours,
            device,
            mean,
            std,
            args.continuous_mode,
            args.feature_ablation,
        )
        shared = np.asarray(shared_test[starts - VAL_END], dtype=np.float32)
        probability = (
            (1.0 - activation_blend) * shared
            + activation_blend * continuous_probability
        )
        magnitude = np.expm1(np.log1p(base) + residual_blend * residual)
        prediction = np.where(probability >= threshold, magnitude, 0.0).astype(np.float32)
        target = np.asarray(store.od[starts], dtype=np.float32)
        metrics.update(
            torch.from_numpy(prediction),
            torch.from_numpy(target),
        )
        low_flow = (target > 0) & (target < 5)
        high_flow = target >= 5
        if bool(low_flow.any()):
            low_flow_metrics.update(
                torch.from_numpy(prediction[low_flow]),
                torch.from_numpy(target[low_flow]),
            )
        if bool(high_flow.any()):
            high_flow_metrics.update(
                torch.from_numpy(prediction[high_flow]),
                torch.from_numpy(target[high_flow]),
            )
        error = prediction.astype(np.float64) - target.astype(np.float64)
        for local_index, hour in enumerate(starts):
            hourly_error = error[local_index].reshape(-1)
            test_hour_stats.append(
                (
                    int(hour),
                    int(hourly_error.size),
                    float(np.abs(hourly_error).sum()),
                    float(np.square(hourly_error).sum()),
                )
            )
    return {
        "activation_blend": activation_blend,
        "continuous_residual_blend": residual_blend,
        "threshold": threshold,
        "validation_mae": float(validation_mae[best]),
        "validation_rmse": float(validation_rmse[best]),
        "validation_pcc": float(validation_pcc[best]),
        "validation_selection_score": selection_score,
        "validation_selection": selection_protocol,
        "validation_normalization_reference": reference,
        "test": metrics.compute(),
        "flow_strata": {
            "low_positive_1_to_4": low_flow_metrics.compute(),
            "high_positive_ge_5": high_flow_metrics.compute(),
        },
        "test_hour_stats": test_hour_stats,
    }


def main() -> None:
    args = parse_args()
    if args.backbone_variant is None:
        args.backbone_variant = (
            "plain" if args.prediction_mode == "direct" else "hurdle"
        )
    if args.prediction_mode == "direct" and args.fixed_activation_blend is not None:
        raise ValueError("Direct count prediction does not use activation blending")
    if args.prediction_mode == "direct" and args.fixed_threshold is not None:
        raise ValueError("Direct count prediction does not use an activation threshold")
    for name in (
        "fixed_activation_blend",
        "fixed_residual_blend",
        "fixed_threshold",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.smoke:
        args.epochs = 2
        args.max_train_hours = 8
        args.max_eval_hours = 4
        args.uniform_edges = 1024
        args.positive_edges = 512
    device = require_cuda(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_suffix = (
        ("smoke_" if args.smoke else "formal_")
        + args.magnitude_loss
        + ("" if args.continuous_mode == "exact" else f"_{args.continuous_mode}")
        if args.run_name is None
        else args.run_name
    )
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "continuous_fusion_od"
        / (
            "h%d_seed%d_%s"
            % (
                args.history_hours,
                args.seed,
                run_suffix,
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = load_normalized()
    mean, std = load_scaler()
    store = ContinuousFeatureStore()
    static_payload = load_static()
    static = torch_static(static_payload, device)
    discrete_static = static_tensors(static_payload, device)
    backbone = create_model("stidgcn", args.history_hours, device=device).to(device)
    backbone_path = (
        EXPERIMENT_ROOT
        / "runs"
        / "stidgcn"
        / (
            "h%d_%s_seed%d_formal"
            % (args.history_hours, args.backbone_variant, args.seed)
        )
        / "best_model.pt"
    )
    if not backbone_path.exists():
        raise FileNotFoundError(
            "Train the STIDGCN-%s backbone first: %s"
            % (args.backbone_variant, backbone_path)
        )
    backbone_payload = torch.load(backbone_path, map_location=device)
    backbone.load_state_dict(backbone_payload["model"])
    if args.joint_backbone:
        backbone.train()
        for parameter in backbone.parameters():
            parameter.requires_grad = True
    else:
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    model = ContinuousHybridOD(
        nodes=N_NODES,
        zero_init_residual=not args.no_zero_init,
        feature_dim=feature_dimension(args.continuous_mode),
        context_ablation=args.context_ablation,
    ).to(device)
    if args.prediction_mode == "direct":
        for parameter in model.activation_head.parameters():
            parameter.requires_grad = False
    optimized_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.joint_backbone:
        optimized_parameters.extend(backbone.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = output_dir / "best_model.pt"
    latest_checkpoint = output_dir / "latest_training.pt"
    history = []
    best = math.inf
    stale = 0
    start_epoch = 1
    elapsed_before_resume = 0.0
    if (
        args.resume
        and args.evaluation_only_checkpoint is None
        and latest_checkpoint.exists()
    ):
        resume_payload = torch.load(latest_checkpoint, map_location=device)
        expected = {
            "history_hours": args.history_hours,
            "seed": args.seed,
            "continuous_mode": args.continuous_mode,
            "feature_ablation": args.feature_ablation,
            "context_ablation": args.context_ablation,
            "joint_backbone": bool(args.joint_backbone),
            "no_zero_init": bool(args.no_zero_init),
            "magnitude_loss": args.magnitude_loss,
            "prediction_mode": args.prediction_mode,
            "backbone_variant": args.backbone_variant,
        }
        recorded = resume_payload.get("configuration", {})
        if recorded != expected:
            raise RuntimeError(
                "Refusing to resume a checkpoint with a different configuration: "
                f"expected={expected}, recorded={recorded}"
            )
        model.load_state_dict(resume_payload["model"])
        if args.joint_backbone and resume_payload.get("backbone") is not None:
            backbone.load_state_dict(resume_payload["backbone"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
        history = list(resume_payload["history"])
        best = float(resume_payload["best"])
        stale = int(resume_payload["stale"])
        start_epoch = int(resume_payload["epoch"]) + 1
        elapsed_before_resume = float(
            resume_payload.get("elapsed_training_seconds", 0.0)
        )
        print(
            json.dumps(
                {
                    "status": "resumed",
                    "epoch": start_epoch - 1,
                    "best_validation_loss": best,
                    "stale_epochs": stale,
                }
            ),
            flush=True,
        )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    training_epochs = 0 if args.evaluation_only_checkpoint is not None else args.epochs
    for epoch in range(start_epoch, training_epochs + 1):
        model.train()
        if args.joint_backbone:
            backbone.train()
        rng = np.random.default_rng(args.seed * 10000 + epoch)
        total = 0.0
        seen = 0
        epoch_started = time.time()
        for starts in iter_hour_batches(
            "train",
            args.hour_batch,
            shuffle=True,
            seed=args.seed + epoch,
            limit=args.max_train_hours,
        ):
            origin, destination = sample_edges(
                store,
                starts,
                args.uniform_edges,
                args.positive_edges,
                rng,
            )
            base = discrete_prediction(
                backbone,
                normalized,
                starts,
                args.history_hours,
                discrete_static,
                device,
                mean,
                std,
            )
            features, o, d, calendar, target = continuous_inputs(
                store,
                static,
                starts,
                origin,
                destination,
                args.history_hours,
                device,
                args.continuous_mode,
                args.feature_ablation,
            )
            base_selected = selected_base(base, origin, destination, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, magnitude_log = model(
                    features,
                    o,
                    d,
                    calendar,
                    model_base_log(base_selected, args.continuous_mode),
                )
                if args.prediction_mode == "direct":
                    direct_count = torch.expm1(
                        magnitude_log[:, : args.uniform_edges]
                    )
                    loss = torch.abs(
                        direct_count - target[:, : args.uniform_edges]
                    ).mean()
                elif args.magnitude_loss == "decision":
                    activation = F.binary_cross_entropy_with_logits(
                        logits[:, : args.uniform_edges],
                        (target[:, : args.uniform_edges] > 0).float(),
                    )
                    probability = torch.sigmoid(logits[:, : args.uniform_edges])
                    raw_magnitude = torch.expm1(
                        magnitude_log[:, : args.uniform_edges]
                    )
                    uniform_target = target[:, : args.uniform_edges]
                    expected_absolute = (
                        probability * torch.abs(raw_magnitude - uniform_target)
                        + (1.0 - probability) * torch.abs(uniform_target)
                    ).mean()
                    positive = target > 0
                    magnitude = F.smooth_l1_loss(
                        torch.expm1(magnitude_log[positive]),
                        target[positive],
                        beta=1.0,
                    )
                    loss = expected_absolute + 0.1 * activation + 0.5 * magnitude
                elif args.magnitude_loss == "raw":
                    activation = F.binary_cross_entropy_with_logits(
                        logits[:, : args.uniform_edges],
                        (target[:, : args.uniform_edges] > 0).float(),
                    )
                    positive = target > 0
                    magnitude = F.smooth_l1_loss(
                        torch.expm1(magnitude_log[positive]),
                        target[positive],
                        beta=1.0,
                    )
                    loss = 0.3 * activation + magnitude
                else:
                    activation = F.binary_cross_entropy_with_logits(
                        logits[:, : args.uniform_edges],
                        (target[:, : args.uniform_edges] > 0).float(),
                    )
                    positive = target > 0
                    magnitude = F.smooth_l1_loss(
                        magnitude_log[positive], torch.log1p(target[positive])
                    )
                    loss = activation + magnitude
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if args.joint_backbone:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(starts)
            seen += len(starts)
        validation = validation_loss(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            args,
            device,
            mean,
            std,
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_loss": validation,
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if validation < best - 1e-6:
            best = validation
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": (
                        backbone.state_dict() if args.joint_backbone else None
                    ),
                    "epoch": epoch,
                    "validation": validation,
                },
                checkpoint,
            )
        else:
            stale += 1
        torch.save(
            {
                "model": model.state_dict(),
                "backbone": (
                    backbone.state_dict() if args.joint_backbone else None
                ),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "history": history,
                "elapsed_training_seconds": (
                    elapsed_before_resume + time.time() - started
                ),
                "configuration": {
                    "history_hours": args.history_hours,
                    "seed": args.seed,
                    "continuous_mode": args.continuous_mode,
                    "feature_ablation": args.feature_ablation,
                    "context_ablation": args.context_ablation,
                    "joint_backbone": bool(args.joint_backbone),
                    "no_zero_init": bool(args.no_zero_init),
                    "magnitude_loss": args.magnitude_loss,
                    "prediction_mode": args.prediction_mode,
                    "backbone_variant": args.backbone_variant,
                },
            },
            latest_checkpoint,
        )
        if stale >= args.patience:
            break
    training_seconds = (
        0.0
        if args.evaluation_only_checkpoint is not None
        else elapsed_before_resume + time.time() - started
    )
    payload_path = (
        args.evaluation_only_checkpoint
        if args.evaluation_only_checkpoint is not None
        else checkpoint
    )
    if not payload_path.is_absolute():
        payload_path = EXPERIMENT_ROOT / payload_path
    payload = torch.load(payload_path, map_location=device)
    model.load_state_dict(payload["model"])
    if args.joint_backbone and payload.get("backbone") is not None:
        backbone.load_state_dict(payload["backbone"])
    backbone.eval()
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    evaluation = evaluate(
        model,
        backbone,
        store,
        normalized,
        static,
        discrete_static,
        args,
        device,
        mean,
        std,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    test_hour_stats = np.asarray(
        evaluation.pop("test_hour_stats"), dtype=np.float64
    )
    np.savez_compressed(
        output_dir / "test_hour_stats.npz",
        hour=test_hour_stats[:, 0].astype(np.int32),
        count=test_hour_stats[:, 1].astype(np.int64),
        absolute_error_sum=test_hour_stats[:, 2],
        squared_error_sum=test_hour_stats[:, 3],
    )
    summary = {
        "model": "ContinuousFusionOD",
        "history_hours": args.history_hours,
        "seed": args.seed,
        "best_epoch": int(payload["epoch"]),
        "selection_loss": float(payload["validation"]),
        "frozen_discrete_branch": (
            None
            if args.joint_backbone
            else "STIDGCN-%s" % args.backbone_variant
        ),
        "continuous_branch": (
            "resolution-adaptive short-term order-state residual"
            if args.continuous_mode == "hierarchical"
            else "marked-event intensity and recency residual"
        ),
        "continuous_mode": args.continuous_mode,
        "feature_dimension": feature_dimension(args.continuous_mode),
        "feature_ablation": args.feature_ablation,
        "context_ablation": args.context_ablation,
        "magnitude_loss": args.magnitude_loss,
        "direct_loss": (
            "raw-count MAE on uniformly sampled OD cells"
            if args.prediction_mode == "direct"
            else None
        ),
        "prediction_mode": args.prediction_mode,
        "backbone_variant": args.backbone_variant,
        "uses_activation_probability": args.prediction_mode == "hurdle",
        "uses_activation_threshold": args.prediction_mode == "hurdle",
        "fallback_in_validation_grid": (
            args.fixed_activation_blend is None
            and args.fixed_residual_blend is None
            and args.continuous_mode != "continuous_only"
        ),
        "joint_backbone_training": bool(args.joint_backbone),
        "zero_initialized_magnitude_residual": not args.no_zero_init,
        "fixed_evaluation": {
            "activation_blend": args.fixed_activation_blend,
            "residual_blend": args.fixed_residual_blend,
            "threshold": args.fixed_threshold,
        },
        "evaluation_selection": args.evaluation_selection,
        "parameter_count_trainable": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            + (
                sum(p.numel() for p in backbone.parameters())
                if args.joint_backbone
                else 0
            )
        ),
        "parameter_count_total": int(
            sum(p.numel() for p in model.parameters())
            + sum(p.numel() for p in backbone.parameters())
        ),
        "device": args.device,
        "strict_window": True,
        "full_run": not args.smoke and args.max_train_hours is None and args.max_eval_hours is None,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_test_hours": int(len(test_hour_stats)),
        "inference_hours_per_second": (
            float(len(test_hour_stats) / inference_seconds)
            if inference_seconds > 0
            else None
        ),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "evaluation_only_checkpoint": (
            str(payload_path) if args.evaluation_only_checkpoint is not None else None
        ),
        "evaluation": evaluation,
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
