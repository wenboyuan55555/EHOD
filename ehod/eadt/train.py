from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, TRAIN_END, VAL_END
from ehod.common.device import require_cuda
from ehod.eadt.model import DirectCountEventTransitionAdapter, TaxiAwareEventTransitionAdapter
from ehod.evaluation.metrics import StreamingMetrics
from baselines.event._shared.train_with_hcf import INDEX_PATH, logit_numpy


HISTORY_HOURS = 12
BACKBONES = (
    "count_fusion",
    "continuous_fusion",
    "stidgcn",
    "stid",
    "cmod",
    "bootcf",
    "stpro",
    "odmixer",
    "odpf_no_event",
    "stan",
    "ha",
    "xgboost",
    "lstm",
    "duet_mipf",
    "patchstg_mipf",
    "stdn_mipf",
    "lightst_mipf",
    "strep_mipf",
)
SLOT_ROOT = EXPERIMENT_ROOT / "cache/event_transition_slots"
SLOT_NAMES = (
    "event_type",
    "venue",
    "role",
    "scale",
    "tau_start",
    "tau_end",
    "arrival_distance_km",
    "departure_distance_km",
)


class EventTransitionSlotStore:
    def __init__(self) -> None:
        missing = [name for name in SLOT_NAMES if not (SLOT_ROOT / (name + ".npy")).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing event-transition slots %s; run python -m "
                "ehod.eadt.prepare_slots first" % missing
            )
        self.arrays = {
            name: np.load(SLOT_ROOT / (name + ".npy"), mmap_mode="r")
            for name in SLOT_NAMES
        }

    def batch(
        self,
        indices: np.ndarray,
        device: torch.device,
        slot_mode: str = "factual",
        permutation: np.ndarray | None = None,
        scale_mode: str = "observed",
    ):
        values = []
        for name in SLOT_NAMES:
            dtype = np.int64 if name in {"event_type", "venue", "role"} else np.float32
            source = indices
            if slot_mode == "shuffle_time" and name in {"tau_start", "tau_end"}:
                if permutation is None:
                    raise ValueError("shuffle_time requires a fixed permutation")
                source = permutation[indices]
            values.append(
                torch.from_numpy(np.asarray(self.arrays[name][source], dtype=dtype)).to(device)
            )
        if scale_mode == "unit":
            event_type = values[SLOT_NAMES.index("event_type")]
            values[SLOT_NAMES.index("scale")] = (event_type >= 0).to(torch.float32)
        return values

    def contains(self, name: str, value: int) -> np.ndarray:
        if name not in {"event_type", "venue"}:
            raise ValueError("Unsupported holdout field %r" % name)
        return np.any(np.asarray(self.arrays[name]) == int(value), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES, default="count_fusion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-seed", type=int, default=0)
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=None,
        help="Validation-selected hurdle threshold for an external backbone.",
    )
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Optional event-backbone cache prefix; defaults to --backbone.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--expectation-cutoff", type=float, default=0.10)
    parser.add_argument("--positive-weight", type=float, default=0.25)
    parser.add_argument("--square-weight", type=float, default=0.03)
    parser.add_argument("--corr-weight", type=float, default=0.05)
    parser.add_argument("--activation-weight", type=float, default=0.05)
    parser.add_argument("--correction-weight", type=float, default=0.002)
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--event-rank", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--activation-cap", type=float, default=2.0)
    parser.add_argument("--magnitude-cap", type=float, default=1.25)
    parser.add_argument(
        "--loss-preset",
        choices=(
            "validated_three_term",
            "original",
            "hurdle",
            "hurdle_flow",
            "custom",
        ),
        default="validated_three_term",
    )
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument(
        "--anchor-mode", choices=("dual", "start_only"), default="dual"
    )
    parser.add_argument(
        "--kernel-mode", choices=("cauchy", "gaussian"), default="cauchy"
    )
    parser.add_argument(
        "--exposure-mode",
        choices=("directional", "symmetric"),
        default="directional",
    )
    parser.add_argument(
        "--susceptibility-mode",
        choices=("od_specific", "base_only", "shared"),
        default="od_specific",
    )
    parser.add_argument(
        "--prediction-mode", choices=("hurdle", "direct"), default="hurdle"
    )
    parser.add_argument(
        "--head-mode", choices=("both", "activation_only", "magnitude_only"),
        default="both",
    )
    parser.add_argument(
        "--base-state-mode", choices=("full", "no_base"), default="full"
    )
    parser.add_argument(
        "--event-slot-mode", choices=("factual", "shuffle_time"), default="factual"
    )
    parser.add_argument(
        "--scale-mode", choices=("observed", "unit"), default="observed"
    )
    parser.add_argument("--type-mode", choices=("id", "shared"), default="id")
    parser.add_argument("--venue-mode", choices=("id", "shared"), default="id")
    parser.add_argument("--holdout-type", type=int, default=None)
    parser.add_argument("--holdout-venue", type=int, default=None)
    parser.add_argument("--run-name", default="taxi_aware_eadt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional exact output directory (used by packaged supplemental baselines).",
    )
    parser.add_argument(
        "--selection",
        choices=("balanced", "mae"),
        default="balanced",
        help="Validation-only checkpoint rule; balanced gives MAE/RMSE/PCC equal normalized weight.",
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--evaluation-only-checkpoint",
        type=Path,
        default=None,
        help="Evaluate one validation-selected checkpoint without retraining.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def loss_weights(args: argparse.Namespace) -> dict:
    if args.loss_preset == "validated_three_term":
        # Selected once on NYC validation, seed 0, from the preregistered 5x5
        # positive/activation grid using the existing normalized
        # MAE+RMSE-PCC checkpoint score.  Frozen everywhere else.
        return {
            "flow": 1.0,
            "positive": 0.25,
            "rmse": 0.0,
            "correlation": 0.0,
            "activation": 0.0125,
            "correction": 0.0,
        }
    if args.loss_preset == "original":
        return {
            "flow": 1.0,
            "positive": float(args.positive_weight),
            "rmse": float(args.square_weight),
            "correlation": float(args.corr_weight),
            "activation": float(args.activation_weight),
            "correction": float(args.correction_weight),
        }
    if args.loss_preset == "hurdle":
        return {
            "flow": 0.0,
            "positive": 1.0,
            "rmse": 0.0,
            "correlation": 0.0,
            "activation": 1.0,
            "correction": 0.002,
        }
    if args.loss_preset == "hurdle_flow":
        return {
            "flow": 1.0,
            "positive": 0.25,
            "rmse": 0.0,
            "correlation": 0.0,
            "activation": 0.05,
            "correction": 0.002,
        }
    return {
        "flow": float(args.flow_weight),
        "positive": float(args.positive_weight),
        "rmse": float(args.square_weight),
        "correlation": float(args.corr_weight),
        "activation": float(args.activation_weight),
        "correction": float(args.correction_weight),
    }


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = prediction.float() - prediction.float().mean()
    y = target.float() - target.float().mean()
    denominator = torch.sqrt(
        x.square().sum().clamp_min(1.0e-6)
        * y.square().sum().clamp_min(1.0e-6)
    )
    return 1.0 - (x * y).sum() / denominator


def metric_dict(prediction: np.ndarray, target: np.ndarray) -> dict:
    metrics = StreamingMetrics()
    metrics.update(
        torch.from_numpy(np.asarray(prediction, dtype=np.float32)),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
    )
    return metrics.compute()


def numpy_metrics(prediction: np.ndarray, target: np.ndarray):
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    error = prediction - target
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    x = prediction - prediction.mean()
    y = target - target.mean()
    denominator = math.sqrt(float(np.square(x).sum()) * float(np.square(y).sum()))
    pcc = float((x * y).sum() / denominator) if denominator > 0 else 0.0
    return mae, rmse, pcc


def expected_prediction(logit: np.ndarray, logmag: np.ndarray, cutoff: float):
    probability = 1.0 / (1.0 + np.exp(-logit))
    magnitude = np.expm1(np.maximum(logmag, 0.0))
    return np.where(probability >= float(cutoff), probability * magnitude, 0.0).astype(
        np.float32
    )


def splitwise_permutation(hour: np.ndarray, seed: int) -> np.ndarray:
    """Create a deterministic within-split permutation for placebo controls."""
    permutation = np.arange(len(hour), dtype=np.int64)
    rng = np.random.default_rng(seed)
    split_masks = (
        hour < TRAIN_END,
        (hour >= TRAIN_END) & (hour < VAL_END),
        hour >= VAL_END,
    )
    for mask in split_masks:
        selected = np.flatnonzero(mask)
        permutation[selected] = rng.permutation(selected)
    return permutation


def base_result_path(backbone: str, seed: int):
    if backbone == "continuous_fusion":
        return (
            EXPERIMENT_ROOT
            / "runs/continuous_fusion_od"
            / ("h12_seed%d_formal_decision" % seed)
            / "result.json"
        )
    if backbone == "count_fusion":
        return (
            EXPERIMENT_ROOT
            / "runs/continuous_fusion_od"
            / ("h12_seed%d_hierarchical_no_stats_v1" % seed)
            / "result.json"
        )
    if backbone in {"cmod", "bootcf"}:
        return (
            EXPERIMENT_ROOT
            / "runs"
            / backbone
            / ("h12_imported_seed%d_formal" % seed)
            / "result.json"
        )
    return (
        EXPERIMENT_ROOT
        / "runs"
        / backbone
        / ("h12_hurdle_seed%d_formal" % seed)
        / "result.json"
    )


def base_threshold(backbone: str, seed: int) -> float:
    payload = json.loads(base_result_path(backbone, seed).read_text(encoding="utf-8"))
    return float(payload["evaluation"]["threshold"])


@torch.no_grad()
def adapter_outputs(
    model,
    indices,
    slots,
    origin,
    destination,
    base_logit,
    base_logmag,
    device,
    batch_size,
    slot_mode="factual",
    permutation=None,
    scale_mode="observed",
):
    model.eval()
    logits = np.empty(len(indices), dtype=np.float32)
    logmags = np.empty(len(indices), dtype=np.float32)
    activation_delta = np.empty(len(indices), dtype=np.float32)
    magnitude_delta = np.empty(len(indices), dtype=np.float32)
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        event = slots.batch(
            selected,
            device,
            slot_mode=slot_mode,
            permutation=permutation,
            scale_mode=scale_mode,
        )
        o = torch.from_numpy(origin[selected]).long().to(device)
        d = torch.from_numpy(destination[selected]).long().to(device)
        base_l = torch.from_numpy(base_logit[selected]).to(device)
        base_m = torch.from_numpy(base_logmag[selected]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            value_l, value_m, value_da, value_dm = model(
                *event, o, d, base_l, base_m
            )
        size = len(selected)
        logits[begin : begin + size] = value_l.float().cpu().numpy()
        logmags[begin : begin + size] = value_m.float().cpu().numpy()
        activation_delta[begin : begin + size] = value_da.float().cpu().numpy()
        magnitude_delta[begin : begin + size] = value_dm.float().cpu().numpy()
    return logits, logmags, activation_delta, magnitude_delta


@torch.no_grad()
def direct_adapter_outputs(
    model,
    indices,
    slots,
    origin,
    destination,
    base_count,
    device,
    batch_size,
    slot_mode="factual",
    permutation=None,
    scale_mode="observed",
):
    model.eval()
    prediction = np.empty(len(indices), dtype=np.float32)
    count_delta = np.empty(len(indices), dtype=np.float32)
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        event = slots.batch(
            selected,
            device,
            slot_mode=slot_mode,
            permutation=permutation,
            scale_mode=scale_mode,
        )
        o = torch.from_numpy(origin[selected]).long().to(device)
        d = torch.from_numpy(destination[selected]).long().to(device)
        base = torch.from_numpy(base_count[selected]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            value, value_delta = model(*event, o, d, base)
        size = len(selected)
        prediction[begin : begin + size] = value.float().cpu().numpy()
        count_delta[begin : begin + size] = value_delta.float().cpu().numpy()
    return prediction, count_delta


def save_predictions(
    path,
    indices,
    hour,
    edge,
    target,
    probability,
    magnitude,
    base_logit,
    base_logmag,
    event_logit,
    event_logmag,
    activation_delta,
    magnitude_delta,
    cutoff,
    threshold,
):
    base_hard = np.where(
        probability[indices] >= threshold, magnitude[indices], 0.0
    ).astype(np.float32)
    base_expected = expected_prediction(
        base_logit[indices], base_logmag[indices], cutoff
    )
    event_expected = expected_prediction(event_logit, event_logmag, cutoff)
    np.savez_compressed(
        path,
        sample_index=indices.astype(np.int64),
        sample_hour=hour[indices].astype(np.int32),
        sample_edge=edge[indices].astype(np.int32),
        target=target[indices].astype(np.float32),
        base_prediction=base_hard,
        base_expected_prediction=base_expected,
        event_prediction=event_expected,
        event_logit=event_logit.astype(np.float32),
        event_log_magnitude=event_logmag.astype(np.float32),
        activation_delta=activation_delta.astype(np.float32),
        magnitude_delta=magnitude_delta.astype(np.float32),
    )
    return base_hard, base_expected, event_expected


def save_direct_predictions(
    path,
    indices,
    hour,
    edge,
    target,
    probability,
    magnitude,
    event_prediction,
    count_delta,
    threshold,
):
    base_hard = np.where(
        probability[indices] >= threshold, magnitude[indices], 0.0
    ).astype(np.float32)
    base_continuous = (probability[indices] * magnitude[indices]).astype(np.float32)
    np.savez_compressed(
        path,
        sample_index=indices.astype(np.int64),
        sample_hour=hour[indices].astype(np.int32),
        sample_edge=edge[indices].astype(np.int32),
        target=target[indices].astype(np.float32),
        base_prediction=base_hard,
        base_expected_prediction=base_continuous,
        event_prediction=event_prediction.astype(np.float32),
        count_delta=count_delta.astype(np.float32),
    )
    return base_hard, base_continuous, event_prediction


def main() -> None:
    args = parse_args()
    if args.holdout_type is not None and args.holdout_venue is not None:
        raise ValueError("Use at most one of --holdout-type and --holdout-venue")
    if args.validation_only and args.evaluation_only_checkpoint is not None:
        raise ValueError(
            "--validation-only and --evaluation-only-checkpoint are mutually exclusive"
        )
    device = require_cuda(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    index = np.load(INDEX_PATH, mmap_mode="r")
    hour = np.asarray(index["sample_hour"], dtype=np.int32)
    edge = np.asarray(index["sample_edge"], dtype=np.int32)
    target = np.asarray(index["sample_target"], dtype=np.float32)
    origin = (edge // N_NODES).astype(np.int64)
    destination = (edge % N_NODES).astype(np.int64)
    slots = EventTransitionSlotStore()

    cache_dir = EXPERIMENT_ROOT / "cache/event_backbones/h12"
    cache_key = args.cache_key or args.backbone
    probability_path = cache_dir / (cache_key + "_probability.npy")
    if not probability_path.exists():
        probability_path = cache_dir / "strict_activation_probability.npy"
    probability = np.asarray(np.load(probability_path, mmap_mode="r"), dtype=np.float32)
    magnitude = np.asarray(
        np.load(cache_dir / (cache_key + "_magnitude.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    base_logit = logit_numpy(probability)
    base_logmag = np.log1p(np.maximum(magnitude, 0.0)).astype(np.float32)
    base_continuous_count = (probability * magnitude).astype(np.float32)

    train_index = np.flatnonzero(hour < TRAIN_END)
    validation_index = np.flatnonzero((hour >= TRAIN_END) & (hour < VAL_END))
    test_index = np.flatnonzero(hour >= VAL_END)
    evaluation_scope = "all event-indexed test samples"
    holdout = None
    if args.holdout_type is not None:
        holdout = slots.contains("event_type", args.holdout_type)
        evaluation_scope = "unseen event type %d" % args.holdout_type
    elif args.holdout_venue is not None:
        holdout = slots.contains("venue", args.holdout_venue)
        evaluation_scope = "unseen venue %d" % args.holdout_venue
    if holdout is not None:
        train_index = train_index[~holdout[train_index]]
        validation_index = validation_index[~holdout[validation_index]]
        test_index = test_index[holdout[test_index]]
        if len(test_index) == 0:
            raise ValueError("Holdout produced no test samples: %s" % evaluation_scope)
    if args.max_train_samples is not None:
        train_index = train_index[: args.max_train_samples]
    if args.max_eval_samples is not None:
        validation_index = validation_index[: args.max_eval_samples]
        test_index = test_index[: args.max_eval_samples]

    permutation = (
        splitwise_permutation(hour, args.seed + 104729)
        if args.event_slot_mode == "shuffle_time"
        else None
    )
    common_model_args = {
        "nodes": N_NODES,
        "node_dim": args.node_dim,
        "event_rank": args.event_rank,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "anchor_mode": args.anchor_mode,
        "kernel_mode": args.kernel_mode,
        "exposure_mode": args.exposure_mode,
        "susceptibility_mode": args.susceptibility_mode,
        "base_state_mode": args.base_state_mode,
        "type_mode": args.type_mode,
        "venue_mode": args.venue_mode,
        "event_types": int(np.max(slots.arrays["event_type"])) + 1,
        "venues": int(np.max(slots.arrays["venue"])) + 1,
    }
    if args.prediction_mode == "direct":
        if args.head_mode != "both":
            raise ValueError("head-mode applies only to the hurdle adapter")
        if args.kernel_mode != "cauchy":
            raise ValueError("kernel-mode comparison applies only to the hurdle adapter")
        model = DirectCountEventTransitionAdapter(
            initial_count_scale=args.magnitude_cap,
            **{key: value for key, value in common_model_args.items() if key != "kernel_mode"},
        ).to(device)
    else:
        model = TaxiAwareEventTransitionAdapter(
            initial_activation_scale=args.activation_cap,
            initial_magnitude_scale=args.magnitude_cap,
            head_mode=args.head_mode,
            **common_model_args,
        ).to(device)
    objective_weights = loss_weights(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else EXPERIMENT_ROOT
        / "runs/event_transport/h12"
        / args.backbone
        / (args.run_name + "_seed%d" % args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    rng = np.random.default_rng(args.seed)
    best_score = None
    stale = 0
    history_rows = []
    started = time.time()
    threshold = (
        float(args.base_threshold)
        if args.base_threshold is not None
        else base_threshold(args.backbone, args.backbone_seed)
    )
    validation_base_prediction = np.where(
        probability[validation_index] >= threshold,
        magnitude[validation_index],
        0.0,
    ).astype(np.float32)
    validation_base_mae, validation_base_rmse, validation_base_pcc = numpy_metrics(
        validation_base_prediction, target[validation_index]
    )
    # Constant frozen outputs have zero PCC, especially in tiny smoke runs.
    # In that case use the absolute PCC term; ordinary comparisons keep their
    # original normalized checkpoint score.
    validation_pcc_scale = validation_base_pcc if abs(validation_base_pcc) > 1e-12 else 1.0

    training_epochs = 0 if args.evaluation_only_checkpoint is not None else args.epochs
    for epoch in range(1, training_epochs + 1):
        model.train()
        order = train_index.copy()
        rng.shuffle(order)
        total_loss = 0.0
        seen = 0
        for begin in range(0, len(order), args.batch_size):
            selected = order[begin : begin + args.batch_size]
            event = slots.batch(
                selected,
                device,
                slot_mode=args.event_slot_mode,
                permutation=permutation,
                scale_mode=args.scale_mode,
            )
            o = torch.from_numpy(origin[selected]).long().to(device)
            d = torch.from_numpy(destination[selected]).long().to(device)
            base_l = torch.from_numpy(base_logit[selected]).to(device)
            base_m = torch.from_numpy(base_logmag[selected]).to(device)
            base_count = torch.from_numpy(base_continuous_count[selected]).to(device)
            y = torch.from_numpy(target[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if args.prediction_mode == "direct":
                    expected, count_delta = model(*event, o, d, base_count)
                    flow_mae = torch.abs(expected - y).mean()
                    correction = count_delta.square().mean()
                    loss = (
                        objective_weights["flow"] * flow_mae
                        + objective_weights["correction"] * correction
                    )
                else:
                    logit, logmag, activation_delta, magnitude_delta = model(
                        *event, o, d, base_l, base_m
                    )
                    p = torch.sigmoid(logit)
                    mu = torch.expm1(logmag)
                    expected = p * mu
                    positive = y > 0
                    flow_mae = torch.abs(expected - y).mean()
                    positive_loss = (
                        F.smooth_l1_loss(mu[positive], y[positive], beta=1.0)
                        if bool(positive.any())
                        else mu.new_zeros(())
                    )
                    rmse_loss = torch.sqrt((expected - y).square().mean() + 1.0e-6)
                    corr_loss = correlation_loss(expected, y)
                    activation_loss = F.binary_cross_entropy_with_logits(
                        logit, positive.float()
                    )
                    correction = (
                        activation_delta.square().mean()
                        + magnitude_delta.square().mean()
                    )
                    loss = (
                        objective_weights["flow"] * flow_mae
                        + objective_weights["positive"] * positive_loss
                        + objective_weights["rmse"] * rmse_loss
                        + objective_weights["correlation"] * corr_loss
                        + objective_weights["activation"] * activation_loss
                        + objective_weights["correction"] * correction
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(selected)
            seen += len(selected)

        if args.prediction_mode == "direct":
            validation_prediction, _ = direct_adapter_outputs(
                model,
                validation_index,
                slots,
                origin,
                destination,
                base_continuous_count,
                device,
                args.batch_size,
                args.event_slot_mode,
                permutation,
                args.scale_mode,
            )
        else:
            val_logit, val_logmag, _, _ = adapter_outputs(
                model,
                validation_index,
                slots,
                origin,
                destination,
                base_logit,
                base_logmag,
                device,
                args.batch_size,
                args.event_slot_mode,
                permutation,
                args.scale_mode,
            )
            validation_prediction = expected_prediction(
                val_logit, val_logmag, args.expectation_cutoff
            )
        mae, rmse, pcc = numpy_metrics(
            validation_prediction, target[validation_index]
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "validation_mae": mae,
            "validation_rmse": rmse,
            "validation_pcc": pcc,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if args.selection == "balanced":
            score = (
                mae / validation_base_mae
                + rmse / validation_base_rmse
                - pcc / validation_pcc_scale,
                mae,
                rmse,
                -pcc,
            )
        else:
            score = (mae, rmse, -pcc)
        if best_score is None or score < best_score:
            best_score = score
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation": row},
                checkpoint,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    if args.evaluation_only_checkpoint is None:
        (output_dir / "history.json").write_text(
            json.dumps(history_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    payload_path = args.evaluation_only_checkpoint or checkpoint
    if not payload_path.is_absolute():
        payload_path = EXPERIMENT_ROOT / payload_path
    payload = torch.load(payload_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if args.prediction_mode == "direct":
        validation_prediction, val_delta = direct_adapter_outputs(
            model,
            validation_index,
            slots,
            origin,
            destination,
            base_continuous_count,
            device,
            args.batch_size,
            args.event_slot_mode,
            permutation,
            args.scale_mode,
        )
        save_direct_predictions(
            output_dir / "validation_predictions.npz",
            validation_index,
            hour,
            edge,
            target,
            probability,
            magnitude,
            validation_prediction,
            val_delta,
            threshold,
        )
    else:
        val_logit, val_logmag, val_da, val_dm = adapter_outputs(
            model,
            validation_index,
            slots,
            origin,
            destination,
            base_logit,
            base_logmag,
            device,
            args.batch_size,
            args.event_slot_mode,
            permutation,
            args.scale_mode,
        )
        validation_prediction = expected_prediction(
            val_logit, val_logmag, args.expectation_cutoff
        )
        save_predictions(
            output_dir / "validation_predictions.npz",
            validation_index,
            hour,
            edge,
            target,
            probability,
            magnitude,
            base_logit,
            base_logmag,
            val_logit,
            val_logmag,
            val_da,
            val_dm,
            args.expectation_cutoff,
            threshold,
        )
    if args.validation_only:
        validation_result = {
            "backbone": args.backbone,
            "history_hours": HISTORY_HOURS,
            "method": args.run_name,
            "device": str(device),
            "seed": args.seed,
            "backbone_cache_key": cache_key,
            "best_epoch": int(payload["epoch"]),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "evaluation_scope": evaluation_scope,
            "loss_preset": args.loss_preset,
            "loss_weights": objective_weights,
            "mechanism": model.mechanism_parameters(),
            "validation_direct_samples": int(len(validation_index)),
            "validation_base": {
                "mae": float(validation_base_mae),
                "rmse": float(validation_base_rmse),
                "pcc": float(validation_base_pcc),
            },
            "validation": metric_dict(
                validation_prediction, target[validation_index]
            ),
            "selection_score": float(
                numpy_metrics(validation_prediction, target[validation_index])[0]
                / validation_base_mae
                + numpy_metrics(validation_prediction, target[validation_index])[1]
                / validation_base_rmse
                - numpy_metrics(validation_prediction, target[validation_index])[2]
                / validation_pcc_scale
            ),
            "training_protocol": {
                "maximum_epochs": args.epochs,
                "early_stopping_patience": args.patience,
                "epochs_run": len(history_rows),
                "selection": (
                    "equal normalized validation MAE/RMSE/PCC score"
                    if args.selection == "balanced"
                    else "lowest validation direct MAE; RMSE and PCC break ties"
                ),
                "test_metrics_computed": False,
            },
        }
        (output_dir / "validation_result.json").write_text(
            json.dumps(validation_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(validation_result, ensure_ascii=False, indent=2))
        return

    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    if args.prediction_mode == "direct":
        event_prediction, test_delta = direct_adapter_outputs(
            model,
            test_index,
            slots,
            origin,
            destination,
            base_continuous_count,
            device,
            args.batch_size,
            args.event_slot_mode,
            permutation,
            args.scale_mode,
        )
    else:
        test_logit, test_logmag, test_da, test_dm = adapter_outputs(
            model,
            test_index,
            slots,
            origin,
            destination,
            base_logit,
            base_logmag,
            device,
            args.batch_size,
            args.event_slot_mode,
            permutation,
            args.scale_mode,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    if args.prediction_mode == "direct":
        base_prediction, base_expected, event_prediction = save_direct_predictions(
            output_dir / "test_predictions.npz",
            test_index,
            hour,
            edge,
            target,
            probability,
            magnitude,
            event_prediction,
            test_delta,
            threshold,
        )
        event_hard = event_prediction
    else:
        base_prediction, base_expected, event_prediction = save_predictions(
            output_dir / "test_predictions.npz",
            test_index,
            hour,
            edge,
            target,
            probability,
            magnitude,
            base_logit,
            base_logmag,
            test_logit,
            test_logmag,
            test_da,
            test_dm,
            args.expectation_cutoff,
            threshold,
        )
        event_probability = 1.0 / (1.0 + np.exp(-test_logit))
        event_magnitude = np.expm1(np.maximum(test_logmag, 0.0))
        event_hard = np.where(
            event_probability >= threshold, event_magnitude, 0.0
        ).astype(np.float32)
    base_metrics = metric_dict(base_prediction, target[test_index])
    base_expected_metrics = metric_dict(base_expected, target[test_index])
    event_metrics = metric_dict(event_prediction, target[test_index])
    direct_mode = args.prediction_mode == "direct"
    result = {
        "backbone": args.backbone,
        "history_hours": HISTORY_HOURS,
        "method": args.run_name,
        "architecture": (
            "taxi-aware event-anchored direct count adapter"
            if direct_mode
            else "taxi-aware event-anchored demand transition adapter"
        ),
        "architecture_config": {
            "event_support": "venue-centered pickup/dropoff endpoint exposure",
            "temporal_mechanism": "continuous start-anchored arrival and end-anchored departure lifecycle",
            "event_potential": "event semantics weighted by lifecycle and endpoint exposure",
            "od_interaction": "event exposure x frozen-Hurdle OD susceptibility",
            "outputs": (
                ["direct_count", "count_delta"]
                if direct_mode
                else ["activation_delta", "magnitude_delta"]
            ),
            "prediction_mode": args.prediction_mode,
            "head_mode": args.head_mode,
            "kernel_mode": args.kernel_mode,
            "base_state_mode": args.base_state_mode,
            "event_slot_mode": args.event_slot_mode,
            "scale_mode": args.scale_mode,
            "hard_phase_bins": None,
            "demand_transport": None,
            "context": None,
            "learned_gate": None,
            "peak_head": None,
            "gru": None,
            "node_dim": args.node_dim,
            "event_rank": args.event_rank,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "initial_activation_scale": args.activation_cap,
            "initial_magnitude_scale": args.magnitude_cap,
            "learned_mechanism": model.mechanism_parameters(),
        },
        "device": str(device),
        "seed": args.seed,
        "backbone_seed": args.backbone_seed,
        "backbone_cache_key": cache_key,
        "best_epoch": int(payload["epoch"]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "evaluation_scope": evaluation_scope,
        "loss_preset": args.loss_preset,
        "loss_weights": objective_weights,
        "holdout_type": args.holdout_type,
        "holdout_venue": args.holdout_venue,
        "expectation_cutoff": args.expectation_cutoff,
        "base_threshold": threshold,
        "training_seconds": (
            0.0
            if args.evaluation_only_checkpoint is not None
            else time.time() - started
        ),
        "inference_seconds": inference_seconds,
        "inference_direct_samples": int(len(test_index)),
        "training_protocol": {
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "epochs_run": len(history_rows),
            "selection": (
                "equal normalized validation MAE/RMSE/PCC score"
                if args.selection == "balanced"
                else "lowest validation direct MAE; RMSE and PCC break ties"
            ),
            "test_used_for_model_selection": False,
            "evaluation_only_checkpoint": (
                str(payload_path)
                if args.evaluation_only_checkpoint is not None
                else None
            ),
        },
        "supervision": {
            "flow_target": "next-hour event-anchored OD count",
            "activation_target": None if direct_mode else "1[y > 0]",
            "positive_magnitude_target": None if direct_mode else "y | y > 0",
            "peak_target": None,
        },
        "test_direct_samples": int(len(test_index)),
        "base": base_metrics,
        "base_expected_output": base_expected_metrics,
        "event": event_metrics,
        "event_hard_threshold": (
            None
            if direct_mode
            else metric_dict(event_hard, target[test_index])
        ),
        "change": {
            "mae_percent": 100.0
            * (base_metrics["mae"] - event_metrics["mae"])
            / base_metrics["mae"],
            "rmse_percent": 100.0
            * (base_metrics["rmse"] - event_metrics["rmse"])
            / base_metrics["rmse"],
            "pcc_delta": event_metrics["pcc"] - base_metrics["pcc"],
        },
        "adapter_only_change_from_base_expectation": {
            "mae_percent": 100.0
            * (base_expected_metrics["mae"] - event_metrics["mae"])
            / base_expected_metrics["mae"],
            "rmse_percent": 100.0
            * (base_expected_metrics["rmse"] - event_metrics["rmse"])
            / base_expected_metrics["rmse"],
            "pcc_delta": event_metrics["pcc"] - base_expected_metrics["pcc"],
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
