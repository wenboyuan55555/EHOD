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
from baselines.event._shared.adapters import CDGONEventAdapter, ConcatEventAdapter, FuseCrossAttentionEventAdapter, LegacyAttentionEventAdapter, MatrixEventAdapter, QAQueryEventAdapter, SemanticSpatialMoEEventAdapter, StructuredEventAdapter, T3DualPredictorEventAdapter
from ehod.evaluation.metrics import StreamingMetrics
from ehod.common.paths import EVENT_INDEX_PATH


INDEX_PATH = EVENT_INDEX_PATH
FEATURE_PATH = EXPERIMENT_ROOT / "cache" / "event_structured_features.npy"
EVENT_MATRIX_PATH = EXPERIMENT_ROOT / "cache" / "event_matrix_features.npy"
# The final adapter used a finer validation-only operating-point search than
# the original ten-epoch comparison.  Give every trainable event method the
# same grid so neither the proposed adapter nor a comparison method receives a
# threshold/strength tuning advantage.
THRESHOLDS = np.arange(0.025, 1.0, 0.025, dtype=np.float32)
STRENGTHS = np.arange(0.0, 1.0001, 0.125, dtype=np.float32)
METHOD_SOURCE = {
    "qa_query": {
        "paper": "Integrating Query Data for Enhanced Traffic Forecasting",
        "doi": "10.1016/j.knosys.2024.112315",
        "transfer": "query attention, query convolution, pattern transformation, and gated fusion applied to frozen HCF states",
    },
    "cdgon_ode": {
        "paper": "Physics-informed Neural ODE for Post-disaster Mobility Recovery",
        "doi": "10.1145/3637528.3672027",
        "transfer": "one-step event perturbation with learned recovery-to-normal and directed-pair interaction applied to frozen HCF states",
    },
}


def create_event_adapter(method: str):
    if method == "structured":
        return StructuredEventAdapter()
    if method == "matrix":
        return MatrixEventAdapter()
    if method == "legacy_attention":
        return LegacyAttentionEventAdapter()
    if method == "t3_dual":
        return T3DualPredictorEventAdapter()
    if method == "fuse_cross_attention":
        return FuseCrossAttentionEventAdapter()
    if method == "semantic_spatial_moe":
        return SemanticSpatialMoEEventAdapter()
    if method == "concat":
        return ConcatEventAdapter()
    if method == "qa_query":
        return QAQueryEventAdapter()
    if method == "cdgon_ode":
        return CDGONEventAdapter()
    raise ValueError(f"Unknown event-adapter method: {method}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument(
        "--method",
        choices=[
            "concat",
            "matrix",
            "structured",
            "legacy_attention",
            "t3_dual",
            "fuse_cross_attention",
            "semantic_spatial_moe",
            "qa_query",
            "cdgon_ode",
        ],
        required=True,
    )
    parser.add_argument("--base-threshold", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Optional event-backbone cache prefix; defaults to --backbone.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-square-weight", type=float, default=0.0)
    parser.add_argument("--activation-weight", type=float, default=0.1)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-root",
        default="event_adapters",
        choices=["event_adapters", "event_adapter_trials"],
    )
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def logit_numpy(probability):
    value = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value)).astype(np.float32)


def metric_dict(prediction, target):
    metrics = StreamingMetrics()
    metrics.update(
        torch.from_numpy(prediction.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
    )
    return metrics.compute()


@torch.no_grad()
def adapter_outputs(model, indices, features, origin, destination, base_logit, base_logmag, device, batch_size):
    model.eval()
    logits = np.empty(len(indices), dtype=np.float32)
    logmag = np.empty(len(indices), dtype=np.float32)
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        event = torch.from_numpy(np.asarray(features[selected], dtype=np.float32)).to(device)
        o = torch.from_numpy(origin[selected]).long().to(device)
        d = torch.from_numpy(destination[selected]).long().to(device)
        base_l = torch.from_numpy(base_logit[selected]).to(device)
        base_m = torch.from_numpy(base_logmag[selected]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            value_l, value_m = model(event, o, d, base_l, base_m)
        logits[begin : begin + len(selected)] = value_l.float().cpu().numpy()
        logmag[begin : begin + len(selected)] = value_m.float().cpu().numpy()
    return logits, logmag


def validation_grid(base_logit, base_logmag, adapted_logit, adapted_logmag, target):
    best = None
    for strength in STRENGTHS:
        logit = base_logit + strength * (adapted_logit - base_logit)
        logmag = base_logmag + strength * (adapted_logmag - base_logmag)
        probability = 1.0 / (1.0 + np.exp(-logit))
        magnitude = np.expm1(np.maximum(logmag, 0.0))
        for threshold in THRESHOLDS:
            prediction = np.where(probability >= threshold, magnitude, 0.0)
            mae = float(np.mean(np.abs(prediction - target)))
            row = (mae, float(strength), float(threshold))
            if best is None or row < best:
                best = row
    return best


def main():
    args = parse_args()
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
    features = np.load(
        EVENT_MATRIX_PATH if args.method == "matrix" else FEATURE_PATH,
        mmap_mode="r",
    )
    cache_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    cache_key = args.cache_key or args.backbone
    probability_path = cache_dir / (cache_key + "_probability.npy")
    if not probability_path.exists():
        probability_path = cache_dir / "strict_activation_probability.npy"
    probability = np.asarray(
        np.load(probability_path, mmap_mode="r"),
        dtype=np.float32,
    )
    magnitude = np.asarray(
        np.load(cache_dir / (cache_key + "_magnitude.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    base_logit = logit_numpy(probability)
    base_logmag = np.log1p(np.maximum(magnitude, 0.0)).astype(np.float32)
    train_index = np.flatnonzero(hour < TRAIN_END)
    validation_index = np.flatnonzero((hour >= TRAIN_END) & (hour < VAL_END))
    test_index = np.flatnonzero(hour >= VAL_END)
    model = create_event_adapter(args.method)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / args.output_root
        / ("h%d" % args.history_hours)
        / args.backbone
        / ((args.run_name or args.method) + "_seed%d" % args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    history = []
    best = math.inf
    stale = 0
    started = time.time()
    rng = np.random.default_rng(args.seed)
    validation_subset = validation_index
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_index.copy()
        rng.shuffle(order)
        total = 0.0
        n = 0
        for begin in range(0, len(order), args.batch_size):
            selected = order[begin : begin + args.batch_size]
            event = torch.from_numpy(np.asarray(features[selected], dtype=np.float32)).to(device)
            o = torch.from_numpy(origin[selected]).long().to(device)
            d = torch.from_numpy(destination[selected]).long().to(device)
            base_l = torch.from_numpy(base_logit[selected]).to(device)
            base_m = torch.from_numpy(base_logmag[selected]).to(device)
            y = torch.from_numpy(target[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logit, logmag = model(event, o, d, base_l, base_m)
                p = torch.sigmoid(logit)
                mu = torch.expm1(logmag)
                expected_mae = (
                    p * torch.abs(mu - y) + (1.0 - p) * torch.abs(y)
                ).mean()
                expected_square = (
                    p * torch.square(mu - y) + (1.0 - p) * torch.square(y)
                ).mean()
                activation = F.binary_cross_entropy_with_logits(logit, (y > 0).float())
                positive = y > 0
                positive_loss = F.smooth_l1_loss(mu[positive], y[positive], beta=1.0)
                loss = (
                    expected_mae
                    + args.expected_square_weight * expected_square
                    + args.activation_weight * activation
                    + args.positive_weight * positive_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(selected)
            n += len(selected)
        adapted_l, adapted_m = adapter_outputs(
            model,
            validation_subset,
            features,
            origin,
            destination,
            base_logit,
            base_logmag,
            device,
            args.batch_size,
        )
        selection = validation_grid(
            base_logit[validation_subset],
            base_logmag[validation_subset],
            adapted_l,
            adapted_m,
            target[validation_subset],
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "validation_direct_mae": selection[0],
            "strength": selection[1],
            "threshold": selection[2],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection[0] < best - 1e-7:
            best = selection[0]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "selection": selection,
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.validation_only:
        adapted_l, adapted_m = adapter_outputs(
            model,
            validation_index,
            features,
            origin,
            destination,
            base_logit,
            base_logmag,
            device,
            args.batch_size,
        )
        strength = float(payload["selection"][1])
        threshold = float(payload["selection"][2])
        final_logit = base_logit[validation_index] + strength * (
            adapted_l - base_logit[validation_index]
        )
        final_logmag = base_logmag[validation_index] + strength * (
            adapted_m - base_logmag[validation_index]
        )
        final_probability = 1.0 / (1.0 + np.exp(-final_logit))
        final_magnitude = np.expm1(np.maximum(final_logmag, 0.0))
        prediction = np.where(
            final_probability >= threshold, final_magnitude, 0.0
        ).astype(np.float32)
        validation_metrics = metric_dict(prediction, target[validation_index])
        summary = {
            "backbone": args.backbone,
            "history_hours": args.history_hours,
            "method": args.method,
            "run_name": args.run_name or args.method,
            "seed": args.seed,
            "device": str(device),
            "best_epoch": int(payload["epoch"]),
            "adapter_strength": strength,
            "event_threshold": threshold,
            "validation": validation_metrics,
            "training_seconds": time.time() - started,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "source_mechanism": METHOD_SOURCE.get(args.method),
            "training_protocol": {
                "maximum_epochs": args.epochs,
                "early_stopping_patience": args.patience,
                "epochs_run": len(history),
                "selection": "lowest validation direct MAE",
                "strength_grid": [float(value) for value in STRENGTHS],
                "threshold_grid": [float(value) for value in THRESHOLDS],
            },
            "training_objective": {
                "expected_square_weight": args.expected_square_weight,
                "activation_weight": args.activation_weight,
                "positive_weight": args.positive_weight,
            },
        }
        (output_dir / "validation_result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    adapted_l, adapted_m = adapter_outputs(
        model,
        test_index,
        features,
        origin,
        destination,
        base_logit,
        base_logmag,
        device,
        args.batch_size,
    )
    strength = float(payload["selection"][1])
    threshold = float(payload["selection"][2])
    final_logit = base_logit[test_index] + strength * (
        adapted_l - base_logit[test_index]
    )
    final_logmag = base_logmag[test_index] + strength * (
        adapted_m - base_logmag[test_index]
    )
    final_probability = 1.0 / (1.0 + np.exp(-final_logit))
    final_magnitude = np.expm1(np.maximum(final_logmag, 0.0))
    event_prediction = np.where(
        final_probability >= threshold, final_magnitude, 0.0
    ).astype(np.float32)
    base_prediction = np.where(
        probability[test_index] >= args.base_threshold,
        magnitude[test_index],
        0.0,
    ).astype(np.float32)
    base_metrics = metric_dict(base_prediction, target[test_index])
    event_metrics = metric_dict(event_prediction, target[test_index])
    result = {
        "backbone": args.backbone,
        "history_hours": args.history_hours,
        "method": args.method,
        "backbone_cache_key": cache_key,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": int(payload["epoch"]),
        "adapter_strength": strength,
        "event_threshold": threshold,
        "base_threshold": args.base_threshold,
        "training_seconds": time.time() - started,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "source_mechanism": METHOD_SOURCE.get(args.method),
        "training_protocol": {
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "epochs_run": len(history),
            "selection": "lowest validation direct MAE",
            "strength_grid": [float(value) for value in STRENGTHS],
            "threshold_grid": [float(value) for value in THRESHOLDS],
        },
        "test_direct_samples": int(len(test_index)),
        "base": base_metrics,
        "event": event_metrics,
        "change": {
            "mae_percent": 100.0 * (base_metrics["mae"] - event_metrics["mae"]) / base_metrics["mae"],
            "rmse_percent": 100.0 * (base_metrics["rmse"] - event_metrics["rmse"]) / base_metrics["rmse"],
            "pcc_delta": event_metrics["pcc"] - base_metrics["pcc"],
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
