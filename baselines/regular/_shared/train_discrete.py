from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, VAL_END, TEST_END, calendar_features, history_batch, inverse_log_standardize, iter_hour_batches, load_normalized, load_od, load_scaler, load_static
from ehod.common.device import require_cuda
from ehod.data.events import ScheduledEventStore
from ehod.evaluation.metrics import StreamingMetrics
from baselines.regular._shared.factory import MODEL_NAMES, create_model, forward_model, static_tensors


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


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--variant", choices=["plain", "native", "hurdle"], required=True)
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-hours", type=int, default=None)
    parser.add_argument("--max-eval-hours", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run-directory name; defaults to the standard protocol name.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="skip fitting and evaluate an existing best_model.pt checkpoint",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--event-mode",
        choices=["native", "none"],
        default="native",
        help=(
            "OD-PF only: use its native planned-event matrix or remove the "
            "matrix to expose a no-event backbone for adapter replacement"
        ),
    )
    return parser.parse_args()


def variant_allowed(model: str, variant: str) -> bool:
    if variant == "native":
        return model in ("stidgcn", "stpro", "odpf")
    return True


def make_batch(normalized, od, starts, history_hours, device):
    history = torch.from_numpy(
        history_batch(normalized, starts, history_hours)
    ).to(device)
    target_scaled = torch.from_numpy(np.asarray(normalized[starts], dtype=np.float32)).to(
        device
    )
    target_raw = torch.from_numpy(np.asarray(od[starts], dtype=np.float32)).to(device)
    calendar = torch.from_numpy(calendar_features(starts)).to(device)
    return history, target_scaled, target_raw, calendar


def event_tensor(event_store, starts, device):
    if event_store is None:
        return None
    return torch.from_numpy(event_store.batch_matrix(starts)).to(device)


def training_loss(variant, model_name, prediction, target_scaled, target_raw):
    if variant == "hurdle":
        mask = target_raw > 0
        return F.smooth_l1_loss(prediction[mask], target_scaled[mask])
    if variant == "plain":
        return F.smooth_l1_loss(prediction, target_scaled)
    if model_name == "stpro":
        mask = target_raw > 0
        return F.l1_loss(prediction[mask], target_scaled[mask])
    if model_name == "stidgcn":
        mask = target_raw > 0
        return F.l1_loss(prediction[mask], target_scaled[mask])
    if model_name == "odpf":
        mask = target_raw >= 2
        return F.mse_loss(prediction[mask], target_scaled[mask])
    raise ValueError("Unsupported native variant")


@torch.no_grad()
def validation_score(
    model_name,
    variant,
    model,
    normalized,
    od,
    history_hours,
    static,
    device,
    mean,
    std,
    batch_size,
    limit,
    event_store=None,
):
    model.eval()
    absolute = 0.0
    n = 0
    for starts in iter_hour_batches(
        "validation", batch_size, shuffle=False, limit=limit
    ):
        history, target_scaled, target_raw, calendar = make_batch(
            normalized, od, starts, history_hours, device
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            prediction_scaled = forward_model(
                model_name,
                model,
                history,
                calendar,
                static,
                event_matrix=event_tensor(event_store, starts, device),
                target_starts=torch.from_numpy(starts.astype(np.int64)).to(device),
            )
        prediction = inverse_log_standardize(prediction_scaled, mean, std)
        if variant == "hurdle":
            mask = target_raw > 0
        elif variant == "native" and model_name == "odpf":
            mask = target_raw >= 2
        elif variant == "native" and model_name in ("stidgcn", "stpro"):
            mask = target_raw > 0
        else:
            mask = torch.ones_like(target_raw, dtype=torch.bool)
        absolute += float((prediction[mask] - target_raw[mask]).abs().sum().item())
        n += int(mask.sum().item())
    return absolute / max(n, 1)


@torch.no_grad()
def predict_raw(
    model_name,
    model,
    normalized,
    starts,
    history_hours,
    static,
    device,
    mean,
    std,
    event_store=None,
):
    history = torch.from_numpy(
        history_batch(normalized, starts, history_hours)
    ).to(device)
    calendar = torch.from_numpy(calendar_features(starts)).to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        scaled = forward_model(
            model_name,
            model,
            history,
            calendar,
            static,
            event_matrix=event_tensor(event_store, starts, device),
            target_starts=torch.from_numpy(starts.astype(np.int64)).to(device),
        )
    return inverse_log_standardize(scaled, mean, std).cpu().numpy().astype(np.float32)


def update_threshold_bins(probability, magnitude, actual, bins):
    p = probability.reshape(-1)
    m = np.maximum(magnitude.reshape(-1), 0.0).astype(np.float64, copy=False)
    y = actual.reshape(-1).astype(np.float64, copy=False)
    base = np.abs(y)
    delta = np.abs(m - y) - base
    index = np.searchsorted(THRESHOLDS, p, side="right")
    bins += np.bincount(index, weights=delta, minlength=len(THRESHOLDS) + 1)
    return float(base.sum())


@torch.no_grad()
def evaluate_hurdle(
    model_name,
    model,
    normalized,
    od,
    history_hours,
    static,
    device,
    mean,
    std,
    batch_size,
    max_eval_hours,
    event_store=None,
):
    gate_dir = EXPERIMENT_ROOT / "runs/activation" / ("h%d_seed0" % history_hours)
    validation_probability = np.load(
        gate_dir / "validation_probability.npy", mmap_mode="r"
    )
    test_probability = np.load(gate_dir / "test_probability.npy", mmap_mode="r")
    bins = np.zeros(len(THRESHOLDS) + 1, dtype=np.float64)
    base_error = 0.0
    validation_n = 0
    validation_hours = 1008 if max_eval_hours is None else min(1008, max_eval_hours)
    for starts in iter_hour_batches(
        "validation", batch_size, shuffle=False, limit=validation_hours
    ):
        magnitude = predict_raw(
            model_name,
            model,
            normalized,
            starts,
            history_hours,
            static,
            device,
            mean,
            std,
            event_store,
        )
        actual = np.asarray(od[starts], dtype=np.float32)
        probability = np.asarray(
            validation_probability[starts - 6720], dtype=np.float32
        )
        base_error += update_threshold_bins(probability, magnitude, actual, bins)
        validation_n += actual.size
    active_delta = np.cumsum(bins[:0:-1])[::-1]
    validation_mae = (base_error + active_delta) / max(validation_n, 1)
    best_index = int(np.argmin(validation_mae))
    threshold = float(THRESHOLDS[best_index])
    metrics = StreamingMetrics()
    test_hours = 1032 if max_eval_hours is None else min(1032, max_eval_hours)
    for starts in iter_hour_batches(
        "test", batch_size, shuffle=False, limit=test_hours
    ):
        magnitude = predict_raw(
            model_name,
            model,
            normalized,
            starts,
            history_hours,
            static,
            device,
            mean,
            std,
            event_store,
        )
        probability = np.asarray(test_probability[starts - VAL_END], dtype=np.float32)
        prediction = np.where(probability >= threshold, magnitude, 0.0).astype(np.float32)
        metrics.update(
            torch.from_numpy(prediction),
            torch.from_numpy(np.asarray(od[starts], dtype=np.float32)),
        )
    return {
        "threshold": threshold,
        "validation_mae": float(validation_mae[best_index]),
        "test": metrics.compute(),
    }


@torch.no_grad()
def evaluate_direct(
    model_name,
    model,
    normalized,
    od,
    history_hours,
    static,
    device,
    mean,
    std,
    batch_size,
    max_eval_hours,
    event_store=None,
):
    metrics = StreamingMetrics()
    test_hours = 1032 if max_eval_hours is None else min(1032, max_eval_hours)
    for starts in iter_hour_batches(
        "test", batch_size, shuffle=False, limit=test_hours
    ):
        prediction = predict_raw(
            model_name,
            model,
            normalized,
            starts,
            history_hours,
            static,
            device,
            mean,
            std,
            event_store,
        )
        metrics.update(
            torch.from_numpy(prediction),
            torch.from_numpy(np.asarray(od[starts], dtype=np.float32)),
        )
    return {"test": metrics.compute()}


def main() -> None:
    args = parse_args()
    if args.model != "odpf" and args.event_mode != "native":
        raise ValueError("--event-mode none is only defined for OD-PF")
    if not variant_allowed(args.model, args.variant):
        raise ValueError("%s does not define a native zero variant" % args.model)
    if args.smoke:
        args.epochs = 2
        args.max_train_hours = 4
        args.max_eval_hours = 2
    device = require_cuda(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_model_name = (
        "odpf_no_event"
        if args.model == "odpf" and args.event_mode == "none"
        else args.model
    )
    default_run_name = "h%d_%s_seed%d_%s" % (
        args.history_hours,
        args.variant,
        args.seed,
        "smoke" if args.smoke else "formal",
    )
    output_dir = (
        EXPERIMENT_ROOT / "runs" / run_model_name / (args.run_name or default_run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    od = load_od()
    normalized = load_normalized()
    mean, std = load_scaler()
    static = static_tensors(load_static(), device)
    event_store = (
        ScheduledEventStore()
        if args.model == "odpf" and args.event_mode == "native"
        else None
    )
    model = create_model(args.model, args.history_hours, device=device).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()
    checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"
    history_rows = []
    start_epoch = 1
    best = math.inf
    stale = 0
    if args.resume and last_checkpoint.exists():
        payload = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        history_rows = payload["history"]
        start_epoch = int(payload["epoch"]) + 1
        best = float(payload["best"])
        stale = int(payload["stale"])
    if args.evaluation_only:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        history_path = output_dir / "history.json"
        if not history_rows and history_path.exists():
            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = args.epochs + 1
    peak_memory = 0
    full_started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        total = 0.0
        seen = 0
        epoch_started = time.time()
        for starts in iter_hour_batches(
            "train",
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            limit=args.max_train_hours,
        ):
            history, target_scaled, target_raw, calendar = make_batch(
                normalized, od, starts, args.history_hours, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = forward_model(
                    args.model,
                    model,
                    history,
                    calendar,
                    static,
                    event_matrix=event_tensor(event_store, starts, device),
                    target_starts=torch.from_numpy(starts.astype(np.int64)).to(device),
                )
                loss = training_loss(
                    args.variant, args.model, prediction, target_scaled, target_raw
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(starts)
            seen += len(starts)
        validation = validation_score(
            args.model,
            args.variant,
            model,
            normalized,
            od,
            args.history_hours,
            static,
            device,
            mean,
            std,
            args.batch_size,
            2 if args.smoke else args.max_eval_hours,
            event_store,
        )
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_selection_mae": validation,
            "seconds": time.time() - epoch_started,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        # Once an AMP update has made the live weights non-finite, later
        # epochs cannot recover: every comparison against a NaN validation
        # score is false and the loop merely burns through the patience
        # counter before reloading the last finite best checkpoint.  Stop at
        # the first non-finite epoch and evaluate that already-selected best
        # checkpoint instead.  This leaves finite runs and model selection
        # unchanged while preventing a poisoned ``last_model.pt`` from being
        # used as a future resume point.
        if not math.isfinite(row["train_loss"]) or not math.isfinite(validation):
            (output_dir / "history.json").write_text(
                json.dumps(history_rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "stopped_nonfinite_epoch": epoch,
                        "selected_best_validation_mae": best,
                    }
                ),
                flush=True,
            )
            break
        if validation < best - 1.0e-7:
            best = validation
            stale = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_selection_mae": validation,
                },
                checkpoint,
            )
        else:
            stale += 1
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "history": history_rows,
            },
            last_checkpoint,
        )
        (output_dir / "history.json").write_text(
            json.dumps(history_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if stale >= args.patience:
            break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    if args.variant == "hurdle" and not args.smoke:
        evaluation = evaluate_hurdle(
            args.model,
            model,
            normalized,
            od,
            args.history_hours,
            static,
            device,
            mean,
            std,
            args.batch_size,
            args.max_eval_hours,
            event_store,
        )
    else:
        evaluation = evaluate_direct(
            args.model,
            model,
            normalized,
            od,
            args.history_hours,
            static,
            device,
            mean,
            std,
            args.batch_size,
            args.max_eval_hours,
            event_store,
        )
    summary = {
        "model": run_model_name,
        "architecture_model": args.model,
        "variant": args.variant,
        "history_hours": args.history_hours,
        "seed": args.seed,
        "run_name": args.run_name or default_run_name,
        "best_epoch": int(payload["epoch"]),
        "selection_mae": float(payload["validation_selection_mae"]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "peak_gpu_memory_bytes": peak_memory,
        "training_seconds": (
            sum(float(row.get("seconds", 0.0)) for row in history_rows)
            if args.evaluation_only
            else time.time() - full_started
        ),
        "strict_window": True,
        "planned_events": (
            {
                "enabled": True,
                "events": event_store.event_count,
                "venues": event_store.venue_count,
                "method": "OD-PF-style three-channel event matrix",
            }
            if event_store is not None
            else {
                "enabled": False,
                "method": (
                    "OD-PF event matrix removed for adapter-replacement control"
                    if args.model == "odpf"
                    else None
                ),
            }
        ),
        "device": args.device,
        "full_run": not args.smoke and args.max_train_hours is None and args.max_eval_hours is None,
        "evaluation": evaluation,
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
