#!/usr/bin/env python3
"""Train CMOD as the positive-magnitude stage of the shared portable OD hurdle task."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from ehod.data.od import OD_PATH, EXPERIMENT_ROOT as BENCHMARK_OUTPUT_ROOT
from ehod.common.paths import CONTINUOUS_CACHE_ROOT

import numpy as np
import torch

from baselines.regular.cmod.model import ContinuousWindowCMOD
from baselines.regular._shared.continuous_evaluation import evaluate_hurdle_model
from baselines.regular.cmod.window_data import (
    LAMBDA_SECONDS,
    N_HOURS,
    N_NODES,
    TRAIN_END,
    VAL_END,
    WindowStream,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=CONTINUOUS_CACHE_ROOT)
    parser.add_argument(
        "--od-matrix",
        type=Path,
        default=OD_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_OUTPUT_ROOT / "runs" / "cmod_continuous_hurdle",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-hours", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--message-dim", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=336)
    parser.add_argument("--max-train-targets", type=int, default=None)
    parser.add_argument("--max-eval-targets", type=int, default=None)
    parser.add_argument("--skip-hurdle", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


class PositiveMAELoss(torch.nn.Module):
    """Magnitude loss evaluated only on OD cells whose target count is positive."""

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        positive = truth >= 1.0
        if not bool(positive.any()):
            return prediction.sum() * 0.0
        return torch.abs(prediction[positive] - truth[positive]).mean()


def stream_batches(iterator, batch_size: int):
    batch = []
    for sample in iterator:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def predict_batch(
    model: ContinuousWindowCMOD, samples: list, device: torch.device
) -> torch.Tensor:
    """Keep window encoders independent while batching the expensive OD decoder."""
    embeddings = []
    for _, weights, last_times in samples:
        embeddings.append(
            model.encode_window(
                torch.from_numpy(weights).to(device),
                torch.from_numpy(last_times).to(device),
            )
        )
    return model.decode_embeddings(torch.stack(embeddings, dim=0))


def run_train_epoch(
    model: ContinuousWindowCMOD,
    criterion: PositiveMAELoss,
    optimizer: torch.optim.Optimizer,
    stream: WindowStream,
    od: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model.train()
    total_loss = 0.0
    sample_count = 0
    batch_count = 0
    started = time.time()
    iterator = stream.iterate(args.input_hours, TRAIN_END, args.max_train_targets)
    for samples in stream_batches(iterator, args.batch_size):
        target_hours = [sample[0] for sample in samples]
        target = torch.from_numpy(
            np.asarray(od[target_hours], dtype=np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_batch(model, samples, device)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at target hour {target_hours[0]}")
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
        previous_count = sample_count
        sample_count += len(samples)
        crossed_log_boundary = (
            sample_count // args.log_every != previous_count // args.log_every
        )
        if batch_count == 1 or crossed_log_boundary:
            elapsed = time.time() - started
            print(
                f"train target={target_hours[-1]} samples={sample_count} "
                f"positive_MAE={loss.item():.6f} mean={total_loss / batch_count:.6f} "
                f"speed={sample_count / elapsed:.2f} sample/s",
                flush=True,
            )
    return {
        "positive_mae_loss": total_loss / batch_count,
        "samples": sample_count,
        "batches": batch_count,
        "seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate_positive(
    model: ContinuousWindowCMOD,
    stream: WindowStream,
    od: np.ndarray,
    device: torch.device,
    start: int,
    end: int,
    limit: int | None,
    label: str,
    log_every: int,
    batch_size: int,
) -> dict:
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    n = 0
    hours = 0
    started = time.time()
    iterator = stream.iterate(start, end, limit)
    for samples in stream_batches(iterator, batch_size):
        target_hours = [sample[0] for sample in samples]
        prediction = predict_batch(model, samples, device).clamp_min(0.0)
        target = torch.from_numpy(
            np.asarray(od[target_hours], dtype=np.float32)
        ).to(device)
        positive = target >= 1.0
        error = prediction[positive].double() - target[positive].double()
        sum_abs += torch.abs(error).sum().item()
        sum_sq += (error * error).sum().item()
        n += int(positive.sum().item())
        previous_hours = hours
        hours += len(samples)
        crossed_log_boundary = hours // log_every != previous_hours // log_every
        if previous_hours == 0 or crossed_log_boundary:
            print(
                f"{label} target={target_hours[-1]} hours={hours} "
                f"speed={hours / (time.time() - started):.2f} sample/s",
                flush=True,
            )
    return {
        "positive_mae": sum_abs / n,
        "positive_rmse": math.sqrt(sum_sq / n),
        "positive_n": n,
        "n_target_hours": hours,
        "seconds": time.time() - started,
    }


@torch.no_grad()
def write_magnitude_cache(
    model: ContinuousWindowCMOD,
    stream: WindowStream,
    device: torch.device,
    output_path: Path,
    log_every: int,
    batch_size: int,
) -> np.ndarray:
    """Write validation/test magnitudes once so threshold search reuses exact outputs."""
    model.eval()
    shape = (N_HOURS - TRAIN_END, N_NODES, N_NODES)
    cache = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=shape
    )
    started = time.time()
    completed = 0
    iterator = stream.iterate(TRAIN_END, N_HOURS)
    for samples in stream_batches(iterator, batch_size):
        target_hours = np.asarray([sample[0] for sample in samples], dtype=np.int32)
        prediction = predict_batch(model, samples, device).clamp_min(0.0)
        cache[target_hours - TRAIN_END] = prediction.cpu().numpy().astype(np.float32)
        previous_completed = completed
        completed += len(samples)
        crossed_log_boundary = (
            completed // log_every != previous_completed // log_every
        )
        if previous_completed == 0 or crossed_log_boundary:
            print(
                f"magnitude-cache target={target_hours[-1]} hours={completed} "
                f"speed={completed / (time.time() - started):.2f} sample/s",
                flush=True,
            )
    cache.flush()
    del cache
    return np.load(output_path, mmap_mode="r")


def main() -> None:
    args = parse_args()
    if args.input_hours not in {1, 3, 6, 12}:
        raise ValueError("--input-hours must be one of 1, 3, 6, 12")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    hour_weights = np.load(
        args.cache_dir / "continuous_hour_exp_weights.npy", mmap_mode="r"
    )
    hour_last = np.load(
        args.cache_dir / "continuous_hour_last_pickup_second.npy", mmap_mode="r"
    )
    od = np.load(args.od_matrix, mmap_mode="r")
    if hour_weights.shape != (N_HOURS, N_NODES, N_NODES):
        raise ValueError(f"Unexpected cache shape {hour_weights.shape}")
    if od.shape != (N_HOURS, N_NODES, N_NODES):
        raise ValueError(f"Unexpected OD shape {od.shape}")

    stream = WindowStream(hour_weights, hour_last, args.input_hours)
    model = ContinuousWindowCMOD(
        device=args.device,
        n_nodes=N_NODES,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        lambda_seconds=LAMBDA_SECONDS,
        window_seconds=args.input_hours * 3600.0,
    ).to(device)
    criterion = PositiveMAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.json"
    history = []
    best_mae = math.inf
    best_epoch = -1
    stale = 0
    start_epoch = 1
    prior_elapsed_seconds = 0.0

    config = {
        "task": (
            f"full-year OD{N_NODES}, public-time continuous past{args.input_hours}h "
            "-> next1h full OD"
        ),
        "zero_treatment": "shared activation classifier + positive-only CMOD magnitude",
        "loss": "raw-count positive-only MAE",
        "model_selection": "lowest validation positive-flow MAE",
        "split": {
            "train_targets": [args.input_hours, TRAIN_END],
            "validation_targets": [TRAIN_END, VAL_END],
            "test_targets": [VAL_END, N_HOURS],
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    save_json(args.output_dir / "config.json", config)

    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        best_epoch = int(payload["epoch"])
        best_mae = float(payload["validation_positive"]["positive_mae"])
        start_epoch = best_epoch + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            prior_elapsed_seconds = sum(
                float(row["train"]["seconds"])
                + float(row["validation_positive"]["seconds"])
                for row in history
                if int(row["epoch"]) < start_epoch
            )
        print(
            f"resumed checkpoint={args.resume} epoch={best_epoch} "
            f"val_positive_MAE={best_mae:.6f}",
            flush=True,
        )

    overall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}", flush=True)
        train = run_train_epoch(model, criterion, optimizer, stream, od, device, args)
        validation = evaluate_positive(
            model,
            stream,
            od,
            device,
            TRAIN_END,
            VAL_END,
            args.max_eval_targets,
            "validation-positive",
            args.log_every,
            args.batch_size,
        )
        row = {"epoch": epoch, "train": train, "validation_positive": validation}
        history.append(row)
        save_json(history_path, history)
        print(
            f"epoch={epoch} train_positive_MAE={train['positive_mae_loss']:.6f} "
            f"val_positive_MAE={validation['positive_mae']:.6f} "
            f"val_positive_RMSE={validation['positive_rmse']:.6f}",
            flush=True,
        )
        if validation["positive_mae"] < best_mae:
            best_mae = validation["positive_mae"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_positive": validation,
                    "config": config,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping; best epoch={best_epoch}", flush=True)
                break

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["model"])
    best_validation = evaluate_positive(
        model,
        stream,
        od,
        device,
        TRAIN_END,
        VAL_END,
        args.max_eval_targets,
        "best-validation-positive",
        args.log_every,
        args.batch_size,
    )

    hurdle = None
    if not args.skip_hurdle:
        magnitude_cache = write_magnitude_cache(
            model,
            stream,
            device,
            args.output_dir / "magnitude_predictions.npy",
            args.log_every,
            args.batch_size,
        )

        def predict_magnitude(starts: np.ndarray) -> np.ndarray:
            return np.asarray(magnitude_cache[starts - TRAIN_END], dtype=np.float32)

        hurdle = evaluate_hurdle_model(
            "CMOD-Hurdle",
            predict_magnitude,
            args.output_dir,
            device,
            history_hours=args.input_hours,
        )

    summary = {
        "model": "CMOD-Hurdle",
        "task": (
            f"full-year OD{N_NODES}, public-time continuous past{args.input_hours}h "
            "-> next1h full OD"
        ),
        "input_hours": args.input_hours,
        "best_epoch": int(best["epoch"]),
        "best_validation_positive": best_validation,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": prior_elapsed_seconds + time.time() - overall_start,
        "full_run": (
            args.max_train_targets is None
            and args.max_eval_targets is None
            and not args.skip_hurdle
        ),
        "hurdle": hurdle,
        "protocol": config,
    }
    save_json(args.output_dir / "test_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
