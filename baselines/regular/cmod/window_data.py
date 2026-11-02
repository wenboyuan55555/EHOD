#!/usr/bin/env python3
"""Train/evaluate CMOD on exact 2023 trips for a variable-history 1h task."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from ehod.data.od import OD_PATH, EXPERIMENT_ROOT as BENCHMARK_OUTPUT_ROOT
from ehod.common.paths import CONTINUOUS_CACHE_ROOT

import numpy as np
import torch

from baselines.regular.cmod.model import ContinuousWindowCMOD


N_HOURS = 8760
N_NODES = int(os.environ.get("EHOD_N_NODES", "263"))
TRAIN_END = 6720
VAL_END = 7728
LAMBDA_SECONDS = 3600.0
Q = math.exp(-1.0)
INVALID_TARGET_HOURS = {
    int(value)
    for value in os.environ.get("EHOD_INVALID_TARGET_HOURS", "").split(",")
    if value.strip()
}


def valid_target_hour(target_hour: int, input_hours: int) -> bool:
    """Require both the target and every hour in its strict history window."""
    return all(
        hour not in INVALID_TARGET_HOURS
        for hour in range(target_hour - input_hours, target_hour + 1)
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
        "--output-dir", type=Path, default=BENCHMARK_OUTPUT_ROOT / "runs" / "cmod_continuous"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-hours", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--message-dim", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=168)
    parser.add_argument("--max-train-targets", type=int, default=None)
    parser.add_argument("--max-eval-targets", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a saved best_model.pt",
    )
    return parser.parse_args()


class ODLoss(torch.nn.Module):
    """The zero-aware loss released with CMOD."""

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        zero_mask = truth < 1
        positive_prediction = prediction > 0
        return torch.mean(
            ((prediction - truth) ** 2) * (~zero_mask)
            + ((positive_prediction * prediction - truth) ** 2) * zero_mask
        )


class StreamingMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_target_abs = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.zero_count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        x = prediction.reshape(-1).double()
        y = target.reshape(-1).double()
        error = x - y
        self.n += x.numel()
        self.sum_abs += torch.abs(error).sum().item()
        self.sum_sq += (error * error).sum().item()
        self.sum_target_abs += torch.abs(y).sum().item()
        self.sum_x += x.sum().item()
        self.sum_y += y.sum().item()
        self.sum_x2 += (x * x).sum().item()
        self.sum_y2 += (y * y).sum().item()
        self.sum_xy += (x * y).sum().item()
        self.zero_count += int((y < 0.5).sum().item())

    def compute(self) -> dict:
        mse = self.sum_sq / self.n
        covariance = self.sum_xy - self.sum_x * self.sum_y / self.n
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / self.n
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / self.n
        denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        return {
            "mae": self.sum_abs / self.n,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "wape": self.sum_abs / max(self.sum_target_abs, 1e-12),
            "pcc": covariance / denominator if denominator > 0 else 0.0,
            "target_zero_rate": self.zero_count / self.n,
            "n_matrix_cells": self.n,
        }


class WindowStream:
    """Sequentially combine exact hourly event weights into strict windows."""

    def __init__(
        self, hour_weights: np.ndarray, hour_last: np.ndarray, input_hours: int
    ) -> None:
        self.hour_weights = hour_weights
        self.hour_last = hour_last
        self.input_hours = int(input_hours)

    def iterate(self, start: int, end: int, limit: int | None = None):
        if start < self.input_hours or end > len(self.hour_weights) or start >= end:
            raise ValueError(f"Invalid target interval [{start}, {end})")
        if limit is not None:
            end = min(end, start + limit)

        weighted_at_window_end = np.zeros((N_NODES, N_NODES), dtype=np.float32)
        for hour in range(start - self.input_hours, start):
            age = start - 1 - hour
            weighted_at_window_end += np.asarray(
                self.hour_weights[hour], dtype=np.float32
            ) * np.float32(Q**age)

        q_window = np.float32(Q**self.input_hours)
        for target_hour in range(start, end):
            last_absolute = np.max(
                np.asarray(
                    self.hour_last[target_hour - self.input_hours : target_hour],
                    dtype=np.float64,
                ),
                axis=0,
            )
            active = last_absolute >= 0
            row_scale = np.zeros(N_NODES, dtype=np.float32)
            row_scale[active] = np.exp(
                (target_hour * 3600.0 - last_absolute[active]) / LAMBDA_SECONDS
            ).astype(np.float32)
            normalized_to_last_event = weighted_at_window_end * row_scale[:, None]
            relative_last = np.zeros(N_NODES, dtype=np.float32)
            relative_last[active] = (
                last_absolute[active]
                - (target_hour - self.input_hours) * 3600.0
            ).astype(np.float32)
            if valid_target_hour(target_hour, self.input_hours):
                yield target_hour, normalized_to_last_event, relative_last

            if target_hour + 1 < end:
                weighted_at_window_end *= np.float32(Q)
                weighted_at_window_end += np.asarray(
                    self.hour_weights[target_hour], dtype=np.float32
                )
                weighted_at_window_end -= q_window * np.asarray(
                    self.hour_weights[target_hour - self.input_hours], dtype=np.float32
                )
                np.maximum(weighted_at_window_end, 0.0, out=weighted_at_window_end)


def grouped(iterator, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_train_epoch(
    model: ContinuousWindowCMOD,
    criterion: ODLoss,
    optimizer: torch.optim.Optimizer,
    stream: WindowStream,
    od: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model.train()
    total_loss = 0.0
    count = 0
    started = time.time()
    iterator = stream.iterate(args.input_hours, TRAIN_END, args.max_train_targets)
    for batch in grouped(iterator, args.batch_size):
        target_hours = [item[0] for item in batch]
        weight_tensor = torch.from_numpy(
            np.stack([item[1] for item in batch]).astype(np.float32, copy=False)
        ).to(device=device)
        last_tensor = torch.from_numpy(
            np.stack([item[2] for item in batch]).astype(np.float32, copy=False)
        ).to(device=device)
        target = torch.from_numpy(
            np.stack(
                [np.asarray(od[target_hour], dtype=np.float32) for target_hour in target_hours]
            )
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = torch.stack(
            [
                model.encode_window(weight_tensor[index], last_tensor[index])
                for index in range(len(batch))
            ]
        )
        prediction = model.decode_embeddings(embeddings)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at target hour {target_hours[-1]}"
            )
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        count += len(batch)
        if count == len(batch) or count % args.log_every < len(batch):
            elapsed = time.time() - started
            print(
                f"train target={target_hours[-1]} samples={count} loss={loss.item():.6f} "
                f"mean_loss={total_loss / count:.6f} speed={count / elapsed:.2f} sample/s",
                flush=True,
            )
    return {
        "loss": total_loss / count,
        "samples": count,
        "seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate(
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
    metrics = StreamingMetrics()
    count = 0
    started = time.time()
    iterator = stream.iterate(start, end, limit)
    for batch in grouped(iterator, batch_size):
        target_hours = [item[0] for item in batch]
        weight_tensor = torch.from_numpy(
            np.stack([item[1] for item in batch]).astype(np.float32, copy=False)
        ).to(device=device)
        last_tensor = torch.from_numpy(
            np.stack([item[2] for item in batch]).astype(np.float32, copy=False)
        ).to(device=device)
        embeddings = torch.stack(
            [
                model.encode_window(weight_tensor[index], last_tensor[index])
                for index in range(len(batch))
            ]
        )
        prediction = model.decode_embeddings(embeddings)
        prediction = torch.clamp(prediction, min=0.0)
        target = torch.from_numpy(
            np.stack(
                [np.asarray(od[target_hour], dtype=np.float32) for target_hour in target_hours]
            )
        ).to(device=device)
        metrics.update(prediction, target)
        count += len(batch)
        if count == len(batch) or count % log_every < len(batch):
            print(
                f"{label} target={target_hours[-1]} samples={count} "
                f"speed={count / (time.time() - started):.2f} sample/s",
                flush=True,
            )
    result = metrics.compute()
    result.update(
        {
            "target_start_hour": start,
            "target_end_hour_exclusive": min(end, start + limit) if limit else end,
            "n_target_hours": count,
            "seconds": time.time() - started,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if args.input_hours not in {1, 3, 6, 12}:
        raise ValueError("--input-hours must be one of 1, 3, 6, 12")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

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
        raise ValueError(f"Unexpected continuous cache shape: {hour_weights.shape}")
    if od.shape != (N_HOURS, N_NODES, N_NODES):
        raise ValueError(f"Unexpected OD matrix shape: {od.shape}")

    stream = WindowStream(hour_weights, hour_last, args.input_hours)
    model = ContinuousWindowCMOD(
        device=args.device,
        n_nodes=N_NODES,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        lambda_seconds=LAMBDA_SECONDS,
        window_seconds=args.input_hours * 3600.0,
    ).to(device)
    criterion = ODLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = {
        "task": f"past {args.input_hours} hours -> next 1 hour, full 263x263 OD",
        "year": 2023,
        "input_source": "exact individual pickup timestamps and OD endpoints",
        "input_window_hours": args.input_hours,
        "forecast_horizon_hours": 1,
        "split": {
            "train_targets": [args.input_hours, TRAIN_END],
            "validation_targets": [TRAIN_END, VAL_END],
            "test_targets": [VAL_END, N_HOURS],
        },
        "continuous_aggregation": "CMOD exponential message aggregation, exact sufficient statistics",
        "lambda_seconds": LAMBDA_SECONDS,
        "model_selection": "lowest validation raw-count MAE",
        "loss": "original CMOD zero-aware OD loss",
        "gradient_clip": args.gradient_clip,
        "args": vars(args),
    }
    config["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in config["args"].items()}
    save_json(args.output_dir / "config.json", config)

    history_path = args.output_dir / "history.json"
    history = []
    best_mae = math.inf
    best_epoch = -1
    rounds = 0
    checkpoint = args.output_dir / "best_model.pt"
    start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        best_epoch = int(resume["epoch"])
        best_mae = float(resume["validation"]["mae"])
        start_epoch = best_epoch + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(
            f"resumed checkpoint={args.resume} best_epoch={best_epoch} "
            f"best_val_MAE={best_mae:.6f}",
            flush=True,
        )
    overall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}", flush=True)
        train_result = run_train_epoch(
            model, criterion, optimizer, stream, od, device, args
        )
        validation = evaluate(
            model,
            stream,
            od,
            device,
            TRAIN_END,
            VAL_END,
            args.max_eval_targets,
            "validation",
            args.log_every,
            args.batch_size,
        )
        record = {"epoch": epoch, "train": train_result, "validation": validation}
        history.append(record)
        save_json(history_path, history)
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.6f} "
            f"val_MAE={validation['mae']:.6f} val_RMSE={validation['rmse']:.6f} "
            f"val_PCC={validation['pcc']:.6f}",
            flush=True,
        )
        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            best_epoch = epoch
            rounds = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "config": config,
                },
                checkpoint,
            )
        else:
            rounds += 1
            if rounds >= args.patience:
                print(f"early stopping; best epoch={best_epoch}", flush=True)
                break

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["model"])
    validation_best = evaluate(
        model,
        stream,
        od,
        device,
        TRAIN_END,
        VAL_END,
        args.max_eval_targets,
        "best-validation-reeval",
        args.log_every,
        args.batch_size,
    )
    test = evaluate(
        model,
        stream,
        od,
        device,
        VAL_END,
        N_HOURS,
        args.max_eval_targets,
        "test",
        args.log_every,
        args.batch_size,
    )
    summary = {
        "model": f"CMOD-{args.input_hours}h-continuous",
        "best_epoch": int(best["epoch"]),
        "best_checkpoint": str(checkpoint.resolve()),
        "validation": validation_best,
        "test": test,
        "training_seconds": time.time() - overall_start,
        "full_run": args.max_train_targets is None and args.max_eval_targets is None,
        "protocol": config,
    }
    save_json(args.output_dir / "test_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
