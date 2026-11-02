"""ST-ReP representation pretraining + original Ridge downstream OD adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_NODES, history_indices, iter_hour_batches, load_od, load_static, target_hours
from ehod.common.device import require_cuda
from ehod.evaluation.metrics import StreamingMetrics
from baselines.regular._shared.mipf import MIPFDiagnostics, reconstruct_od
from baselines.regular._shared.marginal_data import PROTOCOL_VERSION, inverse_marginal, load_marginal_cache
from baselines.regular._shared.upstream_models import create_strep


from ehod.common.paths import EXPERIMENT_ROOT as REPOSITORY_ROOT
SOURCE_LOCK = REPOSITORY_ROOT / "baselines/sources_lock.json"
RIDGE_ALPHAS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-hours", type=int, choices=(1, 3, 6, 12), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-ridge-rows", type=int, default=100000)
    parser.add_argument("--mipf-max-iterations", type=int, default=200)
    parser.add_argument("--mipf-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-train-hours", type=int, default=None)
    parser.add_argument("--max-eval-hours", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_tensors(cache, starts, history_hours, device):
    indices = history_indices(starts, history_hours)
    history = torch.from_numpy(
        np.asarray(cache["normalized"][indices], dtype=np.float32)
    ).to(device, non_blocking=True)
    target = torch.from_numpy(
        np.asarray(cache["normalized"][starts], dtype=np.float32)[:, None]
    ).to(device, non_blocking=True)
    observed_hour = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(
        -history_hours, 0, dtype=np.int64
    )[None, :]
    temporal = np.stack((observed_hour % 24, (observed_hour // 24) % 7), axis=-1)
    temporal = torch.from_numpy(temporal).long().to(device, non_blocking=True)
    return history, temporal, target


def huber(prediction: torch.Tensor, target: torch.Tensor, delta: float = 2.0):
    error = torch.abs(prediction - target)
    return torch.where(
        error <= delta, 0.5 * error.square(), delta * error - 0.5 * delta * delta
    ).mean()


def strep_learning_rate(
    epoch: int,
    base_learning_rate: float,
    total_epochs: int,
    warmup_epochs: int = 5,
    warmup_initial: float = 1.0e-6,
    decay_rate: float = 0.1,
) -> float:
    """Match the release's epoch-wise CosineLRScheduler exactly."""

    if epoch < 1 or total_epochs < 1:
        raise ValueError("epoch and total_epochs must be positive")
    # The source initializes the optimizer at warmup_initial, then calls
    # scheduler.step(i) after epoch i. Thus epoch e uses the value at t=e-1.
    t = epoch - 1
    if t < warmup_epochs:
        return warmup_initial + t * (
            (base_learning_rate - warmup_initial) / warmup_epochs
        )
    cycle = t // total_epochs
    within_cycle = t - cycle * total_epochs
    maximum = base_learning_rate * decay_rate**cycle
    return 0.5 * maximum * (
        1.0 + math.cos(math.pi * within_cycle / total_epochs)
    )


def pretraining_loss(outputs, history_hours: int):
    input_value, reconstruction, target, prediction = outputs
    prediction_loss = huber(prediction, target)
    reconstruction_loss = huber(reconstruction, input_value)
    ground = torch.cat((input_value, target), dim=1)
    reconstructed = torch.cat((reconstruction, prediction), dim=1)
    ground = ground.permute(0, 2, 3, 1).reshape(-1, 2, history_hours + 1)
    reconstructed = reconstructed.permute(0, 2, 3, 1).reshape(
        -1, 2, history_hours + 1
    )
    scales = [scale for scale in (2, 4, 8, 16) if scale <= history_hours + 1]
    if scales:
        ground_scales = [F.avg_pool1d(ground, scale, scale) for scale in scales]
        prediction_scales = [
            F.avg_pool1d(reconstructed, scale, scale) for scale in scales
        ]
        multiscale = huber(
            torch.cat(prediction_scales, dim=-1), torch.cat(ground_scales, dim=-1)
        )
    else:
        multiscale = prediction_loss.new_zeros(())
    # PEMS04 upstream configuration: alpha=0.4, beta=0.5.
    return (
        0.4 * prediction_loss + 0.5 * reconstruction_loss + 0.1 * multiscale,
        prediction_loss,
        reconstruction_loss,
        multiscale,
    )


@torch.no_grad()
def validation_pretrain_loss(model, cache, args, device) -> float:
    model.eval()
    set_seed(100000 + args.seed)
    total = 0.0
    count = 0
    for starts in iter_hour_batches(
        "validation", args.eval_batch_size, limit=args.max_eval_hours
    ):
        history, temporal, target = batch_tensors(
            cache, starts, args.history_hours, device
        )
        loss = pretraining_loss(model(history, temporal, target), args.history_hours)[0]
        total += float(loss.item()) * len(starts)
        count += len(starts)
    return total / max(count, 1)


def pretrain(model, cache, args, device, output_dir: Path) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    checkpoint = output_dir / "encoder_best.pt"
    best = math.inf
    stale = 0
    rows = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        learning_rate = strep_learning_rate(
            epoch, args.learning_rate, args.epochs
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        count = 0
        for starts in iter_hour_batches(
            "train",
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            limit=args.max_train_hours,
        ):
            history, temporal, target = batch_tensors(
                cache, starts, args.history_hours, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss_values = pretraining_loss(
                model(history, temporal, target), args.history_hours
            )
            loss_values[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += np.asarray([float(value.detach().item()) for value in loss_values]) * len(starts)
            count += len(starts)
        validation = validation_pretrain_loss(model, cache, args, device)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": totals[0] / max(count, 1),
            "train_prediction_loss": totals[1] / max(count, 1),
            "train_reconstruction_loss": totals[2] / max(count, 1),
            "train_multiscale_loss": totals[3] / max(count, 1),
            "validation_pretraining_loss": validation,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if validation < best - 1.0e-8:
            best = validation
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)
        else:
            stale += 1
        (output_dir / "pretraining_history.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if stale >= args.patience:
            break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    return {
        "best_epoch": int(payload["epoch"]),
        "epochs_run": len(rows),
        "best_validation_pretraining_loss": best,
        "seconds": time.time() - started,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": {
                "name": "CosineLRScheduler",
                "t_initial": args.epochs,
                "lr_min": 0.0,
                "decay_rate": 0.1,
                "warmup_epochs": 5,
                "warmup_initial": 1.0e-6,
            },
        },
    }


@torch.no_grad()
def sampled_representation_rows(
    model,
    cache,
    split: str,
    args,
    device,
    maximum_rows: int,
):
    model.eval()
    hours = target_hours(split)
    limit = args.max_train_hours if split == "train" else args.max_eval_hours
    if limit is not None:
        hours = hours[:limit]
    total = len(hours) * N_NODES
    rng = np.random.default_rng(args.seed + (17 if split == "train" else 29))
    selected = np.sort(
        rng.choice(total, size=min(maximum_rows, total), replace=False).astype(np.int64)
    )
    origin_features = []
    destination_features = []
    origin_target = []
    destination_target = []
    offset = 0
    for begin in range(0, len(hours), args.eval_batch_size):
        starts = hours[begin : begin + args.eval_batch_size]
        history, temporal, target = batch_tensors(
            cache, starts, args.history_hours, device
        )
        origin, destination = model.representations(history, temporal)
        origin = origin[:, 0].cpu().numpy().reshape(-1, origin.shape[-1])
        destination = destination[:, 0].cpu().numpy().reshape(-1, destination.shape[-1])
        target = target[:, 0].cpu().numpy().reshape(-1, 2)
        stop = offset + len(starts) * N_NODES
        lo = np.searchsorted(selected, offset, side="left")
        hi = np.searchsorted(selected, stop, side="left")
        local = selected[lo:hi] - offset
        if len(local):
            origin_features.append(origin[local])
            destination_features.append(destination[local])
            origin_target.append(target[local, 0])
            destination_target.append(target[local, 1])
        offset = stop
    return (
        np.concatenate(origin_features),
        np.concatenate(destination_features),
        np.concatenate(origin_target),
        np.concatenate(destination_target),
    )


def fit_ridge_models(model, cache, args, device, output_dir: Path):
    train = sampled_representation_rows(
        model, cache, "train", args, device, args.max_ridge_rows
    )
    validation = sampled_representation_rows(
        model, cache, "validation", args, device, args.max_ridge_rows
    )
    models = []
    audit = []
    for channel, (train_x, valid_x, train_y, valid_y) in enumerate(
        (
            (train[0], validation[0], train[2], validation[2]),
            (train[1], validation[1], train[3], validation[3]),
        )
    ):
        scores = []
        for alpha in RIDGE_ALPHAS:
            candidate = Ridge(alpha=alpha).fit(train_x, train_y)
            prediction = candidate.predict(valid_x)
            error = prediction - valid_y
            score = float(np.sqrt(np.square(error).mean()) + np.abs(error).mean())
            scores.append({"alpha": alpha, "normalized_rmse_plus_mae": score})
        selected = min(scores, key=lambda row: row["normalized_rmse_plus_mae"])
        ridge = Ridge(alpha=selected["alpha"]).fit(train_x, train_y)
        models.append(ridge)
        audit.append(
            {
                "channel": "origin" if channel == 0 else "destination",
                "train_rows": len(train_y),
                "validation_rows": len(valid_y),
                "selected_alpha": selected["alpha"],
                "validation_grid": scores,
            }
        )
    with (output_dir / "ridge_models.pkl").open("wb") as stream:
        pickle.dump(models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return models, audit


@torch.no_grad()
def evaluate(
    model,
    ridges,
    cache,
    split,
    args,
    device,
    collect_predictions=False,
):
    model.eval()
    od = load_od()
    prior = np.asarray(load_static()["seasonal_mean"], dtype=np.float32)
    od_metrics = StreamingMetrics()
    marginal_metrics = StreamingMetrics()
    predictions = []
    hours_all = []
    nonconverged = 0
    iterations = []
    for starts in iter_hour_batches(
        split, args.eval_batch_size, limit=args.max_eval_hours
    ):
        history, temporal, _ = batch_tensors(cache, starts, args.history_hours, device)
        origin, destination = model.representations(history, temporal)
        origin = ridges[0].predict(origin[:, 0].cpu().numpy().reshape(-1, 64))
        destination = ridges[1].predict(
            destination[:, 0].cpu().numpy().reshape(-1, 64)
        )
        scaled = np.stack(
            (origin.reshape(len(starts), N_NODES), destination.reshape(len(starts), N_NODES)),
            axis=-1,
        ).astype(np.float32)
        marginal = inverse_marginal(scaled, cache["mean"], cache["std"])
        target_marginal = np.asarray(cache["raw"][starts], dtype=np.float32)
        marginal_metrics.update(torch.from_numpy(marginal), torch.from_numpy(target_marginal))
        prediction, diagnostics = reconstruct_od(
            prior[starts % 168],
            marginal,
            max_iterations=args.mipf_max_iterations,
            tolerance=args.mipf_tolerance,
            return_diagnostics=True,
        )
        assert isinstance(diagnostics, MIPFDiagnostics)
        nonconverged += 0 if diagnostics.converged else len(starts)
        iterations.append(diagnostics.iterations)
        target = np.asarray(od[starts], dtype=np.float32)
        od_metrics.update(torch.from_numpy(prediction), torch.from_numpy(target))
        if collect_predictions:
            predictions.append(prediction.astype(np.float16))
            hours_all.append(starts.astype(np.int32))
    return {
        "od": od_metrics.compute(),
        "marginal": marginal_metrics.compute(),
        "mipf": {
            "nonconverged_samples": nonconverged,
            "mean_batch_iterations": float(np.mean(iterations)) if iterations else 0.0,
        },
    }, (np.concatenate(predictions) if predictions else None), (
        np.concatenate(hours_all) if hours_all else None
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.max_train_hours = 4
        args.max_eval_hours = 4
        args.max_ridge_rows = 128
    set_seed(args.seed)
    device = require_cuda(args.device)
    suffix = "smoke" if args.smoke else "formal"
    output_dir = (
        EXPERIMENT_ROOT
        / "runs/seven_baselines/strep_mipf"
        / ("h%d_seed%d_%s" % (args.history_hours, args.seed, suffix))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return
    cache = load_marginal_cache()
    model, model_audit = create_strep(args.history_hours, N_NODES, device)
    pretraining = pretrain(model, cache, args, device, output_dir)
    ridge_started = time.time()
    ridges, ridge_audit = fit_ridge_models(model, cache, args, device, output_dir)
    ridge_seconds = time.time() - ridge_started
    validation, _, _ = evaluate(model, ridges, cache, "validation", args, device)
    test = None
    prediction_cache = None
    if not args.validation_only:
        collect = args.history_hours == 12 and args.max_eval_hours is None
        test, prediction, hours = evaluate(
            model, ridges, cache, "test", args, device, collect_predictions=collect
        )
        if prediction is not None:
            prediction_path = output_dir / "test_prediction_fp16.npy"
            hours_path = output_dir / "test_target_hours.npy"
            np.save(prediction_path, prediction)
            np.save(hours_path, hours)
            prediction_cache = {
                "prediction": str(prediction_path),
                "target_hours": str(hours_path),
                "dtype": "float16",
                "shape": list(prediction.shape),
            }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "city": ACTIVE_CITY,
        "method": "strep_mipf",
        "display_name": "ST-ReP-MIPF",
        "history_hours": args.history_hours,
        "prediction_horizon": 1,
        "seed": args.seed,
        "device": str(device),
        "model_audit": model_audit,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretraining": pretraining,
        "ridge": ridge_audit,
        "ridge_seconds": ridge_seconds,
        "selection": {
            "encoder": "minimum upstream validation pretraining loss",
            "ridge": "minimum validation normalized RMSE+MAE on upstream alpha grid",
            "final_audit_metric": "validation full raw-count OD MAE after common MIPF",
        },
        "validation": validation,
        "test": test,
        "prediction_cache": prediction_cache,
        "split": {"train": [12, 6720], "validation": [6720, 7728], "test": [7728, 8760]},
        "source_lock": str(SOURCE_LOCK),
        "source_lock_sha256": hashlib.sha256(SOURCE_LOCK.read_bytes()).hexdigest(),
    }
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
