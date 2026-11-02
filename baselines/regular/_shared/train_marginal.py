"""Unified audited runner for the locally supplied seven-baseline extension.

Only methods whose real upstream core and required inputs have passed audit are
registered here.  A missing result is preferable to a generic substitute under
an upstream method's name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_NODES, history_indices, iter_hour_batches, load_od, load_static
from ehod.common.device import require_cuda
from ehod.evaluation.metrics import StreamingMetrics
from baselines.regular._shared.mipf import MIPFDiagnostics, reconstruct_od
from baselines.regular._shared.marginal_data import PROTOCOL_VERSION, inverse_marginal, load_marginal_cache
from baselines.regular._shared.upstream_models import create_duet, create_lightst, create_patchstg, create_stdn


METHODS = ("stdn_mipf", "patchstg_mipf", "duet_mipf", "lightst_mipf")
from ehod.common.paths import EXPERIMENT_ROOT as REPOSITORY_ROOT
SOURCE_LOCK = REPOSITORY_ROOT / "baselines/sources_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--history-hours", type=int, choices=(1, 3, 6, 12), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-hours", type=int, default=None)
    parser.add_argument("--max-eval-hours", type=int, default=None)
    parser.add_argument("--mipf-max-iterations", type=int, default=200)
    parser.add_argument("--mipf-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_lock_sha256() -> str:
    return hashlib.sha256(SOURCE_LOCK.read_bytes()).hexdigest()


def laplacian_position_encoding(adjacency: np.ndarray, width: int = 32) -> torch.Tensor:
    value = np.maximum(np.asarray(adjacency, dtype=np.float64), 0.0)
    value = 0.5 * (value + value.T)
    degree = value.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 1.0e-12
    inverse[positive] = degree[positive] ** -0.5
    laplacian = np.eye(len(value)) - inverse[:, None] * value * inverse[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    result = vectors[:, 1 : width + 1]
    if result.shape[1] < width:
        result = np.pad(result, ((0, 0), (0, width - result.shape[1])))
    return torch.from_numpy(result.astype(np.float32))


def make_model(
    method: str, history_hours: int, device: torch.device
) -> Tuple[torch.nn.Module, dict]:
    if method == "duet_mipf":
        return create_duet(history_hours, 2 * N_NODES)
    static = load_static()
    if method == "stdn_mipf":
        positional = laplacian_position_encoding(static["adjacency"])
        return create_stdn(history_hours, N_NODES, positional, device)
    if method == "lightst_mipf":
        return create_lightst(
            history_hours,
            N_NODES,
            np.asarray(static["affinity"], dtype=np.float32),
            device,
        )
    if method == "patchstg_mipf":
        from ehod.data.od import ZONE_PATH

        zones = pd.read_csv(ZONE_PATH).sort_values("matrix_index")
        locations = zones[["centroid_lat", "centroid_lon"]].to_numpy(np.float32)
        return create_patchstg(
            history_hours,
            N_NODES,
            locations,
            np.asarray(static["affinity"], dtype=np.float32),
            device,
        )
    raise KeyError(method)


def history_tensor(
    normalized: np.ndarray,
    starts: np.ndarray,
    history_hours: int,
    device: torch.device,
) -> torch.Tensor:
    indices = history_indices(starts, history_hours)
    value = np.asarray(normalized[indices], dtype=np.float32)
    return torch.from_numpy(value).to(device, non_blocking=True)


def target_tensor(
    normalized: np.ndarray, starts: np.ndarray, device: torch.device
) -> torch.Tensor:
    value = np.asarray(normalized[starts], dtype=np.float32)
    return torch.from_numpy(value).to(device, non_blocking=True)


def temporal_tensor(
    method: str,
    starts: np.ndarray,
    history_hours: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    starts = np.asarray(starts, dtype=np.int64)
    if method == "stdn_mipf":
        # H observed calendar steps followed by H known future calendar steps;
        # the wrapper reports only the first predicted step.
        offset = np.arange(-history_hours, history_hours, dtype=np.int64)
        hour = starts[:, None] + offset[None, :]
        value = np.stack(((hour // 24) % 7, hour % 24), axis=-1)
        return torch.from_numpy(value).long().to(device, non_blocking=True)
    if method == "patchstg_mipf":
        offset = np.arange(-history_hours, 0, dtype=np.int64)
        hour = starts[:, None] + offset[None, :]
        value = np.stack((hour % 24, (hour // 24) % 7), axis=-1)
        value = np.repeat(value[:, :, None, :], N_NODES, axis=2)
        return torch.from_numpy(value).long().to(device, non_blocking=True)
    return None


def forward_model(
    method: str,
    model: torch.nn.Module,
    history: torch.Tensor,
    temporal: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "duet_mipf":
        flat = history.reshape(history.shape[0], history.shape[1], 2 * N_NODES)
        prediction, auxiliary = model(flat)
        prediction = prediction[:, -1].reshape(len(history), N_NODES, 2)
        return prediction, auxiliary
    if method == "lightst_mipf":
        prediction = model(history)
        return prediction, prediction.new_zeros(())
    if temporal is None:
        raise ValueError("Temporal input is required for %s" % method)
    prediction = model(history, temporal)[:, 0]
    return prediction, prediction.new_zeros(())


def inverse_marginal_torch(
    value: torch.Tensor, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    """Differentiable training inverse; nonnegativity is enforced at inference."""

    center = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
    spread = torch.as_tensor(std, dtype=value.dtype, device=value.device)
    # The upstream raw-scale losses must retain a gradient when a randomly
    # initialized prediction maps below zero.  Both clamping and the negative
    # tail of expm1 can create an effectively dead branch.  Extend expm1
    # linearly below log-count zero; the value and first derivative match at
    # zero, while `inverse_marginal` still enforces nonnegative inference.
    log_count = value * spread + center
    return torch.where(log_count >= 0.0, torch.expm1(log_count), log_count)


def duet_learning_rate_factor(scheduler_epoch: int) -> float:
    """Released DUET type3 schedule, called after each completed epoch."""

    if scheduler_epoch < 0:
        raise ValueError("scheduler_epoch must be nonnegative")
    return 1.0 if scheduler_epoch < 3 else 0.9 ** (scheduler_epoch - 3)


def training_objective(
    method: str,
    prediction_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    target_raw: torch.Tensor,
    auxiliary: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "duet_mipf":
        primary = F.huber_loss(prediction_scaled, target_scaled, delta=0.5)
        return primary + auxiliary, primary
    if method == "lightst_mipf":
        primary = F.l1_loss(prediction_scaled, target_scaled)
        return primary, primary
    prediction_raw = inverse_marginal_torch(prediction_scaled, mean, std)
    absolute = torch.abs(prediction_raw - target_raw)
    if method == "patchstg_mipf":
        mask = (target_raw != 0.0).float()
        mask = mask / mask.mean().clamp_min(1.0e-6)
        primary = torch.nan_to_num(absolute * mask).mean()
    else:
        primary = absolute.mean()
    return primary, primary


def lightst_spatial_contrastive(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """The released LightST spatial contrastive loss, per marginal."""

    student = F.normalize(student, p=2, dim=-1)
    teacher = F.normalize(teacher, p=2, dim=-1)
    weight = F.cosine_similarity(student, teacher, dim=-1)
    joined = torch.cat((student, teacher), dim=0)
    negative = torch.exp(
        torch.matmul(joined, joined.transpose(2, 3).contiguous()) / temperature
    )
    positive = torch.exp(
        torch.sum(student * teacher, dim=-1) * weight / temperature
    )
    positive = torch.cat((positive, positive), dim=0).sum(dim=1)
    denominator = negative.sum(dim=-1).sum(dim=1)
    return (-torch.log(positive / (positive + denominator))).mean()


def lightst_distillation_objective(
    model, history, target, target_raw, mean, std
):
    student_prediction, student_embedding = model.student_forward(history)
    with torch.no_grad():
        teacher_prediction, _, teacher_spatial = model.teacher_forward(history)
    primary = F.l1_loss(
        inverse_marginal_torch(student_prediction, mean, std), target_raw
    )
    # The original 12-step KL uses the temporal output axis.  That axis has
    # length one in this task, so the auditable one-step adaptation distils the
    # node distribution independently for the two marginal channels.
    student_log = F.log_softmax(student_prediction, dim=1)
    teacher_log = F.log_softmax(teacher_prediction, dim=1)
    kl = F.kl_div(student_log, teacher_log, reduction="batchmean", log_target=True)
    contrastive = 0.5 * (
        lightst_spatial_contrastive(student_embedding[0], teacher_spatial[0])
        + lightst_spatial_contrastive(student_embedding[1], teacher_spatial[1])
    )
    auxiliary = 10.0 * kl + contrastive
    return primary + auxiliary, primary, auxiliary, kl, contrastive


@torch.no_grad()
def lightst_teacher_validation(model, cache, args, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for starts in iter_hour_batches(
        "validation", args.eval_batch_size, limit=args.max_eval_hours
    ):
        value = history_tensor(cache["normalized"], starts, args.history_hours, device)
        target_raw = torch.from_numpy(
            np.asarray(cache["raw"][starts], dtype=np.float32)
        ).to(device, non_blocking=True)
        prediction = model.teacher_forward(value)[0]
        loss = F.l1_loss(
            inverse_marginal_torch(prediction, cache["mean"], cache["std"]),
            target_raw,
        )
        total += float(loss.item()) * len(starts)
        count += len(starts)
    return total / max(count, 1)


def pretrain_lightst_teacher(model, cache, args, output_dir, device) -> dict:
    parameters = list(model.teacher_origin.parameters()) + list(
        model.teacher_destination.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 45, 60, 75], gamma=0.3
    )
    checkpoint = output_dir / "teacher_best.pt"
    best = math.inf
    stale = 0
    rows = []
    started = time.time()
    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for starts in iter_hour_batches(
            "train",
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            limit=args.max_train_hours,
        ):
            value = history_tensor(cache["normalized"], starts, args.history_hours, device)
            target_raw = torch.from_numpy(
                np.asarray(cache["raw"][starts], dtype=np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.teacher_forward(value)[0]
            loss = F.l1_loss(
                inverse_marginal_torch(prediction, cache["mean"], cache["std"]),
                target_raw,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite LightST teacher loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(starts)
            count += len(starts)
        validation = lightst_teacher_validation(model, cache, args, device)
        row = {
            "epoch": epoch,
            "teacher_train_raw_mae": total / max(count, 1),
            "teacher_validation_raw_mae": validation,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if validation < best - 1.0e-8:
            best = validation
            stale = 0
            torch.save(
                {
                    "origin": model.teacher_origin.state_dict(),
                    "destination": model.teacher_destination.state_dict(),
                    "epoch": epoch,
                },
                checkpoint,
            )
        else:
            stale += 1
        scheduler.step()
        (output_dir / "teacher_training_history.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if stale >= args.patience:
            break
    payload = torch.load(checkpoint, map_location=device)
    model.teacher_origin.load_state_dict(payload["origin"])
    model.teacher_destination.load_state_dict(payload["destination"])
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.teacher_origin.eval()
    model.teacher_destination.eval()
    return {
        "best_epoch": int(payload["epoch"]),
        "epochs_run": len(rows),
        "best_validation_raw_mae": best,
        "seconds": time.time() - started,
    }


def predict_marginal(
    method: str,
    model: torch.nn.Module,
    normalized: np.ndarray,
    starts: np.ndarray,
    history_hours: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    value = history_tensor(normalized, starts, history_hours, device)
    temporal = temporal_tensor(method, starts, history_hours, device)
    prediction, auxiliary = forward_model(method, model, value, temporal)
    prediction = prediction.float()
    prediction_raw = inverse_marginal(prediction.detach().cpu().numpy(), mean, std)
    return prediction_raw, auxiliary


@torch.no_grad()
def evaluate_split(
    method: str,
    model: torch.nn.Module,
    cache: Dict[str, np.ndarray],
    split: str,
    history_hours: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    mipf_max_iterations: int,
    mipf_tolerance: float,
    limit: Optional[int] = None,
    collect_predictions: bool = False,
) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    model.eval()
    # DUET samples its Gumbel channel mask even in eval mode.  A fixed RNG and
    # fixed sample order make validation selection and the frozen cache exactly
    # reproducible.  The policy is reported rather than hidden.
    set_seed(seed)
    marginal_metrics = StreamingMetrics()
    od_metrics = StreamingMetrics()
    od = load_od()
    seasonal_prior = np.asarray(load_static()["seasonal_mean"], dtype=np.float32)
    collected = []
    collected_hours = []
    nonconverged = 0
    maximum_row_error = 0.0
    maximum_column_error = 0.0
    iterations_total = 0
    number_of_batches = 0
    started = time.perf_counter()
    for starts in iter_hour_batches(split, batch_size, limit=limit):
        prediction_marginal, _ = predict_marginal(
            method,
            model,
            cache["normalized"],
            starts,
            history_hours,
            cache["mean"],
            cache["std"],
            device,
        )
        target_marginal = np.asarray(cache["raw"][starts], dtype=np.float32)
        marginal_metrics.update(
            torch.from_numpy(prediction_marginal), torch.from_numpy(target_marginal)
        )
        prediction_od, diagnostics = reconstruct_od(
            seasonal_prior[starts % 168],
            prediction_marginal,
            max_iterations=mipf_max_iterations,
            tolerance=mipf_tolerance,
            return_diagnostics=True,
        )
        assert isinstance(diagnostics, MIPFDiagnostics)
        target_od = np.asarray(od[starts], dtype=np.float32)
        od_metrics.update(torch.from_numpy(prediction_od), torch.from_numpy(target_od))
        if not diagnostics.converged:
            nonconverged += len(starts)
        maximum_row_error = max(maximum_row_error, diagnostics.max_row_error)
        maximum_column_error = max(maximum_column_error, diagnostics.max_column_error)
        iterations_total += diagnostics.iterations
        number_of_batches += 1
        if collect_predictions:
            collected.append(prediction_od.astype(np.float16))
            collected_hours.append(np.asarray(starts, dtype=np.int32))
    hours = (
        np.concatenate(collected_hours)
        if collected_hours
        else np.concatenate(list(iter_hour_batches(split, batch_size, limit=limit)))
    )
    predictions = np.concatenate(collected) if collected else None
    result = {
        "od": od_metrics.compute(),
        "marginal": marginal_metrics.compute(),
        "mipf": {
            "max_iterations": mipf_max_iterations,
            "relative_tolerance": mipf_tolerance,
            "nonconverged_samples": nonconverged,
            "max_row_error": maximum_row_error,
            "max_column_error": maximum_column_error,
            "mean_batch_iterations": iterations_total / max(number_of_batches, 1),
        },
        "seconds": time.perf_counter() - started,
        "evaluation_random_seed": seed,
    }
    return result, predictions, hours


def train(args: argparse.Namespace, output_dir: Path, device: torch.device) -> dict:
    cache = load_marginal_cache()
    model, model_audit = make_model(args.method, args.history_hours, device)
    model = model.to(device)
    teacher_training = None
    if args.method == "lightst_mipf":
        teacher_training = pretrain_lightst_teacher(
            model, cache, args, output_dir, device
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.method in ("stdn_mipf", "duet_mipf", "lightst_mipf"):
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    checkpoint = output_dir / "best.pt"
    best_validation = math.inf
    stale = 0
    history = []
    training_started = time.time()
    scheduler = None
    if args.method == "lightst_mipf":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[25, 45, 60, 75], gamma=0.3
        )
    elif args.method == "stdn_mipf":
        # Released STDN configuration: decay_epoch=10, gamma=0.9.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9
        )
    elif args.method == "duet_mipf":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=duet_learning_rate_factor
        )
    elif args.method == "patchstg_mipf":
        # Exact schedule in the released PatchSTG training entry point.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 35, 40], gamma=0.5
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.method == "lightst_mipf":
            model.teacher_origin.eval()
            model.teacher_destination.eval()
        total_loss = 0.0
        total_primary = 0.0
        total_auxiliary = 0.0
        total_kl = 0.0
        total_contrastive = 0.0
        count = 0
        epoch_started = time.time()
        for starts in iter_hour_batches(
            "train",
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            limit=args.max_train_hours,
        ):
            value = history_tensor(
                cache["normalized"], starts, args.history_hours, device
            )
            temporal = temporal_tensor(args.method, starts, args.history_hours, device)
            target = target_tensor(cache["normalized"], starts, device)
            target_raw = torch.from_numpy(
                np.asarray(cache["raw"][starts], dtype=np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.method == "lightst_mipf":
                loss, primary, auxiliary, kl, contrastive = (
                    lightst_distillation_objective(
                        model,
                        value,
                        target,
                        target_raw,
                        cache["mean"],
                        cache["std"],
                    )
                )
            else:
                prediction, auxiliary = forward_model(
                    args.method, model, value, temporal
                )
                loss, primary = training_objective(
                    args.method,
                    prediction,
                    target,
                    target_raw,
                    auxiliary,
                    cache["mean"],
                    cache["std"],
                )
                kl = loss.new_zeros(())
                contrastive = loss.new_zeros(())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(starts)
            total_primary += float(primary.detach().item()) * len(starts)
            total_auxiliary += float(auxiliary.detach().item()) * len(starts)
            total_kl += float(kl.detach().item()) * len(starts)
            total_contrastive += float(contrastive.detach().item()) * len(starts)
            count += len(starts)

        validation, _, _ = evaluate_split(
            args.method,
            model,
            cache,
            "validation",
            args.history_hours,
            args.eval_batch_size,
            device,
            seed=100000 + args.seed,
            mipf_max_iterations=args.mipf_max_iterations,
            mipf_tolerance=args.mipf_tolerance,
            limit=args.max_eval_hours,
        )
        validation_mae = float(validation["od"]["mae"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(count, 1),
            "train_primary_mae_loss": total_primary / max(count, 1),
            "train_router_auxiliary_loss": total_auxiliary / max(count, 1),
            "train_distillation_kl": total_kl / max(count, 1),
            "train_spatial_contrastive": total_contrastive / max(count, 1),
            "validation_full_od_mae": validation_mae,
            "validation_full_od_rmse": validation["od"]["rmse"],
            "validation_full_od_pcc": validation["od"]["pcc"],
            "mipf_nonconverged_samples": validation["mipf"]["nonconverged_samples"],
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_mae < best_validation - 1.0e-8:
            best_validation = validation_mae
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "model_audit": model_audit,
                },
                checkpoint,
            )
        else:
            stale += 1
        (output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if scheduler is not None:
            scheduler.step()
        if stale >= args.patience:
            break

    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    validation, _, validation_hours = evaluate_split(
        args.method,
        model,
        cache,
        "validation",
        args.history_hours,
        args.eval_batch_size,
        device,
        seed=100000 + args.seed,
        mipf_max_iterations=args.mipf_max_iterations,
        mipf_tolerance=args.mipf_tolerance,
        limit=args.max_eval_hours,
    )
    test = None
    prediction_cache = None
    if not args.validation_only:
        collect = args.history_hours == 12 and args.max_eval_hours is None
        test, predictions, test_hours = evaluate_split(
            args.method,
            model,
            cache,
            "test",
            args.history_hours,
            args.eval_batch_size,
            device,
            seed=200000 + args.seed,
            mipf_max_iterations=args.mipf_max_iterations,
            mipf_tolerance=args.mipf_tolerance,
            limit=args.max_eval_hours,
            collect_predictions=collect,
        )
        if predictions is not None:
            prediction_path = output_dir / "test_prediction_fp16.npy"
            hour_path = output_dir / "test_target_hours.npy"
            np.save(prediction_path, predictions)
            np.save(hour_path, test_hours)
            prediction_cache = {
                "prediction": str(prediction_path),
                "dtype": "float16",
                "shape": list(predictions.shape),
                "target_hours": str(hour_path),
                "paired_evaluation_seed": 200000 + args.seed,
            }
    return {
        "model": model,
        "model_audit": model_audit,
        "cache_root": str(cache["root"]),
        "best_epoch": int(payload["epoch"]),
        "epochs_run": len(history),
        "training_seconds": time.time() - training_started,
        "validation": validation,
        "validation_target_hours": [int(validation_hours[0]), int(validation_hours[-1]) + 1],
        "test": test,
        "prediction_cache": prediction_cache,
        "teacher_training": teacher_training,
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.teacher_epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.max_train_hours = 4
        args.max_eval_hours = 4
    set_seed(args.seed)
    device = require_cuda(args.device)
    suffix = "smoke" if args.smoke else "formal_lr_%g" % args.learning_rate
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "seven_baselines"
        / args.method
        / ("h%d_seed%d_%s" % (args.history_hours, args.seed, suffix))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return
    trained = train(args, output_dir, device)
    model = trained.pop("model")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "city": ACTIVE_CITY,
        "method": args.method,
        "display_name": {
            "stdn_mipf": "STDN-MIPF",
            "patchstg_mipf": "PatchSTG-MIPF",
            "duet_mipf": "DUET-MIPF",
            "lightst_mipf": "LightST-MIPF",
        }[args.method],
        "history_hours": args.history_hours,
        "prediction_horizon": 1,
        "seed": args.seed,
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "optimizer": {
            "name": (
                "Adam"
                if args.method in ("stdn_mipf", "duet_mipf", "lightst_mipf")
                else "AdamW"
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "scheduler": (
                {"name": "MultiStepLR", "milestones": [1, 35, 40], "gamma": 0.5}
                if args.method == "patchstg_mipf"
                else {
                    "name": "MultiStepLR",
                    "milestones": [25, 45, 60, 75],
                    "gamma": 0.3,
                }
                if args.method == "lightst_mipf"
                else {"name": "StepLR", "step_size": 10, "gamma": 0.9}
                if args.method == "stdn_mipf"
                else {
                    "name": "DUET-type3",
                    "constant_through_scheduler_epoch": 3,
                    "subsequent_epoch_factor": 0.9,
                }
                if args.method == "duet_mipf"
                else None
            ),
            "loss": {
                "stdn_mipf": "upstream raw-scale L1 marginal loss",
                "patchstg_mipf": "upstream zero-masked raw-scale MAE marginal loss",
                "duet_mipf": "upstream Huber(delta=0.5) plus unscaled DUET router load-balancing loss",
                "lightst_mipf": "upstream inverse-scaled raw MAE + 10x KL + 1x spatial contrastive distillation",
            }[args.method],
        },
        "selection": "minimum validation full raw-count OD MAE after common MIPF",
        "split": {"train": [12, 6720], "validation": [6720, 7728], "test": [7728, 8760]},
        "mipf": {
            "prior": "training-only mean OD by hour of week",
            "total_reconciliation": "mean of two predicted marginal totals",
            "max_iterations": args.mipf_max_iterations,
            "relative_tolerance": args.mipf_tolerance,
            "support_smoothing": 0.002,
        },
        "evaluation_random_policy": (
            "fixed split-specific RNG and fixed target-hour order for upstream Gumbel mask"
            if args.method == "duet_mipf"
            else "deterministic evaluation order; no stochastic evaluation layer"
        ),
        "source_lock": str(SOURCE_LOCK),
        "source_lock_sha256": source_lock_sha256(),
        **trained,
    }
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
