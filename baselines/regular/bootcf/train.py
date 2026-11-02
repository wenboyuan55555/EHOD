#!/usr/bin/env python3
"""Full-year BootCF adaptation for strict variable-history hurdle prediction."""

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
import torch.nn.functional as F
from torch import nn

from baselines.regular.bootcf.model import (
    MLP_Predictor,
    PAPER_CODE,
    PairMagnitudeMLP,
    WindowBootCF,
    make_target_encoder,
)
from baselines.regular._shared.continuous_evaluation import evaluate_hurdle_model


N_HOURS = 8760
N_NODES = int(os.environ.get("EHOD_N_NODES", "263"))
TRAIN_END = 6720
VAL_END = 7728
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


def valid_target_array(values: np.ndarray, input_hours: int) -> np.ndarray:
    return values[
        np.asarray(
            [valid_target_hour(int(value), input_hours) for value in values],
            dtype=bool,
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-hours", type=int, default=12)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--message-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-lr", type=float, default=1e-5)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--augmentation-noise", type=float, default=0.15)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--probe-stride", type=int, default=12)
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--downstream-epochs", type=int, default=15)
    parser.add_argument("--downstream-patience", type=int, default=4)
    parser.add_argument("--downstream-batch-size", type=int, default=8)
    parser.add_argument("--downstream-lr", type=float, default=1e-3)
    parser.add_argument("--destination-chunk", type=int, default=32)
    parser.add_argument("--max-pretrain-batches", type=int, default=None)
    parser.add_argument("--max-downstream-batches", type=int, default=None)
    parser.add_argument(
        "--cache-dir", type=Path, default=CONTINUOUS_CACHE_ROOT
    )
    parser.add_argument(
        "--od-matrix", type=Path, default=OD_PATH
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BENCHMARK_OUTPUT_ROOT / "runs" / "bootcf_hurdle"
    )
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument(
        "--resume-downstream",
        action="store_true",
        help="Resume the downstream head from best_od_magnitude.pt and downstream_history.json",
    )
    return parser.parse_args()


class ContinuousWindowStream:
    def __init__(self, cache_dir: Path, input_hours: int):
        self.input_hours = int(input_hours)
        self.hour_weights = np.load(
            cache_dir / "continuous_hour_exp_weights.npy", mmap_mode="r"
        )
        self.hour_origin_last = np.load(
            cache_dir / "continuous_hour_last_pickup_second.npy", mmap_mode="r"
        )
        self.hour_destination_last = np.load(
            cache_dir / "continuous_hour_destination_last_pickup_second.npy",
            mmap_mode="r",
        )

    def iterate(self, start: int, end: int, batch_size: int, limit_batches=None):
        if start < self.input_hours:
            raise ValueError(
                f"Target start {start} precedes {self.input_hours}h input window"
            )
        weighted_end = np.zeros((N_NODES, N_NODES), dtype=np.float32)
        for hour in range(start - self.input_hours, start):
            weighted_end += np.asarray(self.hour_weights[hour], dtype=np.float32) * np.float32(
                Q ** (start - 1 - hour)
            )
        q_window = np.float32(Q**self.input_hours)
        forward_batch, reverse_batch, origin_batch, destination_batch, starts_batch = [], [], [], [], []
        yielded = 0
        for target in range(start, end):
            origin_last = np.max(
                np.asarray(
                    self.hour_origin_last[
                        target - self.input_hours : target
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )
            destination_last = np.max(
                np.asarray(
                    self.hour_destination_last[
                        target - self.input_hours : target
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )
            origin_active = origin_last >= 0
            destination_active = destination_last >= 0
            origin_scale = np.zeros(N_NODES, dtype=np.float32)
            destination_scale = np.zeros(N_NODES, dtype=np.float32)
            origin_scale[origin_active] = np.exp(
                (target * 3600.0 - origin_last[origin_active]) / 3600.0
            ).astype(np.float32)
            destination_scale[destination_active] = np.exp(
                (target * 3600.0 - destination_last[destination_active]) / 3600.0
            ).astype(np.float32)
            forward = weighted_end * origin_scale[:, None]
            reverse = weighted_end.T * destination_scale[:, None]
            relative_origin = np.zeros(N_NODES, dtype=np.float32)
            relative_destination = np.zeros(N_NODES, dtype=np.float32)
            relative_origin[origin_active] = (
                origin_last[origin_active]
                - (target - self.input_hours) * 3600.0
            ).astype(np.float32)
            relative_destination[destination_active] = (
                destination_last[destination_active]
                - (target - self.input_hours) * 3600.0
            ).astype(np.float32)
            if valid_target_hour(target, self.input_hours):
                forward_batch.append(forward)
                reverse_batch.append(reverse)
                origin_batch.append(relative_origin)
                destination_batch.append(relative_destination)
                starts_batch.append(target)
            if len(starts_batch) == batch_size:
                yield (
                    np.stack(forward_batch),
                    np.stack(reverse_batch),
                    np.stack(origin_batch),
                    np.stack(destination_batch),
                    np.asarray(starts_batch, dtype=np.int32),
                )
                yielded += 1
                if limit_batches is not None and yielded >= limit_batches:
                    return
                forward_batch, reverse_batch, origin_batch, destination_batch, starts_batch = [], [], [], [], []
            if target + 1 < end:
                weighted_end *= np.float32(Q)
                weighted_end += np.asarray(self.hour_weights[target], dtype=np.float32)
                weighted_end -= q_window * np.asarray(
                    self.hour_weights[target - self.input_hours], dtype=np.float32
                )
                np.maximum(weighted_end, 0.0, out=weighted_end)
        if starts_batch:
            yield (
                np.stack(forward_batch),
                np.stack(reverse_batch),
                np.stack(origin_batch),
                np.stack(destination_batch),
                np.asarray(starts_batch, dtype=np.int32),
            )


def to_device(batch, device):
    return [torch.from_numpy(value).to(device) for value in batch[:4]], batch[4]


@torch.no_grad()
def make_embeddings(model, stream, start, end, batch_size, device, selected=None):
    model.eval()
    selected_set = None if selected is None else set(int(value) for value in selected)
    values, starts_all = [], []
    for batch in stream.iterate(start, end, batch_size):
        tensors, starts = to_device(batch, device)
        embedding = model(*tensors).float().cpu().numpy()
        if selected_set is not None:
            mask = np.asarray([int(value) in selected_set for value in starts])
            if not np.any(mask):
                continue
            embedding = embedding[mask]
            starts = starts[mask]
        values.append(embedding.astype(np.float32))
        starts_all.append(starts)
    return np.concatenate(values), np.concatenate(starts_all)


def validation_probe(encoder, stream, od, args, device):
    train_selected = np.arange(
        args.input_hours, TRAIN_END, args.probe_stride, dtype=np.int32
    )
    val_selected = np.arange(TRAIN_END, VAL_END, args.probe_stride, dtype=np.int32)
    train_selected = valid_target_array(train_selected, args.input_hours)
    val_selected = valid_target_array(val_selected, args.input_hours)
    embeddings, starts = make_embeddings(
        encoder,
        stream,
        args.input_hours,
        VAL_END,
        args.window_batch_size,
        device,
        np.concatenate([train_selected, val_selected]),
    )
    split = len(train_selected)
    labels = np.stack(
        [np.asarray(od[t], dtype=np.float32).sum(axis=1) for t in starts]
    )
    probe = nn.Sequential(
        nn.Linear(embeddings.shape[-1], embeddings.shape[-1] // 2),
        nn.LeakyReLU(),
        nn.Linear(embeddings.shape[-1] // 2, 1),
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    x_train = torch.from_numpy(embeddings[:split]).to(device)
    y_train = torch.from_numpy(labels[:split]).to(device)
    for _ in range(args.probe_epochs):
        order = torch.randperm(split, device=device)
        for begin in range(0, split, 16):
            index = order[begin : begin + 16]
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(x_train[index]).squeeze(-1)
            loss = F.mse_loss(prediction, y_train[index])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        prediction = probe(torch.from_numpy(embeddings[split:]).to(device)).squeeze(-1).cpu().numpy()
    target = labels[split:]
    x = prediction.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    pcc = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(x - y)))
    return {"outflow_mae": mae, "outflow_pcc": pcc}


def pretrain(args, stream, od, device):
    online = WindowBootCF(
        N_NODES,
        args.memory_dim,
        args.message_dim,
        3600.0,
        window_seconds=args.input_hours * 3600.0,
    ).to(device)
    target = make_target_encoder(online).to(device)
    predictor = MLP_Predictor(
        args.memory_dim * 2, args.memory_dim * 2, args.memory_dim * 2
    ).to(device)
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=args.pretrain_lr
    )
    checkpoint = args.output_dir / "best_encoder.pt"
    history = []
    best_pcc = -math.inf
    started = time.time()
    for epoch in range(1, args.pretrain_epochs + 1):
        online.train()
        predictor.train()
        losses = []
        epoch_start = time.time()
        for batch_index, batch in enumerate(
            stream.iterate(
                args.input_hours,
                TRAIN_END,
                args.window_batch_size,
                args.max_pretrain_batches,
            )
        ):
            tensors, _ = to_device(batch, device)
            forward, reverse, origin_last, destination_last = tensors
            online_embedding = online(forward, reverse, origin_last, destination_last)
            prediction = predictor(online_embedding)
            noise = torch.clamp(
                1.0 + args.augmentation_noise * torch.randn_like(forward), 0.5, 1.5
            )
            with torch.no_grad():
                target_embedding = target(
                    forward * noise,
                    reverse * noise.transpose(1, 2),
                    origin_last,
                    destination_last,
                )
            loss = 1.0 - F.cosine_similarity(
                prediction, target_embedding.detach(), dim=-1
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(online.parameters()) + list(predictor.parameters()), 5.0
            )
            optimizer.step()
            losses.append(float(loss.item()))
            progress = (
                (epoch - 1)
                + (batch_index + 1)
                / max(
                    1,
                    (TRAIN_END - args.input_hours) // args.window_batch_size,
                )
            ) / args.pretrain_epochs
            ema = 1.0 - (1.0 - args.ema_beta) * (math.cos(math.pi * progress) + 1.0) / 2.0
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.data.mul_(ema).add_(source.data, alpha=1.0 - ema)
            if batch_index == 0 or (batch_index + 1) % 200 == 0:
                print(
                    f"BootCF pretrain epoch={epoch} batch={batch_index+1} loss={loss.item():.6f}",
                    flush=True,
                )
        probe = validation_probe(online, stream, od, args, device)
        row = {
            "epoch": epoch,
            "ssl_loss": float(np.mean(losses)),
            **probe,
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        (args.output_dir / "pretrain_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if probe["outflow_pcc"] > best_pcc:
            best_pcc = probe["outflow_pcc"]
            torch.save(
                {
                    "model": online.state_dict(),
                    "epoch": epoch,
                    "probe": probe,
                },
                checkpoint,
            )
    best = torch.load(checkpoint, map_location=device)
    online.load_state_dict(best["model"])
    return online, {
        "best_epoch": int(best["epoch"]),
        "best_probe": best["probe"],
        "training_seconds": time.time() - started,
    }


def save_all_embeddings(model, stream, args, device):
    path = args.output_dir / "bootcf_embeddings_float16.npy"
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(N_HOURS, N_NODES, args.memory_dim * 2),
    )
    output[:] = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(
            stream.iterate(args.input_hours, N_HOURS, args.window_batch_size)
        ):
            tensors, starts = to_device(batch, device)
            output[starts] = model(*tensors).float().cpu().numpy().astype(np.float16)
            if batch_index == 0 or (batch_index + 1) % 250 == 0:
                print(f"BootCF embedding batches={batch_index+1}", flush=True)
    output.flush()
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def downstream_positive_mae(model, embeddings, od, start, end, args, device):
    model.eval()
    total, count = 0.0, 0
    selected = valid_target_array(np.arange(start, end, dtype=np.int32), args.input_hours)
    for begin in range(0, len(selected), args.downstream_batch_size):
        starts = selected[begin : begin + args.downstream_batch_size]
        x = torch.from_numpy(np.asarray(embeddings[starts], dtype=np.float32)).to(device)
        target = torch.from_numpy(np.asarray(od[starts], dtype=np.float32)).to(device)
        for destination_start in range(0, N_NODES, args.destination_chunk):
            destination_end = min(destination_start + args.destination_chunk, N_NODES)
            prediction = torch.expm1(
                torch.clamp(model.chunk(x, destination_start, destination_end), min=0.0)
            )
            y = target[:, :, destination_start:destination_end]
            mask = y > 0
            total += torch.abs(prediction[mask] - y[mask]).sum().item()
            count += int(mask.sum().item())
    return total / count


def train_downstream(embeddings, od, args, device):
    model = PairMagnitudeMLP(args.memory_dim * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.downstream_lr)
    checkpoint = args.output_dir / "best_od_magnitude.pt"
    history = []
    best = math.inf
    stale = 0
    start_epoch = 1
    history_path = args.output_dir / "downstream_history.json"
    if args.resume_downstream and checkpoint.exists() and history_path.exists():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best = float(payload["validation"])
        start_epoch = max(int(row["epoch"]) for row in history) + 1
        best_epoch = int(payload["epoch"])
        stale = sum(
            1 for row in reversed(history) if int(row["epoch"]) > best_epoch
        )
        print(
            json.dumps(
                {
                    "resume_downstream_epoch": start_epoch,
                    "best_epoch": best_epoch,
                    "best_validation": best,
                    "optimizer_reinitialized": True,
                }
            ),
            flush=True,
        )
    for epoch in range(start_epoch, args.downstream_epochs + 1):
        model.train()
        order = valid_target_array(
            np.arange(args.input_hours, TRAIN_END, dtype=np.int32),
            args.input_hours,
        )
        np.random.default_rng(args.seed + epoch).shuffle(order)
        losses = []
        for batch_index in range(0, len(order), args.downstream_batch_size):
            if args.max_downstream_batches is not None and batch_index // args.downstream_batch_size >= args.max_downstream_batches:
                break
            starts = order[batch_index : batch_index + args.downstream_batch_size]
            x = torch.from_numpy(np.asarray(embeddings[starts], dtype=np.float32)).to(device)
            target = torch.from_numpy(np.asarray(od[starts], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = None
            positive_count = 0
            for destination_start in range(0, N_NODES, args.destination_chunk):
                destination_end = min(destination_start + args.destination_chunk, N_NODES)
                prediction_log = model.chunk(x, destination_start, destination_end)
                y = target[:, :, destination_start:destination_end]
                mask = y > 0
                if torch.any(mask):
                    chunk_sum = torch.abs(prediction_log[mask] - torch.log1p(y[mask])).sum()
                    loss_sum = chunk_sum if loss_sum is None else loss_sum + chunk_sum
                    positive_count += int(mask.sum().item())
            loss = loss_sum / positive_count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = downstream_positive_mae(
            model, embeddings, od, TRAIN_END, VAL_END, args, device
        )
        row = {
            "epoch": epoch,
            "train_positive_log_mae": float(np.mean(losses)),
            "validation_positive_mae": validation,
        }
        history.append(row)
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation < best:
            best = validation
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation": validation},
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.downstream_patience:
                break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    return model, payload


def main() -> None:
    args = parse_args()
    if args.input_hours not in {1, 3, 6, 12}:
        raise ValueError("--input-hours must be one of 1, 3, 6, 12")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    od = np.load(args.od_matrix, mmap_mode="r")
    stream = ContinuousWindowStream(args.cache_dir, args.input_hours)
    started = time.time()
    if args.skip_pretrain:
        encoder = WindowBootCF(
            N_NODES,
            args.memory_dim,
            args.message_dim,
            3600.0,
            window_seconds=args.input_hours * 3600.0,
        ).to(device)
        payload = torch.load(args.output_dir / "best_encoder.pt", map_location=device)
        encoder.load_state_dict(payload["model"])
        pretrain_summary = {"reused": True, "best_epoch": int(payload["epoch"]), "best_probe": payload["probe"]}
    else:
        encoder, pretrain_summary = pretrain(args, stream, od, device)
    embedding_path = args.output_dir / "bootcf_embeddings_float16.npy"
    if args.resume_downstream and embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r")
        print("reused existing BootCF annual embeddings", flush=True)
    else:
        embeddings = save_all_embeddings(encoder, stream, args, device)
    downstream, downstream_payload = train_downstream(embeddings, od, args, device)

    @torch.no_grad()
    def predict_magnitude(starts: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(embeddings[starts], dtype=np.float32)).to(device)
        chunks = []
        for destination_start in range(0, N_NODES, args.destination_chunk):
            destination_end = min(destination_start + args.destination_chunk, N_NODES)
            prediction = torch.expm1(
                torch.clamp(
                    downstream.chunk(x, destination_start, destination_end), min=0.0
                )
            )
            chunks.append(prediction.cpu().numpy())
        return np.concatenate(chunks, axis=2).astype(np.float32)

    hurdle = evaluate_hurdle_model(
        "BootCF-Hurdle",
        predict_magnitude,
        args.output_dir,
        device,
        history_hours=args.input_hours,
    )
    summary = {
        "model": "BootCF-Hurdle",
        "source_code": str(PAPER_CODE),
        "task": (
            f"full-year OD{N_NODES}, public-time continuous past{args.input_hours}h "
            "-> next1h full OD"
        ),
        "input_hours": args.input_hours,
        "adaptation": (
            f"strict independent {args.input_hours}h windows; exact exponential "
            "event summaries; bootstrap intensity-noise target view"
        ),
        "pretraining": pretrain_summary,
        "downstream_best_epoch": int(downstream_payload["epoch"]),
        "downstream_validation_positive_mae": float(downstream_payload["validation"]),
        "parameter_count_encoder": int(sum(p.numel() for p in encoder.parameters())),
        "parameter_count_downstream": int(sum(p.numel() for p in downstream.parameters())),
        "training_seconds": time.time() - started,
        "full_run": args.max_pretrain_batches is None and args.max_downstream_batches is None,
        "hurdle": hurdle,
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
