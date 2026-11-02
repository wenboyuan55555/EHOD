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

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, calendar_features, history_batch, iter_hour_batches, load_od, load_static
from ehod.common.device import require_cuda
from ehod.hcf.activation import StrictActivationGate, predict_gate_full, selected_gate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hour-batch", type=int, default=4)
    parser.add_argument("--edges-per-hour", type=int, default=16384)
    parser.add_argument("--hours-per-epoch", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def torch_static(payload, device):
    return {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for key, value in payload.items()
        if key in ("edge_nonzero_frequency", "edge_positive_mean", "seasonal_probability")
    }


def sampled_loss(
    model,
    od,
    static,
    starts_np,
    history_hours,
    edges_per_hour,
    device,
    generator,
):
    history_np = history_batch(od, starts_np, history_hours)
    history = torch.from_numpy(history_np).to(device)
    starts = torch.from_numpy(starts_np.astype(np.int64)).to(device)
    batch = len(starts_np)
    origin = torch.randint(
        0, N_NODES, (batch, edges_per_hour), device=device, generator=generator
    )
    destination = torch.randint(
        0, N_NODES, (batch, edges_per_hour), device=device, generator=generator
    )
    features = selected_gate_features(history, starts, origin, destination, static)
    calendar = torch.from_numpy(calendar_features(starts_np)).to(device)
    logits = model(features, origin, destination, calendar)
    batch_index = torch.arange(batch, device=device)[:, None]
    target = (history.new_tensor(np.asarray(od[starts_np], dtype=np.float32))[
        batch_index, origin, destination
    ] > 0).float()
    return F.binary_cross_entropy_with_logits(logits, target), logits, target


@torch.no_grad()
def validation_loss(
    model, od, static, history_hours, edges_per_hour, device, seed, smoke
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total = 0.0
    n = 0
    limit = 16 if smoke else 256
    for starts in iter_hour_batches(
        "validation", 4, shuffle=False, limit=limit
    ):
        loss, logits, target = sampled_loss(
            model,
            od,
            static,
            starts,
            history_hours,
            edges_per_hour,
            device,
            generator,
        )
        count = target.numel()
        total += float(loss.item()) * count
        n += count
    return total / max(n, 1)


@torch.no_grad()
def cache_probabilities(model, od, static, history_hours, device, output_dir):
    for split in ("validation", "test"):
        hours = []
        start, end = (6720, 7728) if split == "validation" else (7728, 8760)
        path = output_dir / (split + "_probability.npy")
        cache = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(end - start, N_NODES, N_NODES),
        )
        for begin in range(start, end, 2):
            finish = min(begin + 2, end)
            starts_np = np.arange(begin, finish, dtype=np.int32)
            history = torch.from_numpy(
                history_batch(od, starts_np, history_hours)
            ).to(device)
            starts = torch.from_numpy(starts_np.astype(np.int64)).to(device)
            calendar = torch.from_numpy(calendar_features(starts_np)).to(device)
            probability = predict_gate_full(
                model, history, starts, calendar, static
            )
            cache[begin - start : finish - start] = probability.astype(np.float16)
        cache.flush()
        hours.append(end - start)


def main() -> None:
    args = parse_args()
    device = require_cuda(args.device)
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "activation"
        / ("h%d_seed%d" % (args.history_hours, args.seed))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    od = load_od()
    static = torch_static(load_static(), device)
    model = StrictActivationGate(nodes=N_NODES).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 101)
    history = []
    best = math.inf
    stale = 0
    checkpoint = output_dir / "best_model.pt"
    epochs = 2 if args.smoke else args.epochs
    limit = 16 if args.smoke else args.hours_per_epoch
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        started = time.time()
        for starts_np in iter_hour_batches(
            "train",
            args.hour_batch,
            shuffle=True,
            seed=args.seed + epoch,
            limit=limit,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, logits, target = sampled_loss(
                model,
                od,
                static,
                starts_np,
                args.history_hours,
                args.edges_per_hour if not args.smoke else 1024,
                device,
                generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * target.numel()
            n += target.numel()
        val = validation_loss(
            model,
            od,
            static,
            args.history_hours,
            args.edges_per_hour if not args.smoke else 1024,
            device,
            args.seed + 999,
            args.smoke,
        )
        row = {
            "epoch": epoch,
            "train_bce": total / max(n, 1),
            "validation_bce": val,
            "seconds": time.time() - started,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)
        if val < best - 1.0e-6:
            best = val
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_bce": val,
                    "history_hours": args.history_hours,
                    "seed": args.seed,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    if not args.smoke:
        cache_probabilities(
            model, od, static, args.history_hours, device, output_dir
        )
    summary = {
        "model": "StrictActivationGate",
        "history_hours": args.history_hours,
        "seed": args.seed,
        "best_epoch": payload["epoch"],
        "best_validation_bce": payload["validation_bce"],
        "strict_window": True,
        "test_evaluated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
