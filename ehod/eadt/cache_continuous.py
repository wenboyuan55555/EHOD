from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, N_NODES
from ehod.common.device import require_cuda
from ehod.common.paths import CONTINUOUS_CACHE_ROOT, EVENT_INDEX_PATH

INDEX_PATH = EVENT_INDEX_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cmod", "bootcf"], required=True)
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-batch-size", type=int, default=65536)
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Output prefix; defaults to the model name for seed-0 compatibility.",
    )
    return parser.parse_args()


def cmod_cache(args, index, output, device):
    from baselines.regular.cmod.model import ContinuousWindowCMOD
    from baselines.regular.cmod.window_data import LAMBDA_SECONDS, WindowStream

    hour_weights = np.load(
        CONTINUOUS_CACHE_ROOT / "continuous_hour_exp_weights.npy",
        mmap_mode="r",
    )
    hour_last = np.load(
        CONTINUOUS_CACHE_ROOT / "continuous_hour_last_pickup_second.npy",
        mmap_mode="r",
    )
    stream = WindowStream(hour_weights, hour_last, args.history_hours)
    model = ContinuousWindowCMOD(
        device=args.device,
        n_nodes=N_NODES,
        message_dimension=128,
        memory_dimension=128,
        lambda_seconds=LAMBDA_SECONDS,
        window_seconds=args.history_hours * 3600.0,
    ).to(device)
    payload = torch.load(args.source_dir / "best_model.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    sample_hour = index["sample_hour"]
    sample_edge = index["sample_edge"]
    hours, starts, counts = np.unique(
        sample_hour, return_index=True, return_counts=True
    )
    validation_test_cache = args.source_dir / "magnitude_predictions.npy"
    cached = (
        np.load(validation_test_cache, mmap_mode="r")
        if validation_test_cache.exists()
        else None
    )
    with torch.no_grad():
        for group, hour in enumerate(hours):
            begin = int(starts[group])
            end = begin + int(counts[group])
            edges = np.asarray(sample_edge[begin:end], dtype=np.int64)
            if cached is not None and int(hour) >= 6720:
                matrix = np.asarray(cached[int(hour) - 6720], dtype=np.float32).reshape(-1)
                output[begin:end] = matrix[edges].astype(np.float16)
                continue
            _, weights, last_times = next(
                stream.iterate(int(hour), int(hour) + 1)
            )
            embedding = model.encode_window(
                torch.from_numpy(weights).to(device),
                torch.from_numpy(last_times).to(device),
            )
            for pair_begin in range(0, len(edges), args.pair_batch_size):
                selected = edges[pair_begin : pair_begin + args.pair_batch_size]
                origin = torch.from_numpy(selected // N_NODES).long().to(device)
                destination = torch.from_numpy(selected % N_NODES).long().to(device)
                pair = torch.cat(
                    [embedding[origin], embedding[destination]], dim=-1
                )
                magnitude = torch.clamp(model.predict_od.w(pair).squeeze(-1), min=0.0)
                output[
                    begin + pair_begin : begin + pair_begin + len(selected)
                ] = magnitude.cpu().numpy().astype(np.float16)
            if group == 0 or (group + 1) % 250 == 0:
                print(
                    json.dumps(
                        {"model": "cmod", "event_hours_completed": group + 1}
                    ),
                    flush=True,
                )


def bootcf_cache(args, index, output, device):
    from baselines.regular.bootcf.model import PairMagnitudeMLP

    embeddings = np.load(
        args.source_dir / "bootcf_embeddings_float16.npy", mmap_mode="r"
    )
    model = PairMagnitudeMLP(embeddings.shape[-1]).to(device)
    payload = torch.load(args.source_dir / "best_od_magnitude.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    sample_hour = np.asarray(index["sample_hour"], dtype=np.int64)
    sample_edge = np.asarray(index["sample_edge"], dtype=np.int64)
    with torch.no_grad():
        for begin in range(0, len(sample_hour), args.pair_batch_size):
            end = min(begin + args.pair_batch_size, len(sample_hour))
            hour = sample_hour[begin:end]
            edge = sample_edge[begin:end]
            origin = edge // N_NODES
            destination = edge % N_NODES
            source_embedding = torch.from_numpy(
                np.asarray(embeddings[hour, origin], dtype=np.float32)
            ).to(device)
            destination_embedding = torch.from_numpy(
                np.asarray(embeddings[hour, destination], dtype=np.float32)
            ).to(device)
            pair = torch.cat([source_embedding, destination_embedding], dim=-1)
            magnitude = torch.expm1(
                torch.clamp(model.w(pair).squeeze(-1), min=0.0)
            )
            output[begin:end] = magnitude.cpu().numpy().astype(np.float16)


def main():
    args = parse_args()
    device = require_cuda(args.device)
    index = np.load(INDEX_PATH, mmap_mode="r")
    output_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = args.cache_key or args.model
    output_path = output_dir / (cache_key + "_magnitude.npy")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(index["sample_hour"]),),
    )
    started = time.time()
    if args.model == "cmod":
        cmod_cache(args, index, output, device)
    else:
        bootcf_cache(args, index, output, device)
    output.flush()
    metadata = {
        "model": args.model,
        "cache_key": cache_key,
        "history_hours": args.history_hours,
        "source_dir": str(args.source_dir.resolve()),
        "samples": int(len(index["sample_hour"])),
        "device": args.device,
        "seconds": time.time() - started,
    }
    (output_dir / (cache_key + "_magnitude.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
