from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, iter_hour_batches, load_normalized, load_od, load_scaler, load_static
from ehod.common.device import require_cuda
from baselines.regular._shared.factory import create_model, static_tensors
from ehod.common.paths import EVENT_INDEX_PATH
from baselines.regular._shared.train_discrete import predict_raw


INDEX_PATH = EVENT_INDEX_PATH
DISCRETE_MODELS = (
    "stidgcn",
    "stid",
    "stpro",
    "odmixer",
    "stan",
    "odpf",
    "odpf_no_event",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=DISCRETE_MODELS, required=True)
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--variant", default="hurdle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Output prefix; defaults to the model name for seed-0 compatibility.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory when training and event roots differ.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = require_cuda(args.device)
    index = np.load(INDEX_PATH, mmap_mode="r")
    sample_hour = index["sample_hour"]
    sample_edge = index["sample_edge"]
    hours, starts, counts = np.unique(
        sample_hour, return_index=True, return_counts=True
    )
    architecture_model = "odpf" if args.model == "odpf_no_event" else args.model
    run_dir = args.source_dir or (
        EXPERIMENT_ROOT
        / "runs"
        / args.model
        / (
            "h%d_%s_seed%d_formal"
            % (args.history_hours, args.variant, args.seed)
        )
    )
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = create_model(architecture_model, args.history_hours, device=device).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalized = load_normalized()
    od = load_od()
    mean, std = load_scaler()
    static = static_tensors(load_static(), device)
    event_store = None
    if args.model == "odpf":
        from ehod.data.events import ScheduledEventStore

        event_store = ScheduledEventStore()
    output_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = args.cache_key or args.model
    output_path = output_dir / (cache_key + "_magnitude.npy")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(sample_hour),)
    )
    for begin in range(0, len(hours), args.batch_size):
        selected_hours = hours[begin : begin + args.batch_size].astype(np.int32)
        prediction = predict_raw(
            architecture_model,
            model,
            normalized,
            selected_hours,
            args.history_hours,
            static,
            device,
            mean,
            std,
            event_store,
        ).reshape(len(selected_hours), -1)
        for local in range(len(selected_hours)):
            group = begin + local
            sample_begin = int(starts[group])
            sample_end = sample_begin + int(counts[group])
            output[sample_begin:sample_end] = prediction[
                local, sample_edge[sample_begin:sample_end]
            ].astype(np.float16)
    output.flush()
    metadata = {
        "model": args.model,
        "cache_key": cache_key,
        "architecture_model": architecture_model,
        "history_hours": args.history_hours,
        "variant": args.variant,
        "seed": args.seed,
        "samples": int(len(sample_hour)),
        "unique_event_hours": int(len(hours)),
        "checkpoint": str(checkpoint),
        "device": args.device,
    }
    (output_dir / (cache_key + "_magnitude.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
