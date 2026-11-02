from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, calendar_features, history_batch, load_od, load_static
from ehod.common.device import require_cuda
from ehod.hcf.activation import StrictActivationGate, predict_gate_full
from ehod.common.paths import EVENT_INDEX_PATH
from ehod.hcf.train_activation import torch_static


INDEX_PATH = EVENT_INDEX_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = require_cuda(args.device)
    index = np.load(INDEX_PATH, mmap_mode="r")
    sample_hour = index["sample_hour"]
    sample_edge = index["sample_edge"]
    output_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "strict_activation_probability.npy"
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(sample_hour),)
    )
    gate_dir = EXPERIMENT_ROOT / "runs" / "activation" / (
        "h%d_seed0" % args.history_hours
    )
    payload = torch.load(gate_dir / "best_model.pt", map_location=device)
    model = StrictActivationGate(nodes=N_NODES).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    static = torch_static(load_static(), device)
    od = load_od()
    hours, starts, counts = np.unique(
        sample_hour, return_index=True, return_counts=True
    )
    for begin in range(0, len(hours), 2):
        selected_hours = hours[begin : begin + 2].astype(np.int32)
        history = torch.from_numpy(
            history_batch(od, selected_hours, args.history_hours)
        ).to(device)
        starts_tensor = torch.from_numpy(selected_hours.astype(np.int64)).to(device)
        calendar = torch.from_numpy(calendar_features(selected_hours)).to(device)
        probability = predict_gate_full(
            model, history, starts_tensor, calendar, static
        ).reshape(len(selected_hours), -1)
        for local in range(len(selected_hours)):
            group = begin + local
            sample_begin = int(starts[group])
            sample_end = sample_begin + int(counts[group])
            output[sample_begin:sample_end] = probability[
                local, sample_edge[sample_begin:sample_end]
            ].astype(np.float16)
    output.flush()
    metadata = {
        "history_hours": args.history_hours,
        "samples": int(len(sample_hour)),
        "unique_event_hours": int(len(hours)),
        "strict_window": True,
        "source_checkpoint": str(gate_dir / "best_model.pt"),
        "device": args.device,
    }
    (output_dir / "strict_activation_probability.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
