from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, VAL_END, iter_hour_batches, load_normalized, load_od, load_scaler, load_static
from ehod.common.device import require_cuda
from ehod.data.events import ScheduledEventStore
from ehod.evaluation.metrics import StreamingMetrics
from baselines.regular._shared.factory import create_model, static_tensors
from baselines.regular._shared.train_discrete import predict_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("odpf",), default="odpf")
    parser.add_argument("--history-hours", type=int, choices=(1, 3, 6, 12), default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-run-name", default=None)
    parser.add_argument("--output-run-name", default=None)
    args = parser.parse_args()
    device = require_cuda(args.device)
    source_run_name = args.source_run_name or (
        "h%d_hurdle_seed%d_formal" % (args.history_hours, args.seed)
    )
    run_dir = EXPERIMENT_ROOT / "runs" / args.model / source_run_name
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = create_model(args.model, args.history_hours, device=device).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalized = load_normalized()
    od = load_od()
    mean, std = load_scaler()
    static = static_tensors(load_static(), device)
    event_store = ScheduledEventStore()
    probability = np.load(
        EXPERIMENT_ROOT
        / "runs"
        / "activation"
        / ("h%d_seed0" % args.history_hours)
        / "test_probability.npy",
        mmap_mode="r",
    )
    threshold = float(result["evaluation"]["threshold"])
    metrics = StreamingMetrics()
    for starts in iter_hour_batches("test", args.batch_size):
        magnitude = predict_raw(
            args.model,
            model,
            normalized,
            starts,
            args.history_hours,
            static,
            device,
            mean,
            std,
            event_store,
        )
        gate = np.asarray(probability[starts - VAL_END], dtype=np.float32)
        prediction = np.where(gate >= threshold, magnitude, 0.0).astype(np.float32)
        target = np.asarray(od[starts], dtype=np.float32)
        mask = event_store.affected_edge_mask(starts)
        if mask.any():
            metrics.update(torch.from_numpy(prediction[mask]), torch.from_numpy(target[mask]))
    output = {
        "city": ACTIVE_CITY,
        "backbone": "odpf_native",
        "history_hours": args.history_hours,
        "method": "native_event_matrix",
        "seed": args.seed,
        "device": str(device),
        "test_direct_samples": metrics.n,
        "event": metrics.compute(),
        "global_event": result["evaluation"]["test"],
        "planned_events": result.get("planned_events"),
        "comparison_role": "OD-PF paper-faithful taxi adaptation with its native three-channel planned-event matrix",
        "source_paper": {
            "title": "Real-time Prediction of Transit Origin-Destination Flows During Underground Incidents",
            "doi": "10.1016/j.trc.2024.104622",
        },
        "activation_gate_seed": 0,
        "source_backbone_result": str(result_path),
    }
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "event_adapters"
        / ("h%d" % args.history_hours)
        / "odpf_native"
        / (args.output_run_name or ("native_event_matrix_seed%d" % args.seed))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(output["event"], indent=2))


if __name__ == "__main__":
    main()
