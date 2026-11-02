from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ehod.hcf.features import ContinuousFeatureStore
from ehod.data.od import EXPERIMENT_ROOT, N_NODES, load_normalized, load_scaler, load_static
from ehod.common.device import require_cuda
from baselines.regular._shared.factory import create_model, static_tensors
from ehod.hcf.model import CONTINUOUS_MODES, FEATURE_ABLATIONS, ContinuousHybridOD, feature_dimension
from ehod.common.paths import EVENT_INDEX_PATH
from ehod.hcf.train import fusion_prediction
from ehod.hcf.training_utils import torch_static


INDEX_PATH = EVENT_INDEX_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-hours", type=int, choices=[1, 3, 6, 12], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--continuous-run-name", default="formal_decision")
    parser.add_argument(
        "--continuous-mode",
        choices=CONTINUOUS_MODES,
        default="exact",
    )
    parser.add_argument(
        "--feature-ablation", choices=FEATURE_ABLATIONS, default="none"
    )
    parser.add_argument("--cache-key", default="continuous_fusion")
    parser.add_argument(
        "--source-experiment-root",
        type=Path,
        default=None,
        help="Optional root containing the trained fusion and STIDGCN runs.",
    )
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
    cache_dir = EXPERIMENT_ROOT / "cache" / "event_backbones" / (
        "h%d" % args.history_hours
    )
    shared_probability = np.load(
        cache_dir / "strict_activation_probability.npy", mmap_mode="r"
    )
    source_root = args.source_experiment_root or EXPERIMENT_ROOT
    result_dir = (
        source_root
        / "runs"
        / "continuous_fusion_od"
        / (
            "h%d_seed%d_%s"
            % (args.history_hours, args.seed, args.continuous_run_name)
        )
    )
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    activation_blend = float(result["evaluation"]["activation_blend"])
    residual_blend = float(result["evaluation"]["continuous_residual_blend"])
    normalized = load_normalized()
    mean, std = load_scaler()
    static_payload = load_static()
    static = torch_static(static_payload, device)
    discrete_static = static_tensors(static_payload, device)
    store = ContinuousFeatureStore()
    backbone = create_model("stidgcn", args.history_hours, device=device).to(device)
    backbone_path = (
        source_root
        / "runs"
        / "stidgcn"
        / ("h%d_hurdle_seed%d_formal" % (args.history_hours, args.seed))
        / "best_model.pt"
    )
    backbone.load_state_dict(torch.load(backbone_path, map_location=device)["model"])
    backbone.eval()
    model = ContinuousHybridOD(
        nodes=N_NODES,
        feature_dim=feature_dimension(args.continuous_mode),
    ).to(device)
    model.load_state_dict(
        torch.load(result_dir / "best_model.pt", map_location=device)["model"]
    )
    model.eval()
    magnitude_output = np.lib.format.open_memmap(
        cache_dir / f"{args.cache_key}_magnitude.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(sample_hour),),
    )
    probability_output = np.lib.format.open_memmap(
        cache_dir / f"{args.cache_key}_probability.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(sample_hour),),
    )
    for begin in range(0, len(hours), args.batch_size):
        selected_hours = hours[begin : begin + args.batch_size].astype(np.int32)
        continuous_probability, base, residual = fusion_prediction(
            model,
            backbone,
            store,
            normalized,
            static,
            discrete_static,
            selected_hours,
            args.history_hours,
            device,
            mean,
            std,
            args.continuous_mode,
            args.feature_ablation,
        )
        magnitude = np.expm1(np.log1p(base) + residual_blend * residual).reshape(
            len(selected_hours), -1
        )
        continuous_probability = continuous_probability.reshape(
            len(selected_hours), -1
        )
        for local in range(len(selected_hours)):
            group = begin + local
            sample_begin = int(starts[group])
            sample_end = sample_begin + int(counts[group])
            edges = sample_edge[sample_begin:sample_end]
            magnitude_output[sample_begin:sample_end] = magnitude[
                local, edges
            ].astype(np.float16)
            probability_output[sample_begin:sample_end] = (
                (1.0 - activation_blend)
                * np.asarray(
                    shared_probability[sample_begin:sample_end], dtype=np.float32
                )
                + activation_blend * continuous_probability[local, edges]
            ).astype(np.float16)
    magnitude_output.flush()
    probability_output.flush()
    metadata = {
        "model": args.cache_key,
        "history_hours": args.history_hours,
        "seed": args.seed,
        "source_run_name": args.continuous_run_name,
        "continuous_mode": args.continuous_mode,
        "feature_ablation": args.feature_ablation,
        "source_experiment_root": str(source_root),
        "activation_blend": activation_blend,
        "residual_blend": residual_blend,
        "samples": int(len(sample_hour)),
        "device": args.device,
    }
    (cache_dir / f"{args.cache_key}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(cache_dir / f"{args.cache_key}_magnitude.npy")


if __name__ == "__main__":
    main()
