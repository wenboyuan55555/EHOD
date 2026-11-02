"""Build sparse H12 event-interface caches from frozen added baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ehod.data.od import EXPERIMENT_ROOT, N_NODES, TRAIN_END, VAL_END, load_static
from ehod.common.device import require_cuda
from baselines.regular._shared.mipf import reconstruct_od
from baselines.regular._shared.marginal_data import inverse_marginal, load_marginal_cache
from baselines.event._shared.train_with_hcf import INDEX_PATH
from baselines.regular._shared.train_marginal import make_model, predict_marginal, set_seed
from baselines.regular.strep.train import batch_tensors


METHODS = {
    "duet_mipf": ("formal_lr_0.02", 32),
    "patchstg_mipf": ("formal_lr_0.002", 16),
    "stdn_mipf": ("formal_lr_0.001", 16),
    "lightst_mipf": ("formal_lr_0.003", 32),
    "strep_mipf": ("formal", 16),
}
THRESHOLDS = np.arange(0.05, 1.0, 0.05, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    x = prediction - prediction.mean()
    y = target - target.mean()
    denominator = np.sqrt(np.square(x).sum() * np.square(y).sum())
    return {
        "n": int(len(target)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pcc": float((x * y).sum() / denominator) if denominator > 0 else 0.0,
    }


def formal_directory(method: str, suffix: str, seed: int) -> Path:
    return (
        EXPERIMENT_ROOT
        / "runs/seven_baselines"
        / method
        / ("h12_seed%d_%s" % (seed, suffix))
    )


def load_frozen(method: str, run_dir: Path, device: torch.device, seed: int):
    set_seed(seed)
    if method == "strep_mipf":
        from baselines.regular._shared.upstream_models import create_strep

        model, _ = create_strep(12, N_NODES, device)
        payload = torch.load(run_dir / "encoder_best.pt", map_location=device)
        model.load_state_dict(payload["model"])
        with (run_dir / "ridge_models.pkl").open("rb") as stream:
            ridges = pickle.load(stream)
        model.to(device).eval()
        return model, ridges
    model, _ = make_model(method, 12, device)
    model.to(device)
    payload = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, None


@torch.no_grad()
def marginal_batch(method, model, ridges, cache, starts, device):
    if method != "strep_mipf":
        prediction, _ = predict_marginal(
            method,
            model,
            cache["normalized"],
            starts,
            12,
            cache["mean"],
            cache["std"],
            device,
        )
        return prediction
    history, temporal, _ = batch_tensors(cache, starts, 12, device)
    origin, destination = model.representations(history, temporal)
    origin = ridges[0].predict(origin[:, 0].cpu().numpy().reshape(-1, 64))
    destination = ridges[1].predict(
        destination[:, 0].cpu().numpy().reshape(-1, 64)
    )
    normalized = np.stack(
        (
            origin.reshape(len(starts), N_NODES),
            destination.reshape(len(starts), N_NODES),
        ),
        axis=-1,
    ).astype(np.float32)
    return inverse_marginal(normalized, cache["mean"], cache["std"])


@torch.no_grad()
def fill_learned_splits(
    method,
    model,
    ridges,
    cache,
    event_hour,
    event_edge,
    output,
    batch_size,
    device,
    model_seed,
):
    seasonal = np.asarray(load_static()["seasonal_mean"], dtype=np.float32)
    audit = {}
    for split, lower, upper, seed in (
        ("train", 0, TRAIN_END, model_seed),
        ("validation", TRAIN_END, VAL_END, 100000 + model_seed),
    ):
        selected = np.flatnonzero((event_hour >= lower) & (event_hour < upper))
        unique = np.unique(event_hour[selected])
        set_seed(seed)
        for begin in range(0, len(unique), batch_size):
            starts = unique[begin : begin + batch_size]
            marginal = marginal_batch(
                method, model, ridges, cache, starts, device
            )
            prediction = reconstruct_od(seasonal[starts % 168], marginal)
            left = np.searchsorted(event_hour, starts[0], side="left")
            right = np.searchsorted(event_hour, starts[-1], side="right")
            hours = event_hour[left:right]
            keep = np.isin(hours, starts)
            rows = np.arange(left, right, dtype=np.int64)[keep]
            local = np.searchsorted(starts, event_hour[rows])
            output[rows] = prediction.reshape(len(starts), -1)[
                local, event_edge[rows]
            ]
        audit[split] = {
            "event_rows": int(len(selected)),
            "unique_target_hours": int(len(unique)),
            "rng_seed": seed,
        }
    return audit


def fill_test_from_formal_cache(run_dir, event_hour, event_edge, output):
    prediction = np.load(run_dir / "test_prediction_fp16.npy", mmap_mode="r")
    hours = np.asarray(
        np.load(run_dir / "test_target_hours.npy", mmap_mode="r"), dtype=np.int32
    )
    rows = np.flatnonzero(event_hour >= VAL_END)
    local = np.searchsorted(hours, event_hour[rows])
    if np.any(local >= len(hours)) or not np.array_equal(hours[local], event_hour[rows]):
        raise ValueError("Event test hour is absent from the frozen prediction cache")
    output[rows] = np.asarray(
        prediction.reshape(len(hours), -1)[local, event_edge[rows]]
    )
    return {
        "event_rows": int(len(rows)),
        "unique_target_hours": int(len(np.unique(event_hour[rows]))),
        "source_prediction": str((run_dir / "test_prediction_fp16.npy").resolve()),
        "source_hours": str((run_dir / "test_target_hours.npy").resolve()),
    }


def main() -> None:
    args = parse_args()
    suffix, batch_size = METHODS[args.method]
    run_dir = formal_directory(args.method, suffix, args.seed)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result["history_hours"] != 12
        or result["prediction_horizon"] != 1
        or result["seed"] != args.seed
    ):
        raise ValueError("Only a completed H12-to-1 frozen result is admissible")

    cache_dir = EXPERIMENT_ROOT / "cache/event_backbones/h12"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "seven_" + args.method + (
        "" if args.seed == 0 else "_seed%d" % args.seed
    )
    magnitude_path = cache_dir / (cache_key + "_magnitude.npy")
    metadata_path = cache_dir / (cache_key + "_magnitude.json")
    if metadata_path.exists() and magnitude_path.exists() and not args.overwrite:
        print(metadata_path.read_text(encoding="utf-8"))
        return

    device = require_cuda(args.device)
    index = np.load(INDEX_PATH, mmap_mode="r")
    event_hour = np.asarray(index["sample_hour"], dtype=np.int32)
    event_edge = np.asarray(index["sample_edge"], dtype=np.int64)
    target = np.asarray(index["sample_target"], dtype=np.float32)
    if not np.all(event_hour[1:] >= event_hour[:-1]):
        raise ValueError("Event cache requires hour-sorted event rows")
    marginal_cache = load_marginal_cache()
    model, ridges = load_frozen(args.method, run_dir, device, args.seed)
    output = np.lib.format.open_memmap(
        magnitude_path, mode="w+", dtype=np.float32, shape=(len(event_hour),)
    )
    output[:] = np.nan
    generation = fill_learned_splits(
        args.method,
        model,
        ridges,
        marginal_cache,
        event_hour,
        event_edge,
        output,
        batch_size,
        device,
        args.seed,
    )
    generation["test"] = fill_test_from_formal_cache(
        run_dir, event_hour, event_edge, output
    )
    output.flush()
    if not np.isfinite(output).all():
        raise FloatingPointError("Sparse event magnitude cache is incomplete")

    probability_path = cache_dir / "strict_activation_probability.npy"
    probability = np.asarray(np.load(probability_path, mmap_mode="r"), dtype=np.float32)
    validation = (event_hour >= TRAIN_END) & (event_hour < VAL_END)
    scores = []
    for threshold in THRESHOLDS:
        point = np.where(probability[validation] >= threshold, output[validation], 0.0)
        row = {"threshold": float(threshold), **metrics(target[validation], point)}
        scores.append(row)
    selected = min(scores, key=lambda row: (row["mae"], row["rmse"], -row["pcc"]))
    test = event_hour >= VAL_END
    raw_test = metrics(target[test], output[test])
    interface_test = metrics(
        target[test],
        np.where(probability[test] >= selected["threshold"], output[test], 0.0),
    )
    metadata = {
        "protocol_version": "eadt-seven-baselines-v1",
        "city": result["city"],
        "method": args.method,
        "seed": args.seed,
        "cache_key": cache_key,
        "history_hours": 12,
        "prediction_horizon": 1,
        "uses_event_features": False,
        "frozen_source_result": str(result_path.resolve()),
        "frozen_source_result_sha256": sha256(result_path),
        "frozen_test_prediction_sha256": sha256(
            run_dir / "test_prediction_fp16.npy"
        ),
        "event_index": str(Path(INDEX_PATH).resolve()),
        "event_index_sha256": sha256(Path(INDEX_PATH)),
        "samples": int(len(event_hour)),
        "dtype": "float32",
        "magnitude_cache": str(magnitude_path.resolve()),
        "magnitude_cache_sha256": sha256(magnitude_path),
        "generation": generation,
        "interface": {
            "activation_probability": str(probability_path.resolve()),
            "activation_probability_sha256": sha256(probability_path),
            "threshold_selection": "minimum validation event-cell MAE; RMSE/PCC break ties",
            "threshold_grid": [float(value) for value in THRESHOLDS],
            "selected_threshold": selected["threshold"],
            "selected_validation_metrics": selected,
            "raw_frozen_test_event_metrics": raw_test,
            "zero_residual_interface_test_event_metrics": interface_test,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
