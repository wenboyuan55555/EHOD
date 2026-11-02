from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ehod.common.paths import EVENT_INDEX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-instance cluster bootstrap for paired OD predictions."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-key", default="event_prediction")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--reference-key", default="base_prediction")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pcc(sx: float, sx2: float, sxy: float, sy: float, sy2: float, n: float) -> float:
    numerator = sxy - sx * sy / n
    denominator = math.sqrt(
        max(sx2 - sx * sx / n, 0.0) * max(sy2 - sy * sy / n, 0.0)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def metrics(values: np.ndarray) -> dict:
    n, absolute, squared, sx, sx2, sxy, sy, sy2 = values
    return {
        "n": int(n),
        "mae": float(absolute / n),
        "rmse": float(math.sqrt(max(squared / n, 0.0))),
        "pcc": float(pcc(sx, sx2, sxy, sy, sy2, n)),
    }


def summarize(samples: np.ndarray, favorable: str) -> dict:
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "favorable_probability": float(
            np.mean(samples < 0.0) if favorable == "lower" else np.mean(samples > 0.0)
        ),
    }


def main() -> None:
    args = parse_args()
    candidate_payload = np.load(args.candidate)
    reference_path = args.reference or args.candidate
    reference_payload = np.load(reference_path)
    sample_index = np.asarray(candidate_payload["sample_index"], dtype=np.int64)
    reference_index = np.asarray(reference_payload["sample_index"], dtype=np.int64)
    if not np.array_equal(sample_index, reference_index):
        raise ValueError("Candidate and reference sample_index arrays differ")
    target = np.asarray(candidate_payload["target"], dtype=np.float64)
    reference_target = np.asarray(reference_payload["target"], dtype=np.float64)
    if not np.array_equal(target, reference_target):
        raise ValueError("Candidate and reference target arrays differ")
    candidate = np.asarray(candidate_payload[args.candidate_key], dtype=np.float64)
    reference = np.asarray(reference_payload[args.reference_key], dtype=np.float64)

    index = np.load(EVENT_INDEX_PATH, mmap_mode="r")
    pair_ptr = np.asarray(index["pair_ptr"], dtype=np.int64)
    pair_event = np.asarray(index["pair_event"], dtype=np.int32)
    primary_event = pair_event[pair_ptr[sample_index]]
    pair_counts = pair_ptr[sample_index + 1] - pair_ptr[sample_index]
    event_ids, inverse = np.unique(primary_event, return_inverse=True)
    clusters = np.zeros((len(event_ids), 15), dtype=np.float64)
    for cluster in range(len(event_ids)):
        selected = inverse == cluster
        y = target[selected]
        a = candidate[selected]
        b = reference[selected]
        clusters[cluster] = (
            selected.sum(),
            np.abs(a - y).sum(),
            np.square(a - y).sum(),
            a.sum(),
            np.square(a).sum(),
            (a * y).sum(),
            y.sum(),
            np.square(y).sum(),
            np.abs(b - y).sum(),
            np.square(b - y).sum(),
            b.sum(),
            np.square(b).sum(),
            (b * y).sum(),
            float(np.mean(np.abs(a - y)) < np.mean(np.abs(b - y))),
            float(np.sqrt(np.mean(np.square(a - y))) < np.sqrt(np.mean(np.square(b - y)))),
        )

    aggregate = clusters.sum(axis=0)
    candidate_stats = aggregate[:8]
    reference_stats = np.asarray(
        [
            aggregate[0],
            aggregate[8],
            aggregate[9],
            aggregate[10],
            aggregate[11],
            aggregate[12],
            aggregate[6],
            aggregate[7],
        ]
    )
    candidate_metrics = metrics(candidate_stats)
    reference_metrics = metrics(reference_stats)

    rng = np.random.default_rng(args.seed)
    differences = np.empty((args.iterations, 3), dtype=np.float64)
    for iteration in range(args.iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        total = clusters[selected].sum(axis=0)
        a_stats = total[:8]
        b_stats = np.asarray(
            [
                total[0],
                total[8],
                total[9],
                total[10],
                total[11],
                total[12],
                total[6],
                total[7],
            ]
        )
        a_metrics = metrics(a_stats)
        b_metrics = metrics(b_stats)
        differences[iteration] = (
            a_metrics["mae"] - b_metrics["mae"],
            a_metrics["rmse"] - b_metrics["rmse"],
            a_metrics["pcc"] - b_metrics["pcc"],
        )

    result = {
        "candidate": str(args.candidate),
        "candidate_key": args.candidate_key,
        "reference": str(reference_path),
        "reference_key": args.reference_key,
        "cluster_unit": "primary event instance",
        "event_clusters": int(len(event_ids)),
        "samples": int(len(sample_index)),
        "multi_event_samples": int(np.sum(pair_counts > 1)),
        "bootstrap_iterations": int(args.iterations),
        "bootstrap_seed": int(args.seed),
        "candidate_metrics": candidate_metrics,
        "reference_metrics": reference_metrics,
        "candidate_minus_reference": {
            "mae": candidate_metrics["mae"] - reference_metrics["mae"],
            "rmse": candidate_metrics["rmse"] - reference_metrics["rmse"],
            "pcc": candidate_metrics["pcc"] - reference_metrics["pcc"],
        },
        "bootstrap": {
            "mae_difference": summarize(differences[:, 0], "lower"),
            "rmse_difference": summarize(differences[:, 1], "lower"),
            "pcc_difference": summarize(differences[:, 2], "higher"),
        },
        "event_win_rate": {
            "mae": float(clusters[:, 13].mean()),
            "rmse": float(clusters[:, 14].mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
