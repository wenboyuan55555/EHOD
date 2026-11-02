from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ehod.data.od import ACTIVE_CITY, N_HOURS, N_NODES, load_od
from ehod.common.paths import CHICAGO_DATA_ROOT, CONTINUOUS_CACHE_ROOT, NYC_ORDER_PATH


MICROSECONDS_PER_HOUR = 3_600_000_000
MICROSECONDS_PER_QUARTER = 900_000_000
EPOCH = np.datetime64("2023-01-01T00:00:00", "us").astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def grouped_counts(keys: np.ndarray):
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    return sorted_keys[starts], np.diff(np.r_[starts, len(sorted_keys)]).astype(np.uint32)


def checked_add(target: np.ndarray, keys: np.ndarray, counts: np.ndarray, limit: int):
    current = target[keys].astype(np.uint64)
    updated = current + counts.astype(np.uint64)
    if int(updated.max(initial=0)) > limit:
        raise OverflowError(f"Quarter-hour count exceeds storage limit {limit}")
    target[keys] = updated.astype(target.dtype)


def main() -> None:
    args = parse_args()
    cache = CONTINUOUS_CACHE_ROOT
    cache.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "edge": cache / "continuous_hour_quarter_counts.npy",
        "origin": cache / "continuous_hour_quarter_origin.npy",
        "destination": cache / "continuous_hour_quarter_destination.npy",
    }
    metadata_path = cache / "continuous_hour_quarter_metadata.json"
    if all(path.exists() for path in final_paths.values()) and metadata_path.exists():
        if not args.overwrite:
            print("Quarter-hour cache already exists; use --overwrite to rebuild.")
            return

    temporary_paths = {
        name: path.with_name(path.name + ".building.npy")
        for name, path in final_paths.items()
    }
    existing = [
        path
        for path in (*final_paths.values(), *temporary_paths.values(), metadata_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Partial quarter-hour cache exists; inspect it or rerun with --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    orders = (
        CHICAGO_DATA_ROOT / "orders/chicago_taxi_2023_orders_area77.parquet"
        if ACTIVE_CITY == "chicago"
        else NYC_ORDER_PATH
    )
    origin_column = (
        "pickup_community_area" if ACTIVE_CITY == "chicago" else "pickup_zone_id"
    )
    destination_column = (
        "dropoff_community_area" if ACTIVE_CITY == "chicago" else "dropoff_zone_id"
    )
    shape = (N_HOURS, N_NODES, N_NODES, 4)
    edge = np.lib.format.open_memmap(
        temporary_paths["edge"], mode="w+", dtype=np.uint16, shape=shape
    )
    origin = np.lib.format.open_memmap(
        temporary_paths["origin"],
        mode="w+",
        dtype=np.uint32,
        shape=(N_HOURS, N_NODES, 4),
    )
    destination = np.lib.format.open_memmap(
        temporary_paths["destination"],
        mode="w+",
        dtype=np.uint32,
        shape=(N_HOURS, N_NODES, 4),
    )
    edge[:] = 0
    origin[:] = 0
    destination[:] = 0
    edge_flat = edge.reshape(-1)
    origin_flat = origin.reshape(-1)
    destination_flat = destination.reshape(-1)

    parquet = pq.ParquetFile(orders)
    seen = 0
    started = time.time()
    columns = ["pickup_datetime", origin_column, destination_column]
    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=args.batch_size, columns=columns, use_threads=True
        ),
        start=1,
    ):
        timestamp_us = (
            batch.column(0)
            .to_numpy(zero_copy_only=False)
            .astype("datetime64[us]")
            .astype(np.int64)
        )
        relative_us = timestamp_us - EPOCH
        hour = relative_us // MICROSECONDS_PER_HOUR
        quarter = (relative_us % MICROSECONDS_PER_HOUR) // MICROSECONDS_PER_QUARTER
        source = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64) - 1
        target = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64) - 1
        valid = (
            (hour >= 0)
            & (hour < N_HOURS)
            & (quarter >= 0)
            & (quarter < 4)
            & (source >= 0)
            & (source < N_NODES)
            & (target >= 0)
            & (target < N_NODES)
        )
        if not np.all(valid):
            raise ValueError(
                f"Found {np.count_nonzero(~valid)} invalid orders in batch {batch_number}"
            )

        edge_key = (((hour * N_NODES + source) * N_NODES + target) * 4 + quarter)
        keys, counts = grouped_counts(edge_key)
        checked_add(edge_flat, keys, counts, np.iinfo(np.uint16).max)

        origin_key = ((hour * N_NODES + source) * 4 + quarter)
        keys, counts = grouped_counts(origin_key)
        checked_add(origin_flat, keys, counts, np.iinfo(np.uint32).max)

        destination_key = ((hour * N_NODES + target) * 4 + quarter)
        keys, counts = grouped_counts(destination_key)
        checked_add(destination_flat, keys, counts, np.iinfo(np.uint32).max)

        seen += len(hour)
        if batch_number == 1 or batch_number % 25 == 0:
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "city": ACTIVE_CITY,
                        "batch": batch_number,
                        "rows": seen,
                        "total_rows": parquet.metadata.num_rows,
                        "rows_per_second": seen / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )

    for array in (edge, origin, destination):
        array.flush()

    od = load_od()
    mismatch_cells = 0
    max_abs_difference = 0
    for begin in range(0, N_HOURS, 24):
        end = min(begin + 24, N_HOURS)
        reconstructed = np.asarray(edge[begin:end], dtype=np.uint32).sum(axis=3)
        difference = reconstructed.astype(np.int64) - np.asarray(
            od[begin:end], dtype=np.int64
        )
        mismatch_cells += int(np.count_nonzero(difference))
        max_abs_difference = max(
            max_abs_difference, int(np.max(np.abs(difference), initial=0))
        )
        if not np.array_equal(
            np.asarray(origin[begin:end], dtype=np.uint64),
            np.asarray(edge[begin:end], dtype=np.uint64).sum(axis=2),
        ):
            raise RuntimeError("Origin quarter totals do not match edge cache")
        if not np.array_equal(
            np.asarray(destination[begin:end], dtype=np.uint64),
            np.asarray(edge[begin:end], dtype=np.uint64).sum(axis=1),
        ):
            raise RuntimeError("Destination quarter totals do not match edge cache")
    if mismatch_cells:
        raise RuntimeError(
            f"Quarter cache does not reproduce OD matrix: mismatches={mismatch_cells}, "
            f"max_abs_difference={max_abs_difference}"
        )

    maximum_edge_count = int(np.asarray(edge).max())
    for array in (edge, origin, destination):
        del array
    for name, final_path in final_paths.items():
        os.replace(str(temporary_paths[name]), str(final_path))
    metadata = {
        "city": ACTIVE_CITY,
        "orders": str(orders.resolve()),
        "orders_rows": seen,
        "time_origin": "2023-01-01 00:00:00 local wall-clock time",
        "quarter_order": ["00-15", "15-30", "30-45", "45-60"],
        "edge_shape": list(shape),
        "edge_dtype": "uint16",
        "node_dtype": "uint32",
        "maximum_edge_quarter_count": maximum_edge_count,
        "count_reconstruction_mismatch_cells": mismatch_cells,
        "count_reconstruction_max_abs_difference": max_abs_difference,
        "published_time_resolution": (
            "15-minute privacy-rounded"
            if ACTIVE_CITY == "chicago"
            else "original microsecond timestamp field"
        ),
        "elapsed_seconds": time.time() - started,
        "files": {name: str(path.resolve()) for name, path in final_paths.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

