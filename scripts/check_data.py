#!/usr/bin/env python3
"""Check local EHOD tensors, sample alignment, splits, and optional hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(city, full):
    protocol = json.loads((ROOT / "configs/benchmark.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    dataset = Path(os.environ.get("EHOD_DATA_ROOT", str(ROOT / "data")))
    info = protocol["cities"][city]
    paths = [f for f in manifest["files"] if f["path"].startswith("data/" + city + "/")]
    for item in paths:
        path = dataset / Path(item["path"]).relative_to("data")
        assert path.is_file(), "Missing %s" % path
        assert path.stat().st_size == item["bytes"], "File size mismatch: %s" % path
        if "shape" in item:
            a = np.load(path, mmap_mode="r")
            assert list(a.shape) == item["shape"] and str(a.dtype) == item["dtype"], path
        if full:
            assert sha256(path) == item["sha256"], "SHA-256 mismatch: %s" % path
    base = dataset / city
    n = info["zones"]
    od = np.load(base / "core_od/od_matrix.npy", mmap_mode="r")
    assert od.shape == (8760, n, n)
    normalized = np.load(base / "core_od/normalized/od_log1p_zscore.npy", mmap_mode="r")
    scaler = np.load(base / "core_od/normalized/scaler.npz")
    mean, std = float(scaler["mean"]), float(scaler["std"])
    assert np.isfinite(mean) and std > 0
    zones = pd.read_csv(base / "core_od/zone_index.csv").sort_values("matrix_index")
    assert len(zones) == n and np.array_equal(zones.matrix_index, np.arange(n))
    events = pd.read_csv(base / "events/events.csv")
    assert len(events) == info["scheduled_events"]
    start = pd.to_datetime(events.start_datetime_local, format="mixed")
    end = pd.to_datetime(events.end_datetime_local, format="mixed")
    assert start.notna().all() and end.notna().all() and (end >= start).all()
    index = np.load(base / "events/event_edges_h12.npz")
    hours = index["sample_hour"].astype(np.int64)
    edges = index["sample_edge"].astype(np.int64)
    targets = index["sample_target"]
    assert len(hours) == len(edges) == len(targets)
    assert hours.min() >= 12 and hours.max() < 8760
    assert edges.min() >= 0 and edges.max() < n * n
    order = hours * n * n + edges
    assert (np.diff(order) > 0).all(), "Event cells must be sorted and unique"
    assert not np.isin(hours, info["invalid_target_hours"]).any()
    counts = {}
    for split, (a, b) in protocol["target_hour_splits"].items():
        counts[split] = int(((hours >= a) & (hours < b)).sum())
    assert counts == info["event_cells"], (city, counts, info["event_cells"])
    for a in range(0, len(hours), 100000):
        b = a + 100000
        actual = od[hours[a:b], edges[a:b] // n, edges[a:b] % n]
        np.testing.assert_array_equal(actual, targets[a:b], err_msg="Event targets disagree with OD counts")
    ptr = index["pair_ptr"]
    assert len(ptr) == len(hours) + 1 and ptr[0] == 0
    assert ptr[-1] == len(index["pair_event"]) and (np.diff(ptr) >= 1).all()
    quarter = np.load(base / "continuous_cache/continuous_hour_quarter_counts.npy", mmap_mode="r")
    origin = np.load(base / "continuous_cache/continuous_hour_quarter_origin.npy", mmap_mode="r")
    destination = np.load(base / "continuous_cache/continuous_hour_quarter_destination.npy", mmap_mode="r")
    selected = [12, 100, 6719, 6720, 7728, 8500]
    for hour in selected:
        q = np.asarray(quarter[hour], dtype=np.uint64)
        np.testing.assert_array_equal(q.sum(-1), od[hour])
        np.testing.assert_array_equal(q.sum(1), origin[hour])
        np.testing.assert_array_equal(q.sum(0), destination[hour])
        expected = (np.log1p(np.asarray(od[hour], dtype=np.float32)) - mean) / std
        np.testing.assert_allclose(expected, normalized[hour], rtol=2e-6, atol=2e-6)
    if full:
        for a in range(0, len(od), 168):
            x = od[a:a + 168]
            assert np.isfinite(x).all() and (x >= 0).all()
    print(json.dumps({"city": city, "status": "passed", "files": len(paths),
                      "od_shape": list(od.shape), "events": len(events),
                      "event_cells": counts, "sha256_verified": full}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=("nyc", "chicago", "all"), default="all")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    for city in (["nyc", "chicago"] if args.city == "all" else [args.city]):
        check(city, args.full)
