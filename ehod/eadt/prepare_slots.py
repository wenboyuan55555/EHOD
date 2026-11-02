from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ehod.data.od import EXPERIMENT_ROOT
from ehod.common.paths import EVENT_INDEX_PATH, ZONE_INDEX_PATH


OUTPUT_DIR = EXPERIMENT_ROOT / "cache" / "event_transition_slots"


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat1 = np.radians(latitude)[:, None]
    lat2 = np.radians(latitude)[None, :]
    delta_lat = lat2 - lat1
    delta_lon = np.radians(longitude)[None, :] - np.radians(longitude)[:, None]
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return (
        6371.0088
        * 2.0
        * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
    ).astype(np.float32)


def create_memmap(name: str, dtype, samples: int, slots: int, fill_value) -> np.memmap:
    value = np.lib.format.open_memmap(
        OUTPUT_DIR / (name + ".npy"),
        mode="w+",
        dtype=dtype,
        shape=(samples, slots),
    )
    value[:] = fill_value
    return value


def main() -> None:
    metadata_path = OUTPUT_DIR / "metadata.json"
    required = (
        "event_type.npy",
        "venue.npy",
        "role.npy",
        "scale.npy",
        "tau_start.npy",
        "tau_end.npy",
        "arrival_distance_km.npy",
        "departure_distance_km.npy",
    )
    if metadata_path.exists() and all((OUTPUT_DIR / name).exists() for name in required):
        print(OUTPUT_DIR)
        return

    index = np.load(EVENT_INDEX_PATH, mmap_mode="r")
    sample_hour = np.asarray(index["sample_hour"], dtype=np.int32)
    sample_edge = np.asarray(index["sample_edge"], dtype=np.int32)
    pair_ptr = np.asarray(index["pair_ptr"], dtype=np.int64)
    counts = np.diff(pair_ptr)
    max_events_per_sample = int(counts.max())

    zones = pd.read_csv(ZONE_INDEX_PATH).sort_values("matrix_index")
    distance_km = haversine_matrix(
        zones["centroid_lat"].to_numpy(np.float64),
        zones["centroid_lon"].to_numpy(np.float64),
    )
    origin = sample_edge // len(zones)
    destination = sample_edge % len(zones)
    event_venue_node = np.asarray(index["event_venue_node"], dtype=np.int64)
    event_scale = np.log1p(
        np.maximum(np.asarray(index["event_scale"], dtype=np.float32), 0.0)
    ) / math.log1p(100_000.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_type = create_memmap("event_type", np.int8, len(sample_hour), max_events_per_sample, -1)
    venue = create_memmap("venue", np.int8, len(sample_hour), max_events_per_sample, -1)
    role = create_memmap("role", np.int8, len(sample_hour), max_events_per_sample, -1)
    scale = create_memmap("scale", np.float16, len(sample_hour), max_events_per_sample, 0.0)
    tau_start = create_memmap("tau_start", np.float16, len(sample_hour), max_events_per_sample, 0.0)
    tau_end = create_memmap("tau_end", np.float16, len(sample_hour), max_events_per_sample, 0.0)
    arrival_distance = create_memmap(
        "arrival_distance_km", np.float16, len(sample_hour), max_events_per_sample, 0.0
    )
    departure_distance = create_memmap(
        "departure_distance_km", np.float16, len(sample_hour), max_events_per_sample, 0.0
    )

    pair_event = np.asarray(index["pair_event"], dtype=np.int32)
    pair_role = np.asarray(index["pair_role"], dtype=np.int8)
    source_type = np.asarray(index["event_type"], dtype=np.int8)
    source_venue = np.asarray(index["event_venue"], dtype=np.int8)
    source_start = np.asarray(index["event_start_min"], dtype=np.float32)
    source_end = np.asarray(index["event_end_min"], dtype=np.float32)

    for sample in range(len(sample_hour)):
        center_minute = float(sample_hour[sample]) * 60.0 + 30.0
        begin = int(pair_ptr[sample])
        end = int(pair_ptr[sample + 1])
        for slot, pair in enumerate(range(begin, end)):
            event = int(pair_event[pair])
            venue_node = int(event_venue_node[event])
            event_type[sample, slot] = source_type[event]
            venue[sample, slot] = source_venue[event]
            role[sample, slot] = pair_role[pair]
            scale[sample, slot] = np.float16(event_scale[event])
            tau_start[sample, slot] = np.float16(
                (center_minute - float(source_start[event])) / 60.0
            )
            tau_end[sample, slot] = np.float16(
                (center_minute - float(source_end[event])) / 60.0
            )
            arrival_distance[sample, slot] = np.float16(
                distance_km[int(destination[sample]), venue_node]
            )
            departure_distance[sample, slot] = np.float16(
                distance_km[int(origin[sample]), venue_node]
            )
        if sample and sample % 500_000 == 0:
            print(json.dumps({"samples_done": sample}), flush=True)

    for value in (
        event_type,
        venue,
        role,
        scale,
        tau_start,
        tau_end,
        arrival_distance,
        departure_distance,
    ):
        value.flush()

    metadata = {
        "source": str(EVENT_INDEX_PATH),
        "samples": int(len(sample_hour)),
        "max_events_per_sample": max_events_per_sample,
        "observed_max_events_per_sample": max_events_per_sample,
        "time_reference": "forecast-hour center relative to published event start/end",
        "scale": "log1p(capacity_proxy) / log1p(100000)",
        "distance": "haversine distance between Taxi Zone centroids",
        "hard_phase_bins": False,
        "forecast_safe": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
