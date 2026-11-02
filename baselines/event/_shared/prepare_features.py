from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ehod.data.od import EXPERIMENT_ROOT
from baselines.event._shared.features import EVENT_FEATURE_DIM, N_EVENT_TYPES, N_VENUES, STRUCTURED_DIM, VENUE_ROLE_DIM
from ehod.common.paths import EVENT_INDEX_PATH


SOURCE_INDEX = EVENT_INDEX_PATH
OUTPUT = EXPERIMENT_ROOT / "cache" / "event_structured_features.npy"
METADATA = OUTPUT.with_suffix(".json")


def phase_id(hour: int, start_minute: float, end_minute: float) -> int:
    start = int(np.floor(start_minute / 60.0))
    end = int(np.floor(end_minute / 60.0))
    if start - 3 <= hour <= start - 2:
        return 0
    if hour == start - 1:
        return 1
    if start <= hour < end:
        return 2
    if end <= hour <= end + 1:
        return 3
    return 4


def main() -> None:
    if OUTPUT.exists() and METADATA.exists():
        print(OUTPUT)
        return
    index = np.load(SOURCE_INDEX, mmap_mode="r")
    sample_hour = index["sample_hour"]
    sample_edge = index["sample_edge"]
    pair_ptr = index["pair_ptr"]
    pair_event = index["pair_event"]
    pair_role = index["pair_role"]
    event_type = index["event_type"]
    event_venue = index["event_venue"]
    event_scale = np.log1p(np.maximum(index["event_scale"], 0.0)).astype(np.float32)
    scale_mean = float(event_scale.mean())
    scale_std = max(float(event_scale.std()), 1e-6)
    event_scale = (event_scale - scale_mean) / scale_std
    event_start = index["event_start_min"]
    event_end = index["event_end_min"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.float16,
        shape=(len(sample_hour), EVENT_FEATURE_DIM),
    )
    output[:] = 0
    for sample in range(len(sample_hour)):
        hour = int(sample_hour[sample])
        begin = int(pair_ptr[sample])
        end = int(pair_ptr[sample + 1])
        for pair in range(begin, end):
            event = int(pair_event[pair])
            role = min(int(pair_role[pair]), 1)
            phase = phase_id(hour, event_start[event], event_end[event])
            kind = int(event_type[event])
            scale = float(event_scale[event])
            structured = (role * 5 + phase) * N_EVENT_TYPES + kind
            output[sample, structured] += np.float16(scale)
            venue_role = STRUCTURED_DIM + role * N_VENUES + int(event_venue[event])
            output[sample, venue_role] += np.float16(scale)
            center = hour * 60.0 + 30.0
            time_offset = STRUCTURED_DIM + VENUE_ROLE_DIM
            output[sample, time_offset] += np.float16(
                np.clip((center - event_start[event]) / 360.0, -1.0, 1.0)
            )
            output[sample, time_offset + 1] += np.float16(
                np.clip((center - event_end[event]) / 360.0, -1.0, 1.0)
            )
            output[sample, time_offset + 2] += np.float16(role == 0)
            output[sample, time_offset + 3] += np.float16(role == 1)
    output.flush()
    metadata = {
        "source": str(SOURCE_INDEX),
        "shape": [int(len(sample_hour)), EVENT_FEATURE_DIM],
        "events": int(len(event_type)),
        "venues": int(np.max(event_venue)) + 1,
        "event_types": int(np.max(event_type)) + 1,
        "uses_original_five_phase_window": True,
    }
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
