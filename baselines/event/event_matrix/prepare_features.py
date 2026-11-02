from __future__ import annotations

import json

import numpy as np

from ehod.data.od import EXPERIMENT_ROOT
from ehod.data.events import ScheduledEventStore
from ehod.common.paths import EVENT_INDEX_PATH


def main() -> None:
    index = np.load(EVENT_INDEX_PATH, mmap_mode="r")
    sample_hour = np.asarray(index["sample_hour"], dtype=np.int32)
    sample_edge = np.asarray(index["sample_edge"], dtype=np.int32)
    output_path = EXPERIMENT_ROOT / "cache/event_matrix_features.npy"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        print(output_path)
        return
    hours, starts, counts = np.unique(
        sample_hour, return_index=True, return_counts=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(sample_hour), 3),
    )
    store = ScheduledEventStore()
    for group, hour in enumerate(hours):
        begin = int(starts[group])
        end = begin + int(counts[group])
        matrix = store.batch_matrix(np.asarray([hour], dtype=np.int32))[0].reshape(-1, 3)
        output[begin:end] = matrix[sample_edge[begin:end]].astype(np.float16)
    output.flush()
    metadata_path.write_text(
        json.dumps(
            {
                "source": str(EVENT_INDEX_PATH),
                "samples": int(len(sample_hour)),
                "channels": ["arrival", "departure", "active"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
