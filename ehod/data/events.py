from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import numpy as np
import pandas as pd

from ehod.data.od import ACTIVE_CITY, N_HOURS, N_NODES
from ehod.common.paths import EVENT_CSV_PATH, ZONE_INDEX_PATH


EVENT_CSV = EVENT_CSV_PATH

NYC_VENUE_ZONE_IDS = {
    "Madison Square Garden": 186,
    "Yankee Stadium": 247,
    "Citi Field": 253,
    "Barclays Center": 189,
    "Radio City Music Hall": 161,
    "Flushing Meadows–Corona Park": 93,
    "Forest Hills Stadium": 95,
    "Central Park Great Lawn": 43,
}


def venue_zone_ids(events: pd.DataFrame) -> dict:
    if ACTIVE_CITY == "nyc":
        return NYC_VENUE_ZONE_IDS
    zones = pd.read_csv(ZONE_INDEX_PATH)
    area_to_node = dict(
        zip(zones["community_name"].astype(str), zones["matrix_index"].astype(int))
    )
    unknown = sorted(set(events["borough"].astype(str)) - set(area_to_node))
    if unknown:
        raise ValueError("Unmapped Chicago Community Areas: %s" % unknown)
    return {
        str(row.venue): int(area_to_node[str(row.borough)])
        for row in events[["venue", "borough"]].drop_duplicates().itertuples(index=False)
    }


class ScheduledEventStore:
    """Forecast-safe 2023 schedule with the established five-phase window."""

    def __init__(self, csv_path: Path = EVENT_CSV) -> None:
        events = pd.read_csv(csv_path)
        mapping = venue_zone_ids(events)
        unknown = sorted(set(events["venue"].astype(str)) - set(mapping))
        if unknown:
            raise ValueError("Unmapped venues: %s" % unknown)
        start = pd.to_datetime(events["start_datetime_local"], format="mixed")
        end = pd.to_datetime(events["end_datetime_local"], format="mixed")
        origin = pd.Timestamp("2023-01-01 00:00:00")
        start_hour = np.floor((start - origin) / pd.Timedelta(hours=1)).astype(int)
        end_hour = np.floor((end - origin) / pd.Timedelta(hours=1)).astype(int)
        scale = events["capacity_proxy"].fillna(
            events["public_scale_value"]
        ).fillna(0.0)
        log_scale = np.log1p(np.maximum(scale.to_numpy(np.float32), 0.0))
        log_scale = (log_scale - log_scale.mean()) / max(float(log_scale.std()), 1e-6)
        normalized_scale = np.exp(0.25 * log_scale).astype(np.float32)
        self.by_hour: DefaultDict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        for row, begin, finish, value in zip(
            events.itertuples(index=False), start_hour, end_hour, normalized_scale
        ):
            venue = int(mapping[str(row.venue)])
            if ACTIVE_CITY == "nyc":
                venue -= 1
            for hour in range(int(begin) - 3, int(finish) + 4):
                if hour < 0 or hour >= N_HOURS:
                    continue
                if begin - 3 <= hour <= begin - 2:
                    phase = 0
                elif hour == begin - 1:
                    phase = 1
                elif begin <= hour < finish:
                    phase = 2
                elif finish <= hour <= finish + 1:
                    phase = 3
                else:
                    phase = 4
                self.by_hour[hour].append((venue, phase, float(value)))
        self.event_count = int(len(events))
        self.venue_count = int(events["venue"].nunique())

    def batch_matrix(self, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int32)
        output = np.zeros((len(starts), N_NODES, N_NODES, 3), dtype=np.float32)
        arrival_weight = (0.35, 1.0, 0.15, 0.0, 0.0)
        departure_weight = (0.0, 0.0, 0.10, 1.0, 0.40)
        active_weight = (0.10, 0.25, 1.0, 0.50, 0.20)
        for batch, hour in enumerate(starts):
            for venue, phase, scale in self.by_hour.get(int(hour), []):
                output[batch, :, venue, 0] += scale * arrival_weight[phase]
                output[batch, venue, :, 1] += scale * departure_weight[phase]
                output[batch, :, venue, 2] += scale * active_weight[phase]
                output[batch, venue, :, 2] += scale * active_weight[phase]
        return output

    def affected_edge_mask(self, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int32)
        mask = np.zeros((len(starts), N_NODES, N_NODES), dtype=bool)
        for batch, hour in enumerate(starts):
            for venue, _, _ in self.by_hour.get(int(hour), []):
                mask[batch, :, venue] = True
                mask[batch, venue, :] = True
        return mask


VENUE_ZONE_IDS = NYC_VENUE_ZONE_IDS

__all__ = ["ScheduledEventStore", "VENUE_ZONE_IDS", "venue_zone_ids"]
