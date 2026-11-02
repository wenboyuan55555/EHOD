# EHOD processed datasets

This directory provides the **complete 2023 processed tensors** for NYC and
Chicago. `manifest.json` records every file, byte size, SHA-256 checksum,
and NumPy shape/dtype. Core data and order-state arrays are stored in
separate subdirectories for each city.

## Files

| Relative to `data/<city>/` | Meaning |
| --- | --- |
| `core_od/od_matrix.npy` | Hour × origin × destination trip counts |
| `core_od/zone_index.csv` | Matrix index, public zone ID/name, centroid coordinates |
| `core_od/time_index.csv` | Local hourly time index |
| `core_od/normalized/od_log1p_zscore.npy` | Log1p counts, standardized with training-only scalars |
| `core_od/normalized/scaler.npz` | Exact frozen `mean` and `std` |
| `events/events.csv` | Forecast-safe public event schedule and source provenance |
| `events/event_edges_h12.npz` | Event-cell index, targets, event metadata, and exposure links |
| `events/text_embeddings.npy` | Frozen normalized 384-dimensional all-MiniLM-L6-v2 event text embeddings for baseline adaptations |
| `continuous_cache/continuous_hour_exp_weights.npy` | Hourly exponentially weighted trip sums, for CMOD/BootCF |
| `continuous_cache/continuous_hour_last_pickup_second.npy` | Last pickup time for each origin and hour |
| `continuous_cache/continuous_hour_destination_last_pickup_second.npy` | Last pickup time grouped by destination and hour |
| `continuous_cache/continuous_hour_quarter_counts.npy` | OD counts in the four quarter-hours of each hour |
| `continuous_cache/continuous_hour_quarter_origin.npy` | Quarter-hour origin marginal counts |
| `continuous_cache/continuous_hour_quarter_destination.npy` | Quarter-hour destination marginal counts |

HCF's formal `hierarchical` mode uses the subhour count state and the previous
observed hours. Exact-time summaries are included for the continuous
baselines. The quarter-hour axis is the final axis of each quarter-count
array. These are the existing experiment inputs copied without changing
their numerical representation.

## Time and split conventions

- Hour 0 is local `2023-01-01 00:00`; indices run through 8759.
- An example targeting hour `t` reads only `[t-H, t)` demand/state arrays.
- All histories share target-hour train `[12,6720)`, validation
  `[6720,7728)`, and test `[7728,8760)`.
- The normalization scalars were fitted to hours `[0,6720)`; supervised
  targets start at 12. Static priors use only training data.
- Chicago target hours `1682,8757,8758,8759` are excluded. The released
  arrays retain all 8,760 positions; do not remove rows and renumber time.
- Original local-time and missing-hour handling are retained. The index is
  not a UTC timeline.

## Events

NYC contains 470 scheduled events; Chicago contains 413. The test split
contains 42 and 32 events, respectively. The event index is the union of
venue-related directional OD exposures, extending from three hours before
the scheduled start through three hours after the expected end. Overlapping
exposures share one target cell and retain multiple event links.

The supplied arrays retain the precise sample ordering and float16 feature
quantization used by the experiment code. Schedule inputs use known starts,
expected ends, public scale proxies, event types, and venues. Realized
attendance or realized end times are not forecast inputs. The event index
contains target demand for supervision/evaluation; it must not be supplied
as an input feature.

## Verify local data

```bash
python scripts/check_data.py --city all --full
```

`--full` additionally hashes all extracted files and scans all OD counts for
invalid values. Without it, checks cover shapes, sample alignment, splits,
schedule size, and subhour count consistency at selected hours.

The release distributes processed data, not raw taxi records. Public raw
sources are linked in the root README; event source references are retained
in the schedule CSV files. No new rights to those source datasets are
granted by the code license.
