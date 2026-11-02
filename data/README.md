# EHOD processed datasets

The [EHOD data v1 release](https://github.com/wenboyuan55555/EHOD/releases/tag/data-v1)
provides the **complete 2023 processed tensors** for NYC and Chicago.
`manifest.json` records every file, byte size, SHA-256 checksum,
and NumPy shape/dtype. Core data and order-state arrays are stored in
separate subdirectories for each city.

## Download and extract

From the repository root, run:

```bash
python scripts/download_data.py --city all
```

This uses only Python's standard library and verifies both archive and extracted
file SHA-256 checksums. Downloaded ZIPs are cached in `.downloads/`; files are
installed under `data/`. Use `--city nyc` or `--city chicago` for a single city,
`--data-root /path/to/data` (or `EHOD_DATA_ROOT`) to change the data directory,
and `--download-dir /path/to/downloads` to change the archive cache.
The full uncompressed dataset contains 28 files totaling 13,362,192,456 bytes.
Allow about 15 GB of disk space for data and downloads together.

For a manual download:

- [NYC ZIP](https://github.com/wenboyuan55555/EHOD/releases/download/data-v1/ehod-data-v1-nyc.zip)
- [Chicago ZIP](https://github.com/wenboyuan55555/EHOD/releases/download/data-v1/ehod-data-v1-chicago.zip)
- [SHA256SUMS](https://github.com/wenboyuan55555/EHOD/releases/download/data-v1/SHA256SUMS)
- [Archive sizes and checksums](releases/data-v1.json)

Check the ZIP checksums using `sha256sum -c SHA256SUMS` on Linux, or
`Get-FileHash .downloads/*.zip -Algorithm SHA256` in PowerShell and compare them
with `SHA256SUMS`. Extract each ZIP **into the repository root**, since archive
paths already start with `data/`. For example:

```bash
python -m zipfile -e .downloads/ehod-data-v1-nyc.zip .
python -m zipfile -e .downloads/ehod-data-v1-chicago.zip .
```

Git ignores the large arrays; the Release is the complete data distribution.
The existing code license does not grant additional rights to source datasets;
the provenance and source terms described below continue to apply.

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
