"""Repository-relative locations for the public EHOD benchmark."""
from __future__ import annotations
import os
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]  # repository root
BUNDLE_ROOT = EXPERIMENT_ROOT
ACTIVE_CITY = os.environ.get('EHOD_CITY', 'nyc').strip().lower()
if ACTIVE_CITY not in {'nyc', 'chicago'}:
    raise ValueError('EHOD_CITY must be nyc or chicago')

def _configured_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

DATA_ROOT = _configured_path('EHOD_DATA_ROOT', EXPERIMENT_ROOT / 'data')
OUTPUT_ROOT = _configured_path('EHOD_OUTPUT_ROOT', EXPERIMENT_ROOT / 'outputs')
CITY_EXPERIMENT_ROOT = OUTPUT_ROOT / ACTIVE_CITY
DATA_BUNDLE_ROOT = DATA_ROOT / 'nyc'
CHICAGO_DATA_ROOT = DATA_ROOT / 'chicago'
CHICAGO_EVENT_ROOT = CHICAGO_DATA_ROOT / 'events'
NYC_CORE_ROOT = DATA_BUNDLE_ROOT / 'core_od'
NYC_ORDER_ROOT = DATA_BUNDLE_ROOT / 'orders'
NYC_CONTINUOUS_CACHE_ROOT = DATA_BUNDLE_ROOT / 'continuous_cache'
EVENT_DATA_ROOT = DATA_BUNDLE_ROOT / 'events'
PACKAGED_EXPERIMENT_CACHE_ROOT = OUTPUT_ROOT / 'nyc/cache'
BASELINE_ROOT = EXPERIMENT_ROOT / 'baselines'
NYC_ORDER_PATH = NYC_ORDER_ROOT / 'orders.parquet'
NYC_EVENT_INDEX_PATH = EVENT_DATA_ROOT / 'event_edges_h12.npz'
NYC_EVENT_CSV_PATH = EVENT_DATA_ROOT / 'events.csv'
NYC_ZONE_INDEX_PATH = NYC_CORE_ROOT / 'zone_index.csv'
CHICAGO_EXPERIMENT_ROOT = OUTPUT_ROOT / 'chicago'
CHICAGO_EVENT_INDEX_PATH = CHICAGO_EVENT_ROOT / 'event_edges_h12.npz'
CHICAGO_EVENT_CSV_PATH = CHICAGO_EVENT_ROOT / 'events.csv'
CHICAGO_ZONE_INDEX_PATH = CHICAGO_DATA_ROOT / 'core_od/zone_index.csv'
CHICAGO_CONTINUOUS_CACHE_ROOT = CHICAGO_DATA_ROOT / 'continuous_cache'
CONTINUOUS_CACHE_ROOT = DATA_ROOT / ACTIVE_CITY / 'continuous_cache'
EVENT_INDEX_PATH = DATA_ROOT / ACTIVE_CITY / 'events/event_edges_h12.npz'
EVENT_CSV_PATH = DATA_ROOT / ACTIVE_CITY / 'events/events.csv'
ZONE_INDEX_PATH = DATA_ROOT / ACTIVE_CITY / 'core_od/zone_index.csv'

def required_portable_paths() -> tuple[Path, ...]:
    core = DATA_ROOT / ACTIVE_CITY / 'core_od'
    return (core / 'od_matrix.npy', core / 'normalized/od_log1p_zscore.npy',
            core / 'normalized/scaler.npz', ZONE_INDEX_PATH, EVENT_INDEX_PATH,
            EVENT_CSV_PATH, CONTINUOUS_CACHE_ROOT / 'continuous_hour_exp_weights.npy',
            CONTINUOUS_CACHE_ROOT / 'continuous_hour_last_pickup_second.npy',
            CONTINUOUS_CACHE_ROOT / 'continuous_hour_destination_last_pickup_second.npy',
            CONTINUOUS_CACHE_ROOT / 'continuous_hour_quarter_counts.npy',
            CONTINUOUS_CACHE_ROOT / 'continuous_hour_quarter_origin.npy',
            CONTINUOUS_CACHE_ROOT / 'continuous_hour_quarter_destination.npy')

def assert_portable_layout() -> None:
    missing = [str(p) for p in required_portable_paths() if not p.is_file()]
    if missing:
        raise FileNotFoundError('The local EHOD data directory is incomplete; missing:\n' + '\n'.join(missing))
