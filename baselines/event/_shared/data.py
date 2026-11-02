from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_HOURS, N_NODES, TRAIN_END, TRAIN_START, load_od
from ehod.data.events import venue_zone_ids
from ehod.common.paths import EVENT_CSV_PATH


CACHE_DIR = EXPERIMENT_ROOT / "cache" / "original_event_baselines"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _event_prompt(row) -> str:
    scale = row.capacity_proxy
    if pd.isna(scale):
        scale = row.public_scale_value
    pieces = [
        str(row.event_name),
        "event type " + str(row.subcategory),
        "at " + str(row.venue),
        "scheduled from " + str(row.start_datetime_local),
        "to " + str(row.end_datetime_local),
        "public scale " + ("unknown" if pd.isna(scale) else str(float(scale))),
    ]
    for field in ("competition_or_tour", "organizer_or_home_team", "opponent_or_support"):
        value = getattr(row, field, None)
        if value is not None and not pd.isna(value):
            pieces.append(str(value))
    return ". ".join(pieces)


def _text_embeddings(events: pd.DataFrame, overwrite: bool = False) -> np.ndarray:
    path = CACHE_DIR / "event_text_embeddings.npy"
    if path.exists() and not overwrite:
        return np.load(path)
    packaged = EVENT_CSV_PATH.parent / "text_embeddings.npy"
    if packaged.exists() and not overwrite:
        values = np.load(packaged)
        if values.shape != (len(events), 384):
            raise ValueError("Packaged event text embeddings have an unexpected shape")
        np.save(path, values)
        return values
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(TEXT_MODEL)
    prompts = [_event_prompt(row) for row in events.itertuples(index=False)]
    values = encoder.encode(
        prompts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(path, values)
    return values


def _build_marginals() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    od = load_od()
    marginal = np.empty((N_HOURS, N_NODES, 2), dtype=np.float32)
    for begin in range(0, N_HOURS, 168):
        end = min(begin + 168, N_HOURS)
        block = np.asarray(od[begin:end], dtype=np.float32)
        marginal[begin:end, :, 0] = block.sum(axis=2)
        marginal[begin:end, :, 1] = block.sum(axis=1)
    transformed = np.log1p(marginal)
    train = transformed[TRAIN_START:TRAIN_END]
    mean = train.mean(axis=(0, 1)).astype(np.float32)
    std = train.std(axis=(0, 1)).clip(min=1.0e-6).astype(np.float32)
    normalized = ((transformed - mean) / std).astype(np.float32)

    seasonal_raw = np.zeros((168, N_NODES, 2), dtype=np.float64)
    seasonal_count = np.zeros(168, dtype=np.int32)
    for hour in range(TRAIN_START, TRAIN_END):
        slot = hour % 168
        seasonal_raw[slot] += marginal[hour]
        seasonal_count[slot] += 1
    seasonal_raw /= np.maximum(seasonal_count[:, None, None], 1)
    seasonal_scaled = (
        (np.log1p(seasonal_raw).astype(np.float32) - mean) / std
    ).astype(np.float32)
    return marginal, normalized, mean, std, seasonal_scaled


def _build_event_arrays(
    events: pd.DataFrame, embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = venue_zone_ids(events)
    origin = pd.Timestamp("2023-01-01 00:00:00")
    starts = pd.to_datetime(events["start_datetime_local"], format="mixed")
    ends = pd.to_datetime(events["end_datetime_local"], format="mixed")
    start_hour = np.floor((starts - origin) / pd.Timedelta(hours=1)).astype(int)
    end_hour = np.floor((ends - origin) / pd.Timedelta(hours=1)).astype(int)
    scale = events["capacity_proxy"].fillna(events["public_scale_value"]).fillna(0.0)
    scale = np.log1p(np.maximum(scale.to_numpy(np.float32), 0.0)) / np.log1p(100000.0)
    by_hour: Dict[int, List[Tuple[int, int, int, float, float, float]]] = defaultdict(list)
    for event, row in enumerate(events.itertuples(index=False)):
        venue = int(mapping[str(row.venue)])
        if ACTIVE_CITY == "nyc":
            venue -= 1
        begin = int(start_hour.iloc[event])
        finish = int(end_hour.iloc[event])
        for hour in range(begin - 3, finish + 4):
            if not 0 <= hour < N_HOURS:
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
            by_hour[hour].append(
                (event, venue, phase, float(scale[event]), hour - begin + 0.5, hour - finish + 0.5)
            )
    max_tokens = max(max((len(value) for value in by_hour.values()), default=1), 1)
    event_node = np.zeros((N_HOURS, N_NODES, 6), dtype=np.float32)
    event_tokens = np.zeros((N_HOURS, max_tokens, embeddings.shape[1]), dtype=np.float16)
    event_mask = np.zeros((N_HOURS, max_tokens), dtype=bool)
    event_global = np.zeros((N_HOURS, embeddings.shape[1]), dtype=np.float32)
    arrival_weight = (0.35, 1.0, 0.15, 0.0, 0.0)
    active_weight = (0.10, 0.25, 1.0, 0.50, 0.20)
    departure_weight = (0.0, 0.0, 0.10, 1.0, 0.40)
    for hour, values in by_hour.items():
        for token, (event, venue, phase, value, tau_start, tau_end) in enumerate(values):
            event_tokens[hour, token] = embeddings[event].astype(np.float16)
            event_mask[hour, token] = True
            features = event_node[hour, venue]
            features[0] += value * arrival_weight[phase]
            features[1] += value * active_weight[phase]
            features[2] += value * departure_weight[phase]
            features[3] += value
            features[4] += np.tanh(tau_start / 6.0) * value
            features[5] += np.tanh(tau_end / 6.0) * value
        event_global[hour] = embeddings[[value[0] for value in values]].max(axis=0)
    # MultiheadAttention returns NaN when every token is padding.  A zero dummy
    # token is therefore explicitly marked valid for no-event hours.
    event_mask[~event_mask.any(axis=1), 0] = True
    return event_node, event_tokens, event_mask, event_global


def _retrieved_marginals(
    normalized: np.ndarray, event_global: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    import faiss

    hour = np.arange(N_HOURS)
    calendar = np.stack(
        (
            np.sin(2 * np.pi * (hour % 24) / 24),
            np.cos(2 * np.pi * (hour % 24) / 24),
            np.sin(2 * np.pi * ((hour // 24) % 7) / 7),
            np.cos(2 * np.pi * ((hour // 24) % 7) / 7),
        ),
        axis=1,
    ).astype(np.float32)
    semantic = event_global.astype(np.float32)
    norm = np.linalg.norm(semantic, axis=1, keepdims=True)
    semantic = semantic / np.maximum(norm, 1.0e-6)
    descriptor = np.ascontiguousarray(np.concatenate((semantic, 2.0 * calendar), axis=1))
    faiss.normalize_L2(descriptor)
    train_hours = np.arange(TRAIN_START, TRAIN_END, dtype=np.int32)
    index = faiss.IndexFlatIP(descriptor.shape[1])
    index.add(descriptor[train_hours])
    _, neighbors = index.search(descriptor, 12)
    selected = np.empty(N_HOURS, dtype=np.int32)
    for query in range(N_HOURS):
        candidates = train_hours[neighbors[query]]
        different = candidates[candidates != query]
        selected[query] = int(different[0] if len(different) else candidates[0])
    return normalized[selected].astype(np.float32), selected


def build_original_event_cache(overwrite: bool = False) -> Path:
    metadata_path = CACHE_DIR / "metadata.json"
    required = (
        "marginal_raw.npy",
        "marginal_normalized.npy",
        "marginal_scaler.npz",
        "seasonal_marginal_normalized.npy",
        "event_node.npy",
        "event_tokens.npy",
        "event_token_mask.npy",
        "event_global.npy",
        "retrieved_marginal.npy",
        "retrieved_hour.npy",
    )
    if not overwrite and metadata_path.exists() and all((CACHE_DIR / x).exists() for x in required):
        return CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENT_CSV_PATH)
    embeddings = _text_embeddings(events, overwrite=overwrite)
    marginal, normalized, mean, std, seasonal = _build_marginals()
    event_node, event_tokens, event_mask, event_global = _build_event_arrays(events, embeddings)
    retrieved, retrieved_hour = _retrieved_marginals(normalized, event_global)
    np.save(CACHE_DIR / "marginal_raw.npy", marginal)
    np.save(CACHE_DIR / "marginal_normalized.npy", normalized)
    np.savez(CACHE_DIR / "marginal_scaler.npz", mean=mean, std=std)
    np.save(CACHE_DIR / "seasonal_marginal_normalized.npy", seasonal)
    np.save(CACHE_DIR / "event_node.npy", event_node.astype(np.float16))
    np.save(CACHE_DIR / "event_tokens.npy", event_tokens)
    np.save(CACHE_DIR / "event_token_mask.npy", event_mask)
    np.save(CACHE_DIR / "event_global.npy", event_global.astype(np.float16))
    np.save(CACHE_DIR / "retrieved_marginal.npy", retrieved)
    np.save(CACHE_DIR / "retrieved_hour.npy", retrieved_hour)
    metadata = {
        "city": "chicago" if N_NODES == 77 else "nyc",
        "events": int(len(events)),
        "text_encoder": TEXT_MODEL,
        "text_embedding_dimension": int(embeddings.shape[1]),
        "event_information": "forecast-safe schedule and pre-event attributes only",
        "retrieval_database": [TRAIN_START, TRAIN_END],
        "retrieval_excludes_same_hour": True,
        "marginal_channels": ["origin_outflow", "destination_inflow"],
        "uses_test_targets_for_training_or_retrieval": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CACHE_DIR


def load_original_event_cache() -> Dict[str, np.ndarray]:
    root = build_original_event_cache()
    scaler = np.load(root / "marginal_scaler.npz")
    return {
        "marginal_raw": np.load(root / "marginal_raw.npy", mmap_mode="r"),
        "marginal_normalized": np.load(root / "marginal_normalized.npy", mmap_mode="r"),
        "mean": np.asarray(scaler["mean"], dtype=np.float32),
        "std": np.asarray(scaler["std"], dtype=np.float32),
        "seasonal_marginal": np.load(root / "seasonal_marginal_normalized.npy", mmap_mode="r"),
        "event_node": np.load(root / "event_node.npy", mmap_mode="r"),
        "event_tokens": np.load(root / "event_tokens.npy", mmap_mode="r"),
        "event_token_mask": np.load(root / "event_token_mask.npy", mmap_mode="r"),
        "event_global": np.load(root / "event_global.npy", mmap_mode="r"),
        "retrieved": np.load(root / "retrieved_marginal.npy", mmap_mode="r"),
    }


__all__ = ["CACHE_DIR", "TEXT_MODEL", "build_original_event_cache", "load_original_event_cache"]
