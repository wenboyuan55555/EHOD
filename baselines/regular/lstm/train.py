#!/usr/bin/env python3
"""LSTM training and evaluation for the shared hourly OD task."""

from __future__ import annotations

import copy
import gc
import json
import time
from typing import Dict, List, Tuple

import numpy as np


from baselines.regular._shared.simple_data import CityConfig, write_json, evaluate_hurdle, cache_event_magnitude, base_payload

def run_lstm(
    config: CityConfig,
    od: np.ndarray,
    normalized: np.ndarray,
    mean: float,
    std: float,
    train_hours: np.ndarray,
    validation_hours: np.ndarray,
    test_hours: np.ndarray,
    history_hours: int,
    hidden: int,
    epochs: int,
    patience: int,
    seed: int,
) -> Dict[str, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device(config.device)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device.index:
        raise RuntimeError("%s is unavailable" % config.device)
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    edge_count = config.nodes * config.nodes

    class FlattenedODLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(edge_count, hidden, batch_first=True)
            self.output = nn.Linear(hidden, edge_count)

        def forward(self, history: torch.Tensor) -> torch.Tensor:
            encoded, _ = self.lstm(history)
            return self.output(encoded[:, -1])

    model = FlattenedODLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    micro_batch = config.lstm_micro_batch
    effective_batch = 32
    accumulation_steps = max(effective_batch // micro_batch, 1)
    best_state = None
    best_epoch = -1
    best_validation = float("inf")
    stale = 0
    history_rows: List[Dict[str, float]] = []

    def history_tensor(hours: np.ndarray) -> torch.Tensor:
        index = hours[:, None] + np.arange(-history_hours, 0, dtype=np.int32)[None, :]
        values = np.asarray(normalized[index], dtype=np.float32).reshape(
            len(hours), history_hours, edge_count
        )
        return torch.from_numpy(values).to(device)

    def target_tensors(hours: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = np.asarray(normalized[hours], dtype=np.float32).reshape(
            len(hours), edge_count
        )
        raw = np.asarray(od[hours], dtype=np.float32).reshape(len(hours), edge_count)
        return torch.from_numpy(scaled).to(device), torch.from_numpy(raw).to(device)

    def raw_prediction(scaled: torch.Tensor) -> torch.Tensor:
        return torch.expm1(scaled.float() * std + mean).clamp_min(0.0)

    @torch.no_grad()
    def positive_validation_mae() -> float:
        model.eval()
        absolute = 0.0
        count = 0
        for begin in range(0, len(validation_hours), micro_batch):
            part = validation_hours[begin : begin + micro_batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = raw_prediction(model(history_tensor(part)))
            target = torch.from_numpy(
                np.asarray(od[part], dtype=np.float32).reshape(len(part), edge_count)
            ).to(device)
            positive = target > 0
            absolute += float((prediction[positive] - target[positive]).abs().sum())
            count += int(positive.sum())
        return absolute / max(count, 1)

    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        shuffled = train_hours.copy()
        rng.shuffle(shuffled)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        steps = 0
        for begin in range(0, len(shuffled), micro_batch):
            part = shuffled[begin : begin + micro_batch]
            target_scaled, target_raw = target_tensors(part)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(history_tensor(part))
                positive = target_raw > 0
                loss = F.smooth_l1_loss(prediction[positive], target_scaled[positive])
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            steps += 1
            if steps % accumulation_steps == 0 or begin + micro_batch >= len(shuffled):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * int(positive.sum())
            seen += int(positive.sum())
        validation_mae = positive_validation_mae()
        row = {
            "epoch": epoch,
            "train_positive_smooth_l1": total / max(seen, 1),
            "validation_raw_scale_positive_mae": validation_mae,
            "seconds": time.perf_counter() - epoch_started,
        }
        history_rows.append(row)
        print(json.dumps({"city": config.name, "h": history_hours, **row}), flush=True)
        if validation_mae < best_validation - 1.0e-8:
            best_validation = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("LSTM did not produce a checkpoint")
    model.load_state_dict(best_state)

    @torch.no_grad()
    def predict_full(hours: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        for begin in range(0, len(hours), micro_batch):
            part = hours[begin : begin + micro_batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                value = raw_prediction(model(history_tensor(part)))
            outputs.append(value.cpu().numpy().reshape(len(part), config.nodes, config.nodes))
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def predict_samples(hours: np.ndarray, edge: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(hours, return_inverse=True)
        full = predict_full(unique).reshape(len(unique), edge_count)
        return full[inverse, edge]

    evaluation, inference_seconds = evaluate_hurdle(
        config,
        od,
        validation_hours,
        test_hours,
        history_hours,
        predict_full,
        micro_batch,
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    payload = base_payload(
        config,
        "Flattened OD LSTM",
        history_hours,
        training_seconds,
        inference_seconds,
        evaluation,
        seed,
    )
    payload.update(
        {
            "category": "basic deep learning",
            "device": config.device,
            "architecture": "one-layer LSTM over flattened OD matrices",
            "hidden_size": hidden,
            "parameter_count": parameter_count,
            "training_loss": "SmoothL1 on standardized log1p positive cells only",
            "validation_selection_metric": "raw-scale positive-cell MAE",
            "best_validation_raw_scale_positive_mae": best_validation,
            "best_epoch": best_epoch,
            "epochs_run": len(history_rows),
            "micro_batch_hours": micro_batch,
            "effective_batch_hours": effective_batch,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "library_version": torch.__version__,
        }
    )
    run_dir = config.output_root / "lstm" / (
        "h%d_seed%d" % (history_hours, seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best_model.pt")
    write_json(run_dir / "history.json", history_rows)
    write_json(run_dir / "result.json", payload)
    write_json(run_dir / "config.json", {k: v for k, v in payload.items() if k != "evaluation"})
    if history_hours == 12:
        cache_key = "supplemental_lstm" + ("" if seed == 0 else "_seed%d" % seed)
        cache_event_magnitude(config, cache_key, predict_samples, payload)
    model = best_state = None
    torch.cuda.empty_cache()
    gc.collect()
    return payload
