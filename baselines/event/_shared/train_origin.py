from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ehod.data.od import ACTIVE_CITY, EXPERIMENT_ROOT, N_NODES, history_indices, iter_hour_batches, load_od, load_static
from ehod.common.device import require_cuda
from ehod.data.events import ScheduledEventStore
from ehod.evaluation.metrics import StreamingMetrics
from baselines.event._shared.factory import MODEL_NAMES, T3STID, create_original_event_model
from baselines.event._shared.data import CACHE_DIR, TEXT_MODEL, load_original_event_cache


SOURCE = {
    "t3_stid": {
        "paper": "Event Traffic Forecasting with Sparse Multimodal Data",
        "official_code": "https://github.com/2448845600/EventTrafficForecasting",
        "commit": "0d8ce0b959b7232345cc592c5c874c5219317489",
        "mechanism": "frozen STID traffic encoder; pooled event-text embedding; multimodal and traffic-only heads averaged",
    },
    "fuse_d2stgnn": {
        "paper": "FUSE-Traffic",
        "doi": "10.1145/3748636.3762776",
        "official_code": "https://github.com/GeoAICenter/FUSE-Traffic_Sigspatial2025",
        "commit": "46abd153192b9ea378b16e3cc69f8ea8bbfe64c5",
        "mechanism": "D2STGNN-style diffusion/inherent decoupling followed by traffic-query to event-text cross attention",
    },
    "fuse_concat": {
        "paper": "FUSE-Traffic (Concatenation ablation)",
        "doi": "10.1145/3748636.3762776",
        "official_code": "https://github.com/GeoAICenter/FUSE-Traffic_Sigspatial2025",
        "commit": "46abd153192b9ea378b16e3cc69f8ea8bbfe64c5",
        "mechanism": "the paper's FUSE-Traffic-Concatenation ablation: D2STGNN traffic and pooled event-text embeddings are concatenated without cross-modal alignment",
    },
    "ss_moe": {
        "paper": "LLM-Enhanced Mixture-of-Experts for Explainable Traffic Forecasting under Extreme Events",
        "official_code": None,
        "commit": None,
        "mechanism": "periodic/trend/instant state with spatial, semantic-event and general experts plus learned router",
    },
    "egaf": {
        "paper": "Event-Guided Attention Fusion Network for traffic flow prediction",
        "official_code": None,
        "commit": None,
        "mechanism": "event-guided temporal attention followed by graph-masked spatial structured attention",
    },
    "qa_stgacn": {
        "paper": "Integrating Query Data for Enhanced Traffic Forecasting: A Spatio-Temporal Graph Attention Convolution Network Approach with Delay Modeling",
        "doi": "10.1016/j.knosys.2024.112315",
        "official_code": None,
        "commit": None,
        "mechanism": "event-conditioned node queries drive graph attention over spatiotemporal traffic features",
    },
    "rast": {
        "paper": "Retrieval-Augmented Spatio-Temporal Forecasting",
        "official_code": "https://github.com/RWLinno/RAST",
        "commit": "4b48a47a002dd8d4151afe47be8f436f0c028a67",
        "mechanism": "decoupled temporal/spatial query, training-only retrieval store, attention fusion and residual prediction",
    },
    "cdgon": {
        "paper": "Physics-informed Neural ODE for Post-disaster Mobility Recovery",
        "doi": "10.1145/3637528.3672027",
        "official_code": "https://github.com/tsinghua-fib-lab/CDGON-KDD24",
        "commit": "5bf8d114a485cacc886a5645119edce35479f4a4",
        "mechanism": "one-step graph neural ODE with physics recovery-to-normal term and event-severity residual",
    },
}

DEVIATIONS = {
    "t3_stid": [
        "NYC/Chicago have no released Voyage event vectors, so the frozen text encoder is all-MiniLM-L6-v2",
        "STID is ported from the official implementation but predicts paired origin/destination zone marginals",
    ],
    "fuse_d2stgnn": [
        "official D2STGNN decoupling and cross-modal equations are ported into the unified one-hour runner",
        "forecast-safe schedule text replaces the source dataset's Gemini-generated traffic-event text",
    ],
    "fuse_concat": [
        "the published concatenation ablation is applied to the same D2STGNN task bridge as FUSE-Traffic",
        "forecast-safe schedule text replaces the source dataset's Gemini-generated traffic-event text",
    ],
    "ss_moe": [
        "the paper's unavailable Llama-3-1B prompt expert is represented by the same frozen sentence encoder used by all text baselines",
        "zone OD marginals replace pickup/dropoff node demand",
    ],
    "egaf": [
        "no author code was located; the event-guided temporal/spatial attention is implemented from the paper description",
        "zone OD marginals replace the source traffic variable",
    ],
    "qa_stgacn": [
        "no author code was located; query-aware graph attention is implemented from the paper description",
        "zone OD marginals replace the source traffic variable",
    ],
    "rast": [
        "official query/retrieval/attention design is ported; retrieval is precomputed with a CPU FAISS training-only store",
        "zone OD marginals replace sensor traffic",
    ],
    "cdgon": [
        "post-disaster recovery is converted to rolling one-step scheduled-event forecasting",
        "the last observed state initializes one graph-ODE step and hour-of-week training means define normal mobility",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--history-hours", type=int, choices=(12,), default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--max-train-hours", type=int, default=None)
    parser.add_argument("--max-eval-hours", type=int, default=None)
    parser.add_argument("--ipf-iterations", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor(value: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    result = torch.from_numpy(np.asarray(value))
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device, non_blocking=True)


def make_batch(
    cache: Dict[str, np.ndarray],
    starts: np.ndarray,
    history_hours: int,
    adjacency: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    starts = np.asarray(starts, dtype=np.int32)
    return {
        "history": tensor(
            cache["marginal_normalized"][history_indices(starts, history_hours)],
            device,
            torch.float32,
        ),
        "target": tensor(cache["marginal_normalized"][starts], device, torch.float32),
        "hour": tensor(starts % 24, device, torch.long),
        "weekday": tensor((starts // 24) % 7, device, torch.long),
        "event_node": tensor(cache["event_node"][starts], device, torch.float32),
        "event_tokens": tensor(cache["event_tokens"][starts], device, torch.float32),
        "event_token_mask": tensor(cache["event_token_mask"][starts], device, torch.bool),
        "event_global": tensor(cache["event_global"][starts], device, torch.float32),
        "retrieved": tensor(cache["retrieved"][starts], device, torch.float32),
        "seasonal_marginal": tensor(cache["seasonal_marginal"][starts % 168], device, torch.float32),
        "adjacency": adjacency,
    }


def inverse_marginal(
    scaled: torch.Tensor, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    center = torch.as_tensor(mean, device=scaled.device)
    spread = torch.as_tensor(std, device=scaled.device)
    return torch.expm1(scaled.float() * spread + center).clamp_min(0.0)


@torch.no_grad()
def validation_mae(
    model,
    cache,
    adjacency,
    device,
    history_hours,
    batch_size,
    limit,
    traffic_only: bool = False,
) -> float:
    model.eval()
    absolute = 0.0
    count = 0
    for starts in iter_hour_batches("validation", batch_size, limit=limit):
        batch = make_batch(cache, starts, history_hours, adjacency, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            if traffic_only:
                prediction = model.traffic_only(
                    batch["history"], batch["hour"], batch["weekday"]
                )
            else:
                prediction = model(batch)
        prediction = inverse_marginal(prediction, cache["mean"], cache["std"])
        target = tensor(cache["marginal_raw"][starts], device, torch.float32)
        absolute += float((prediction - target).abs().sum().item())
        count += int(target.numel())
    return absolute / max(count, 1)


def train_stage(
    model,
    cache,
    adjacency,
    device,
    args,
    output_dir: Path,
    stage: str,
    epochs: int,
    traffic_only: bool = False,
) -> Dict:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()
    checkpoint = output_dir / (stage + "_best.pt")
    best = math.inf
    stale = 0
    history_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        started = time.time()
        for starts in iter_hour_batches(
            "train",
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch + (1000 if traffic_only else 0),
            limit=args.max_train_hours,
        ):
            batch = make_batch(cache, starts, args.history_hours, adjacency, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if traffic_only:
                    prediction = model.traffic_only(
                        batch["history"], batch["hour"], batch["weekday"]
                    )
                else:
                    prediction = model(batch)
                loss = F.smooth_l1_loss(prediction, batch["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(starts)
            count += len(starts)
        validation = validation_mae(
            model,
            cache,
            adjacency,
            device,
            args.history_hours,
            args.batch_size,
            8 if args.smoke else args.max_eval_hours,
            traffic_only=traffic_only,
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "validation_marginal_mae": validation,
            "seconds": time.time() - started,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if validation < best - 1.0e-7:
            best = validation
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation_mae": validation},
                checkpoint,
            )
        else:
            stale += 1
        (output_dir / (stage + "_history.json")).write_text(
            json.dumps(history_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if stale >= args.patience:
            break
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    return {
        "best_epoch": int(payload["epoch"]),
        "validation_marginal_mae": float(payload["validation_mae"]),
        "epochs_run": len(history_rows),
    }


def balance_marginals(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outbound = np.maximum(value[..., 0], 0.0).astype(np.float64)
    inbound = np.maximum(value[..., 1], 0.0).astype(np.float64)
    target_total = 0.5 * (outbound.sum(axis=1) + inbound.sum(axis=1))
    outbound *= target_total[:, None] / np.maximum(outbound.sum(axis=1, keepdims=True), 1.0e-8)
    inbound *= target_total[:, None] / np.maximum(inbound.sum(axis=1, keepdims=True), 1.0e-8)
    return outbound, inbound


def reconstruct_od(
    prior: np.ndarray,
    marginal: np.ndarray,
    iterations: int,
) -> np.ndarray:
    outbound, inbound = balance_marginals(marginal)
    matrix = np.maximum(np.asarray(prior, dtype=np.float64), 0.0) + 1.0e-7
    for _ in range(iterations):
        matrix *= (outbound / np.maximum(matrix.sum(axis=2), 1.0e-12))[:, :, None]
        matrix *= (inbound / np.maximum(matrix.sum(axis=1), 1.0e-12))[:, None, :]
    return matrix.astype(np.float32)


@torch.no_grad()
def evaluate(model, cache, adjacency, device, args) -> Dict:
    model.eval()
    od = load_od()
    seasonal_prior = np.asarray(load_static()["seasonal_mean"], dtype=np.float32)
    event_store = ScheduledEventStore()
    global_metrics = StreamingMetrics()
    direct_metrics = StreamingMetrics()
    marginal_metrics = StreamingMetrics()
    test_hours = None if args.max_eval_hours is None else args.max_eval_hours
    inference_seconds = 0.0
    for starts in iter_hour_batches("test", args.batch_size, limit=test_hours):
        batch = make_batch(cache, starts, args.history_hours, adjacency, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            prediction_scaled = model(batch)
        prediction_marginal = inverse_marginal(
            prediction_scaled, cache["mean"], cache["std"]
        ).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        target_marginal = np.asarray(cache["marginal_raw"][starts], dtype=np.float32)
        marginal_metrics.update(
            torch.from_numpy(prediction_marginal), torch.from_numpy(target_marginal)
        )
        prediction = reconstruct_od(
            seasonal_prior[starts % 168], prediction_marginal, args.ipf_iterations
        )
        target = np.asarray(od[starts], dtype=np.float32)
        global_metrics.update(torch.from_numpy(prediction), torch.from_numpy(target))
        mask = event_store.affected_edge_mask(starts)
        if mask.any():
            direct_metrics.update(
                torch.from_numpy(prediction[mask]), torch.from_numpy(target[mask])
            )
    return {
        "global": global_metrics.compute(),
        "event_direct": direct_metrics.compute(),
        "marginal": marginal_metrics.compute(),
        "inference_seconds": inference_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.pretrain_epochs = 2
        args.patience = 2
        args.max_train_hours = 32
        args.max_eval_hours = 8
    set_seed(args.seed)
    device = require_cuda(args.device)
    output_dir = (
        EXPERIMENT_ROOT
        / "runs"
        / "original_event_baselines"
        / args.model
        / ("h12_seed%d_%s" % (args.seed, "smoke" if args.smoke else "formal"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_original_event_cache()
    static = load_static()
    adjacency = tensor(static["adjacency"], device, torch.float32)
    event_dim = int(cache["event_global"].shape[1])
    model = create_original_event_model(
        args.model, args.history_hours, N_NODES, event_dim
    ).to(device)
    started = time.time()
    pretrain = None
    if isinstance(model, T3STID):
        pretrain = train_stage(
            model,
            cache,
            adjacency,
            device,
            args,
            output_dir,
            "traffic_pretrain",
            args.pretrain_epochs,
            traffic_only=True,
        )
        model.freeze_traffic_encoder()
    formal = train_stage(
        model,
        cache,
        adjacency,
        device,
        args,
        output_dir,
        "event_model",
        args.epochs,
    )
    evaluation = evaluate(model, cache, adjacency, device, args)
    source = dict(SOURCE[args.model])
    summary = {
        "city": ACTIVE_CITY,
        "model": args.model,
        "history_hours": args.history_hours,
        "seed": args.seed,
        "status": "paper-architecture task adaptation",
        "not_original_task_reproduction": True,
        "source": source,
        "official_source_basis_used_when_available": source["official_code"] is not None,
        "adaptation_deviations": DEVIATIONS[args.model],
        "task_bridge": {
            "traffic_target": "next-hour origin outflow and destination inflow per zone",
            "od_reconstruction": "iterative proportional fitting",
            "od_prior": "training-only mean OD matrix by hour of week",
            "ipf_iterations": args.ipf_iterations,
        },
        "event_text_encoder": TEXT_MODEL,
        "forecast_safe": True,
        "strict_split": {"train": [12, 6720], "validation": [6720, 7728], "test": [7728, 8760]},
        "pretrain": pretrain,
        "training": formal,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "training_seconds": time.time() - started,
        "cache": str(CACHE_DIR),
        "evaluation": evaluation,
        "device": str(device),
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
