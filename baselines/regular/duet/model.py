"""Thin, auditable loaders for locally supplied upstream baseline models."""

from __future__ import annotations

import importlib
import importlib.util
import configparser
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Type

import torch
import torch.nn as nn



from baselines.regular._shared.upstream import _install_namespace
DUET_ROOT = Path(__file__).resolve().parent / "upstream"
DUET_MODEL_SHA256 = "840690b13fecdd08fdc4a602f3d700ede940731a29ad0fe1de7830b7bb9b22b7"

def load_duet_model_class() -> Tuple[Type[torch.nn.Module], str]:
    """Load the author's DUETModel and apply one verified H=1 shape fix.

    Importing `ts_benchmark.baselines.duet` normally imports the entire TFB
    collection and fails on an unrelated optional `reformer_pytorch` module.
    Namespaces let us import the declared DUET core directly.  The upstream
    dispatcher also calls `.squeeze(1)`, which deletes the temporal dimension
    only when H=1.  Removing that squeeze is shape-preserving for H>1 and is
    necessary for the requested short-history task.
    """

    if not (DUET_ROOT / "ts_benchmark/baselines/duet/models/duet_model.py").exists():
        raise FileNotFoundError("Pinned local DUET source is missing: %s" % DUET_ROOT)
    if str(DUET_ROOT) not in sys.path:
        sys.path.insert(0, str(DUET_ROOT))
    package = "ts_benchmark.baselines.duet"
    _install_namespace(package, DUET_ROOT / "ts_benchmark/baselines/duet")
    _install_namespace(package + ".models", DUET_ROOT / "ts_benchmark/baselines/duet/models")
    cluster_module = importlib.import_module(package + ".layers.linear_extractor_cluster")

    def dispatch_preserving_short_history(self, value):
        expanded = value[self._batch_index]
        return torch.split(expanded, self._part_sizes, dim=0)

    cluster_module.SparseDispatcher.dispatch = dispatch_preserving_short_history
    model_module = importlib.import_module(package + ".models.duet_model")
    return model_module.DUETModel, "remove-dispatch-squeeze-for-H1"

def create_duet(history_hours: int, number_of_variables: int) -> Tuple[torch.nn.Module, dict]:
    model_class, compatibility_patch = load_duet_model_class()
    # Preserve the upstream unified defaults; only the task dimensions and
    # output horizon are changed.  Batch size and optimizer live in the runner.
    config = SimpleNamespace(
        CI=True,
        enc_in=number_of_variables,
        seq_len=history_hours,
        pred_len=1,
        factor=1,
        dropout=0.2,
        output_attention=False,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        e_layers=2,
        activation="gelu",
        fc_dropout=0.2,
        num_experts=4,
        noisy_gating=True,
        k=1,
        hidden_size=256,
        moving_avg=25,
    )
    model = model_class(config)
    return model, {
        **vars(config),
        "compatibility_patch": compatibility_patch,
        "source_root": str(DUET_ROOT),
        "model_file_sha256": DUET_MODEL_SHA256,
    }
