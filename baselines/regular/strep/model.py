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
STREP_ROOT = Path(__file__).resolve().parent / "upstream"

class STRePMarginalPair(nn.Module):
    def __init__(self, origin, destination):
        super().__init__()
        self.origin = origin
        self.destination = destination

    def forward(
        self, history: torch.Tensor, temporal: torch.Tensor, target: torch.Tensor
    ):
        origin = self.origin(
            [history[..., 0:1], temporal, target[..., 0:1]]
        )
        destination = self.destination(
            [history[..., 1:2], temporal, target[..., 1:2]]
        )
        input_value = torch.cat((origin[0], destination[0]), dim=-1)
        reconstruction = torch.cat((origin[1], destination[1]), dim=-1)
        target_value = torch.cat((origin[2], destination[2]), dim=-1)
        prediction = torch.cat((origin[3], destination[3]), dim=-1)
        return input_value, reconstruction, target_value, prediction

    @torch.no_grad()
    def representations(self, history: torch.Tensor, temporal: torch.Tensor):
        origin, _ = self.origin.get_rep([history[..., 0:1], temporal])
        destination, _ = self.destination.get_rep([history[..., 1:2], temporal])
        return origin[:, -1:], destination[:, -1:]

def create_strep(
    history_hours: int, number_of_nodes: int, device: torch.device
) -> Tuple[nn.Module, dict]:
    if str(STREP_ROOT) not in sys.path:
        sys.path.insert(0, str(STREP_ROOT))
    name = "od9_pinned_upstream_strep_model"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STREP_ROOT / "models/STReP.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned ST-ReP model")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    arguments = SimpleNamespace(
        input_length=history_hours,
        predict_length=1,
        in_dim=1,
        pre_dim=1,
        num_nodes=number_of_nodes,
        tau=3,
        d_model=64,
        hasCross=1,
        num_layers=3,
        mask_ratio=0.25,
        loss_type="pred_mask_ms",
        st_emb_dropout=0.1,
        slice_size_per_day=24,
        M=8,
        n_heads=2,
        spatial_dropout=0.1,
        spatial_att_dropout=0.1,
        time_factor=min(3, history_hours),
        rep_dropout=0.1,
        note="EADT_OD_marginal",
    )
    origin = module.Model(arguments).float()
    destination = module.Model(arguments).float()
    for model in (origin, destination):
        for parameter in model.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
    pair = STRePMarginalPair(origin, destination).to(device)
    return pair, {
        "source_root": str(STREP_ROOT),
        "model_file_sha256": "22b83af9aad1fd712661c404734396ba611077caa4081e5d91389e998ad96dd0",
        "downstream_file_sha256": "9810862e3984893f396a8a2f0d74c89842b21d1862f1481b5314118006adfca6",
        "d_model": 64,
        "spatial_proxy_tokens": 8,
        "time_factor": min(3, history_hours),
        "layers": 3,
        "mask_ratio": 0.25,
        "loss": "prediction + masked reconstruction + feasible multi-scale Huber",
        "downstream": "upstream Ridge alpha grid",
        "two_independent_marginal_encoders": True,
    }
