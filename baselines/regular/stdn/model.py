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
STDN_ROOT = Path(__file__).resolve().parent / "upstream"

def _stub_unused_torch_geometric() -> None:
    """Let STDN import when its unused demonstration GCN class is unavailable."""

    try:
        import torch_geometric  # noqa: F401

        return
    except ImportError:
        pass
    root = types.ModuleType("torch_geometric")
    nn_module = types.ModuleType("torch_geometric.nn")
    data_module = types.ModuleType("torch_geometric.data")

    class UnavailableGCNConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("The unused STDN demonstration GCNConv path was invoked")

    class UnavailableData:
        pass

    nn_module.GCNConv = UnavailableGCNConv
    data_module.Data = UnavailableData
    root.nn = nn_module
    root.data = data_module
    sys.modules["torch_geometric"] = root
    sys.modules["torch_geometric.nn"] = nn_module
    sys.modules["torch_geometric.data"] = data_module

def _load_stdn_module(device: torch.device):
    _stub_unused_torch_geometric()
    name = "od9_pinned_upstream_stdn_model"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STDN_ROOT / "model.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned STDN model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    # The source keeps a module-level cuda:0 constant.  Redirecting it is a
    # device-only compatibility fix; all equations and modules are untouched.
    module.device = device
    return module

def _stdn_config(history_hours: int, number_of_nodes: int) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "Data": {
                "dataset_name": "EADT_OD_marginal",
                "num_of_vertices": str(number_of_nodes),
                "time_slice_size": "60",
            },
            "Training": {
                "gpu": "1",
                "L": "2",
                "K": "16",
                "d": "8",
                "node_miss_rate": "0.1",
                "T_miss_len": str(history_hours),
                "order": "3",
                "reference": "3",
                "num_his": str(history_hours),
                # The upstream decoder assumes num_pred == num_his.  It emits
                # H calendar-conditioned steps; the task wrapper selects the
                # first (next-hour) step and never supplies future traffic.
                "num_pred": str(history_hours),
                "in_channels": "1",
                "out_channels": "1",
            },
        }
    )
    return config

class STDNMarginalPair(nn.Module):
    def __init__(self, origin, destination, laplacian_pe: torch.Tensor):
        super().__init__()
        self.origin = origin
        self.destination = destination
        self.register_buffer("laplacian_pe", laplacian_pe.float())

    def forward(self, history: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        origin = self.origin(
            history[..., 0:1], temporal, self.laplacian_pe, "train"
        )[:, :1, :, 0]
        destination = self.destination(
            history[..., 1:2], temporal, self.laplacian_pe, "train"
        )[:, :1, :, 0]
        return torch.stack((origin, destination), dim=-1)

def create_stdn(
    history_hours: int,
    number_of_nodes: int,
    laplacian_pe: torch.Tensor,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    module = _load_stdn_module(device)
    config = _stdn_config(history_hours, number_of_nodes)
    origin = module.make_model(config, bn_decay=0.1).to(device)
    destination = module.make_model(config, bn_decay=0.1).to(device)
    model = STDNMarginalPair(origin, destination, laplacian_pe.to(device)).to(device)
    return model, {
        "source_root": str(STDN_ROOT),
        "model_file_sha256": "6f11f7190f312d959c9de21713f2a8aca3c3fd2b8b113f491bc86a10a82d4317",
        "device_patch": "redirect module-level cuda:0 constant to selected device",
        "unused_dependency_patch": "stub torch_geometric only for unused demonstration GCN class",
        "internal_num_pred": history_hours,
        "reported_pred_len": 1,
        "K": 16,
        "d": 8,
        "layers": 2,
        "graph_order": 3,
        "two_independent_marginal_networks": True,
    }
