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
import numpy as np



from baselines.regular._shared.upstream import _install_namespace
PATCHSTG_ROOT = Path(__file__).resolve().parent / "upstream"

def _install_timm_vision_transformer_compatibility() -> str:
    """Provide the exact timm interfaces used by PatchSTG when timm is absent."""

    try:
        import timm.models.vision_transformer  # noqa: F401

        return "installed-timm"
    except ImportError:
        pass

    class Attention(nn.Module):
        def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            **kwargs,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, value):
            batch, length, dimension = value.shape
            qkv = self.qkv(value).reshape(
                batch, length, 3, self.num_heads, dimension // self.num_heads
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, content = qkv.unbind(0)
            attention = (query @ key.transpose(-2, -1)) * self.scale
            attention = self.attn_drop(attention.softmax(dim=-1))
            value = (attention @ content).transpose(1, 2).reshape(batch, length, dimension)
            return self.proj_drop(self.proj(value))

    class Mlp(nn.Module):
        def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
            **kwargs,
        ):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.drop1 = nn.Dropout(drop)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop2 = nn.Dropout(drop)

        def forward(self, value):
            return self.drop2(self.fc2(self.drop1(self.act(self.fc1(value)))))

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vision = types.ModuleType("timm.models.vision_transformer")
    vision.Attention = Attention
    vision.Mlp = Mlp
    timm.models = models
    models.vision_transformer = vision
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.vision_transformer"] = vision
    return "local-compatible-Attention-and-Mlp"

def _load_patchstg_modules():
    timm_source = _install_timm_vision_transformer_compatibility()
    model_name = "od9_pinned_upstream_patchstg_model"
    if model_name not in sys.modules:
        path = PATCHSTG_ROOT / "models/model.py"
        source = path.read_text(encoding="utf-8")
        # Upstream forces temporal indices back to CPU.  `.long()` preserves
        # their current device and is numerically identical.
        patched = source.replace(".type(torch.LongTensor)", ".long()")
        module = types.ModuleType(model_name)
        module.__file__ = str(path)
        sys.modules[model_name] = module
        exec(compile(patched, str(path), "exec"), module.__dict__)
        module._od9_runtime_patch_sha256 = hashlib.sha256(
            patched.encode("utf-8")
        ).hexdigest()
    utils_name = "od9_pinned_upstream_patchstg_utils"
    if utils_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            utils_name, PATCHSTG_ROOT / "lib/utils.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned PatchSTG utils.py")
        utils = importlib.util.module_from_spec(spec)
        sys.modules[utils_name] = utils
        spec.loader.exec_module(utils)
    return sys.modules[model_name], sys.modules[utils_name], timm_source

class PatchSTGMarginalPair(nn.Module):
    def __init__(self, origin, destination, history_hours: int):
        super().__init__()
        self.origin = origin
        self.destination = destination
        # Upstream returns H forecast channels.  A learned one-step task head
        # is the smallest output adapter that retains both attention paths.
        self.origin_head = nn.Conv2d(history_hours, 1, kernel_size=1)
        self.destination_head = nn.Conv2d(history_hours, 1, kernel_size=1)

    def forward(self, history: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        origin = self.origin(history[..., 0:1], temporal)
        destination = self.destination(history[..., 1:2], temporal)
        origin = self.origin_head(origin)[:, :, :, 0]
        destination = self.destination_head(destination)[:, :, :, 0]
        return torch.stack((origin, destination), dim=-1)

def create_patchstg(
    history_hours: int,
    number_of_nodes: int,
    locations: np.ndarray,
    train_affinity: np.ndarray,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    # Imported lazily to keep the DUET-only path dependency-light.
    import numpy as np

    model_module, utils_module, timm_source = _load_patchstg_modules()
    recursion = 4 if number_of_nodes > 128 else 3
    spatial_patch_number = 2 ** recursion
    spatial_patch_size = int(np.ceil(number_of_nodes / spatial_patch_number))
    parts = utils_module.kdTree(np.asarray(locations).T, recursion, 0)
    original, reordered, all_reordered = utils_module.reorderData(
        parts, np.asarray(train_affinity), spatial_patch_size
    )
    common = dict(
        tem_patchsize=1,
        tem_patchnum=history_hours,
        node_num=number_of_nodes,
        spa_patchsize=spatial_patch_size,
        spa_patchnum=spatial_patch_number,
        tod=24,
        dow=7,
        layers=5,
        factors=2,
        input_dims=64,
        node_dims=32,
        tod_dims=32,
        dow_dims=32,
        ori_parts_idx=original,
        reo_parts_idx=reordered,
        reo_all_idx=all_reordered,
    )
    origin = model_module.PatchSTG(**common)
    destination = model_module.PatchSTG(**common)
    model = PatchSTGMarginalPair(origin, destination, history_hours).to(device)
    return model, {
        "source_root": str(PATCHSTG_ROOT),
        "model_file_sha256": "80064af49ef56e0478bee25f7aa21a6bdad5320fda3850b8785e2f473e5f0a18",
        "runtime_patch_sha256": model_module._od9_runtime_patch_sha256,
        "device_patch": "replace forced CPU LongTensor cast with device-preserving long cast",
        "timm_dependency": timm_source,
        "temporal_patch_size": 1,
        "temporal_patch_number": history_hours,
        "spatial_recursion": recursion,
        "spatial_patch_number": spatial_patch_number,
        "spatial_patch_size": spatial_patch_size,
        "padded_nodes": int(len(all_reordered) - number_of_nodes),
        "one_step_learned_output_head": True,
        "two_independent_marginal_networks": True,
    }
