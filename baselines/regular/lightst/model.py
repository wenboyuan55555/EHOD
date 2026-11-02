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
LIGHTST_ROOT = Path(__file__).resolve().parent / "upstream"

def _load_lightst_module():
    """Load the pinned LightST core with minimal one-step/device fixes.

    The released source creates new CUDA-only Linear layers inside every
    forward call.  Those layers are consequently random, unregistered, and
    absent from checkpoints.  We register the same projections in each model
    constructor and use the selected device.  For the requested next-step
    task, the teacher embedding projection targets ``output_window`` rather
    than the original 12-step ``input_window`` so teacher and student
    embeddings retain matching shapes for H in {1, 3, 6, 12}.
    """

    name = "od9_pinned_upstream_lightst_model"
    if name not in sys.modules:
        path = LIGHTST_ROOT / "model/Teacher.py"
        source = path.read_text(encoding="utf-8")
        end_conv = (
            "\t\tself.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,\n"
            "\t\t\t\t\t\t\t\t\tout_channels=self.output_window, kernel_size=(1, 1), bias=True)"
        )
        registered = end_conv + (
            "\n\t\tself.od9_temporal_projection = nn.Linear("
            "self.residual_channels, self.output_window)"
            "\n\t\tself.od9_embedding_projection = nn.Linear(1, 32)"
        )
        if source.count(end_conv) != 2:
            raise RuntimeError("Pinned LightST constructor signature changed")
        patched = source.replace(end_conv, registered)
        teacher_old = (
            "\t\tttout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, "
            "self.input_window).cuda()(tout[-1].transpose(1,3)).transpose(1,3))\n"
            "\t\tssout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, "
            "self.input_window).cuda()(sout[-1].transpose(1,3)).transpose(1,3))"
        )
        teacher_new = (
            "\t\tttout = self.od9_embedding_projection("
            "self.od9_temporal_projection(tout[-1].transpose(1,3)).transpose(1,3))\n"
            "\t\tssout = self.od9_embedding_projection("
            "self.od9_temporal_projection(sout[-1].transpose(1,3)).transpose(1,3))"
        )
        student_old = "\t\tx_ = nn.Linear(1, 32).cuda()(x)  #final out:  64, 12, 170, 1"
        student_new = "\t\tx_ = self.od9_embedding_projection(x)"
        if teacher_old not in patched or student_old not in patched:
            raise RuntimeError("Pinned LightST forward signature changed")
        patched = patched.replace(teacher_old, teacher_new).replace(
            student_old, student_new
        )
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(patched, str(path), "exec"), module.__dict__)
        module._od9_runtime_patch_sha256 = hashlib.sha256(
            patched.encode("utf-8")
        ).hexdigest()
    return sys.modules[name]

def _lightst_arguments(
    history_hours: int,
    number_of_nodes: int,
    adjacency,
    device: torch.device,
):
    return SimpleNamespace(
        adj_mx=adjacency,
        num_nodes=number_of_nodes,
        input_dim=1,
        input_window=history_hours,
        output_window=1,
        output_dim=1,
        device=device,
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        dropout=0.3,
        subgraph_size=min(20, number_of_nodes),
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        use_curriculum_learning=True,
        task_level=0,
    )

class LightSTMarginalPair(nn.Module):
    def __init__(self, teacher_origin, teacher_destination, student_origin, student_destination):
        super().__init__()
        self.teacher_origin = teacher_origin
        self.teacher_destination = teacher_destination
        self.student_origin = student_origin
        self.student_destination = student_destination

    def teacher_forward(self, history: torch.Tensor):
        origin = self.teacher_origin(history[..., 0:1])
        destination = self.teacher_destination(history[..., 1:2])
        prediction = torch.stack((origin[0][:, 0, :, 0], destination[0][:, 0, :, 0]), dim=-1)
        return prediction, (origin[1], destination[1]), (origin[2], destination[2])

    def student_forward(self, history: torch.Tensor):
        origin = self.student_origin(history[..., 0:1])
        destination = self.student_destination(history[..., 1:2])
        prediction = torch.stack((origin[0][:, 0, :, 0], destination[0][:, 0, :, 0]), dim=-1)
        return prediction, (origin[1], destination[1])

    def forward(self, history: torch.Tensor):
        return self.student_forward(history)[0]

def create_lightst(
    history_hours: int,
    number_of_nodes: int,
    adjacency,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    module = _load_lightst_module()
    arguments = _lightst_arguments(history_hours, number_of_nodes, adjacency, device)
    pair = LightSTMarginalPair(
        module.Teacher(arguments),
        module.Teacher(arguments),
        module.STMLP(arguments),
        module.STMLP(arguments),
    ).to(device)
    return pair, {
        "source_root": str(LIGHTST_ROOT),
        "model_file_sha256": "bdda700f8f5cf02f42f1df2171ebb229daab2131b68ad69ae29e60e6f3b6cfdf",
        "runtime_patch_sha256": module._od9_runtime_patch_sha256,
        "source_checkpoint_sha256": "316d2cd3e33f533c0fb5f45e5b7eb90092265911d8f0774b2d6e36b91da89173",
        "source_checkpoint_used": False,
        "source_checkpoint_reason": "PEMS04-specific node/output shapes; teacher is retrained on each target city's training split",
        "compatibility_patches": [
            "register source forward-created Linear projections",
            "remove hard-coded cuda device from those projections",
            "project teacher representation to one requested output step",
        ],
        "teacher": "upstream adaptive-graph MTGNN core",
        "student": "upstream graph-free STMLP core",
        "distillation_weights": {"kl": 10.0, "spatial_contrastive": 1.0},
        "two_independent_marginal_networks": True,
    }
