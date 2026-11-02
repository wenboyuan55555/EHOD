from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.event.t3.model import T3STID
from baselines.event.fuse.model import FUSED2STGNN
from baselines.event.concat.model import FUSEConcatD2STGNN
from baselines.event.ss_moe.model import SSMoE
from baselines.event.egaf.model import EGAFNet
from baselines.event.qa_stgacn.model import QASTGACN
from baselines.event.rast.model import RAST
from baselines.event.cdgon.model import CDGON

MODEL_NAMES = ('t3_stid', 'fuse_d2stgnn', 'fuse_concat', 'ss_moe', 'egaf', 'qa_stgacn', 'rast', 'cdgon')

def create_original_event_model(
    name: str, history: int, nodes: int, event_dim: int
) -> nn.Module:
    if name == "t3_stid":
        return T3STID(history, nodes, event_dim)
    if name == "fuse_d2stgnn":
        return FUSED2STGNN(nodes, event_dim)
    if name == "fuse_concat":
        return FUSEConcatD2STGNN(nodes, event_dim)
    if name == "ss_moe":
        return SSMoE(history, nodes, event_dim)
    if name == "egaf":
        return EGAFNet(nodes, event_dim)
    if name == "qa_stgacn":
        return QASTGACN(nodes, event_dim)
    if name == "rast":
        return RAST(history, nodes, event_dim)
    if name == "cdgon":
        return CDGON(nodes, event_dim)
    raise KeyError(name)
