"""Strict-window BootCF adapter using exact continuous event summaries."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
from torch import nn


from ehod.common.upstream import load_snapshot
PAPER_CODE = Path(__file__).resolve().parent / "upstream"
_upstream = load_snapshot(PAPER_CODE, 'BootCF.py', "_ehod_upstream_bootcf", aliases=('modules',))
MLP_Predictor, MLPMessageFunction = _upstream.MLP_Predictor, _upstream.MLPMessageFunction


class WindowCTDGEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        memory_dimension: int,
        message_dimension: int,
        output_seconds: float = 3600.0,
        init_lamb: float = 0.5,
        window_seconds: float = 12.0 * 3600.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.output_seconds = output_seconds
        self.window_seconds = float(window_seconds)
        raw_dimension = memory_dimension + n_nodes
        self.message_function = MLPMessageFunction(raw_dimension, message_dimension)
        self.embedding_transform = nn.Sequential(
            nn.Linear(memory_dimension * 2, memory_dimension),
            nn.LeakyReLU(),
            nn.Linear(memory_dimension, memory_dimension),
            nn.LeakyReLU(),
        )
        self.e_lamb = nn.Parameter(torch.tensor([init_lamb], dtype=torch.float32))
        self.static_road_embedding = nn.Embedding(n_nodes, memory_dimension)

    def aggregate_messages(self, weights: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = weights.shape
        zeros = torch.zeros(
            batch,
            nodes,
            self.memory_dimension,
            device=weights.device,
            dtype=weights.dtype,
        )
        row_weight = weights.sum(dim=-1, keepdim=True)
        raw = torch.cat([zeros, weights], dim=-1)
        message = self.message_function(raw)
        active = (row_weight > 0).to(message.dtype)
        return torch.cat([message * active, row_weight], dim=-1)

    def embeddings_from_messages(
        self, messages: torch.Tensor, last_event_seconds: torch.Tensor
    ) -> torch.Tensor:
        active = messages[..., -1:] > 0
        initial_decay = torch.exp(-last_event_seconds[..., None] / self.output_seconds)
        numerator = messages[..., : self.memory_dimension]
        denominator = messages[..., -1:] + initial_decay
        ratio = numerator / torch.clamp(denominator, min=1e-8)
        recent = self.embedding_transform(torch.cat([ratio, numerator], dim=-1))
        last_update = torch.where(
            active.squeeze(-1), last_event_seconds, torch.zeros_like(last_event_seconds)
        )
        recent = recent * torch.exp(
            (last_update[..., None] - self.window_seconds) / self.output_seconds
        )
        static = self.static_road_embedding.weight[None, :, :].expand(
            messages.shape[0], -1, -1
        )
        return self.e_lamb * static + (1.0 - self.e_lamb) * recent


class WindowBootCF(nn.Module):
    """BootCF dual encoders applied independently to each strict 12-hour window."""

    def __init__(
        self,
        n_nodes: int = 263,
        memory_dimension: int = 128,
        message_dimension: int = 128,
        output_seconds: float = 3600.0,
        window_seconds: float = 12.0 * 3600.0,
    ) -> None:
        super().__init__()
        self.encoder_o = WindowCTDGEncoder(
            n_nodes,
            memory_dimension,
            message_dimension,
            output_seconds,
            window_seconds=window_seconds,
        )
        self.encoder_d = WindowCTDGEncoder(
            n_nodes,
            memory_dimension,
            message_dimension,
            output_seconds,
            window_seconds=window_seconds,
        )
        # Retain the linear layer released in BootCF, although the released
        # forward returns concatenated origin/destination representations.
        self.linear = nn.Linear(memory_dimension * 2, memory_dimension)

    def forward(
        self,
        forward_weights: torch.Tensor,
        reverse_weights: torch.Tensor,
        origin_last_seconds: torch.Tensor,
        destination_last_seconds: torch.Tensor,
    ) -> torch.Tensor:
        from_o = self.encoder_o.aggregate_messages(forward_weights)
        from_d = self.encoder_d.aggregate_messages(reverse_weights)
        # Match the released BootCF cross-update convention.
        origin_embedding = self.encoder_o.embeddings_from_messages(
            from_d, destination_last_seconds
        )
        destination_embedding = self.encoder_d.embeddings_from_messages(
            from_o, origin_last_seconds
        )
        return torch.cat([origin_embedding, destination_embedding], dim=-1)


class PairMagnitudeMLP(nn.Module):
    """Memory-bounded form of BootCF's released OD pair decoder."""

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.w = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension // 2, 1),
        )

    def chunk(self, embeddings: torch.Tensor, start: int, end: int) -> torch.Tensor:
        batch, nodes, features = embeddings.shape
        width = end - start
        origin = embeddings[:, :, None, :].expand(batch, nodes, width, features)
        destination = embeddings[:, None, start:end, :].expand(
            batch, nodes, width, features
        )
        return self.w(torch.cat([origin, destination], dim=-1)).squeeze(-1)


def make_target_encoder(online: WindowBootCF) -> WindowBootCF:
    target = copy.deepcopy(online)
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


__all__ = [
    "MLP_Predictor",
    "PAPER_CODE",
    "PairMagnitudeMLP",
    "WindowBootCF",
    "make_target_encoder",
]
