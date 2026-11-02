"""Strict variable-hour window adapter around the original CMOD architecture."""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


from ehod.common.upstream import load_snapshot
BASE_CMOD = Path(__file__).resolve().parent / "upstream"
_upstream = load_snapshot(BASE_CMOD, 'model/CMOD.py', "_ehod_upstream_cmod", aliases=('modules',))
NodeMemory, PredictionLayer = _upstream.NodeMemory, _upstream.PredictionLayer


class WindowedNodeMemory(NodeMemory):
    """Use CMOD's exact exponential aggregator from pre-aggregated event sums.

    ``weight_matrix[i, j]`` equals the sum of
    ``exp((event_time - last_event_time_of_origin_i) / lambda)`` for all i->j
    trips in the strict input window.  It is algebraically the
    same tensor produced by the original Python event loop.
    """

    def reset_window_memory(self, begin_time: float = 0.0) -> None:
        with torch.no_grad():
            self.memory.memory.data.zero_()
            self.memory.memory.data[:, :, -1].fill_(1.0)
            self.memory.last_update.data.fill_(float(begin_time))
            self.region_memory.memory.data.zero_()
            self.region_memory.last_update.data.fill_(float(begin_time))
            self.graph_memory.memory.data.zero_()
            self.graph_memory.last_update.data.fill_(float(begin_time))

    def forward_summary(
        self,
        weight_matrix: torch.Tensor,
        last_event_times: torch.Tensor,
        now_time: float,
        begin_time: float,
    ) -> torch.Tensor:
        row_weight = weight_matrix.sum(dim=1)
        unique_nodes = torch.nonzero(row_weight > 0, as_tuple=False).flatten()
        if unique_nodes.numel() == 0:
            raise ValueError("The input window contains no trips")

        memory = self.memory.get_memory()
        all_nodes = torch.arange(self.n_nodes, device=weight_matrix.device)
        target_embeddings = self.exp_embedding.compute_embedding(
            memory=memory, nodes=all_nodes
        )
        weighted_target_embeddings = torch.mm(weight_matrix, target_embeddings)
        raw_aggregated = torch.cat(
            [weighted_target_embeddings, weight_matrix, row_weight[:, None]], dim=1
        )
        unique_raw_messages = raw_aggregated[unique_nodes, None, :]
        unique_timestamps = last_event_times[unique_nodes]
        unique_messages = torch.cat(
            [
                self.message_function.compute_message(unique_raw_messages[:, :, :-1]),
                unique_raw_messages[:, :, -1:],
            ],
            dim=-1,
        )

        region_memory = self.region_memory.get_memory()
        graph_memory = self.graph_memory.get_memory()
        time_diffs = -self.memory.last_update + float(begin_time)
        static_node_embedding = self.exp_embedding.compute_embedding(
            memory=memory, nodes=all_nodes, time_diffs=time_diffs
        )
        region_ids = torch.arange(self.n_regions, device=weight_matrix.device)
        graph_ids = torch.arange(self.n_graphs, device=weight_matrix.device)
        region_embedding = self.iden_embedding.compute_embedding(
            memory=region_memory, nodes=region_ids, time_diffs=time_diffs
        ).reshape(self.n_regions, -1)
        graph_embedding = self.iden_embedding.compute_embedding(
            memory=graph_memory, nodes=graph_ids, time_diffs=time_diffs
        ).reshape(self.n_graphs, -1)

        A_r = torch.einsum(
            "rhf,nhf->rhn",
            self.Q_r(region_embedding).reshape(
                self.n_regions, self.n_heads, self.features_per_head
            ),
            self.K_r(static_node_embedding).reshape(
                self.n_nodes, self.n_heads, self.features_per_head
            ),
        ) / math.sqrt(self.features_per_head)
        A_g = torch.einsum(
            "ghf,rhf->ghr",
            self.Q_g(graph_embedding).reshape(
                self.n_graphs, self.n_heads, self.features_per_head
            ),
            self.K_g(region_embedding).reshape(
                self.n_regions, self.n_heads, self.features_per_head
            ),
        ) / math.sqrt(self.features_per_head)

        normalized_raw = (
            unique_raw_messages[:, :, :-1] / unique_raw_messages[:, :, -1:]
        )
        region_messages_mid = torch.einsum(
            "rhn,nlhf->rlhf",
            torch.softmax(A_r[:, :, unique_nodes], dim=2),
            self.V_r(normalized_raw.reshape(len(unique_nodes), self.lamb_len, -1)).reshape(
                len(unique_nodes),
                self.lamb_len,
                self.n_heads,
                self.message_per_head,
            ),
        ).reshape(self.n_regions, self.lamb_len, self.message_multi_head)
        region_messages = self.ff_r(region_messages_mid)
        graph_messages_mid = torch.einsum(
            "ghr,rlhf->glhf",
            torch.softmax(A_g, dim=2),
            self.V_g(region_messages_mid).reshape(
                self.n_regions,
                self.lamb_len,
                self.n_heads,
                self.message_per_head,
            ),
        )
        graph_messages = self.ff_g(
            graph_messages_mid.reshape(
                self.n_graphs, self.lamb_len, self.message_multi_head
            )
        )

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
        updated_time_diffs = -updated_last_update + float(now_time)
        recent_node_embeddings = self.exp_embedding.compute_embedding(
            memory=updated_memory, nodes=all_nodes, time_diffs=updated_time_diffs
        )
        updated_region_memory, _ = self.region_memory_updater.get_updated_memory(
            region_ids, region_messages, timestamps=float(now_time)
        )
        updated_graph_memory, _ = self.graph_memory_updater.get_updated_memory(
            graph_ids, graph_messages, timestamps=float(now_time)
        )
        recent_region_embeddings = self.iden_embedding.compute_embedding(
            memory=updated_region_memory, nodes=region_ids
        ).reshape(self.n_regions, -1)
        recent_graph_embeddings = self.iden_embedding.compute_embedding(
            memory=updated_graph_memory, nodes=graph_ids
        ).reshape(self.n_graphs, -1)

        r2n = torch.mean(torch.softmax(A_r, dim=0), dim=1)
        g2r = torch.mean(torch.softmax(A_g, dim=0), dim=1)
        region_node_embeddings = torch.mm(r2n.T, recent_region_embeddings)
        graph_node_embeddings = torch.mm(
            torch.mm(r2n.T, g2r.T), recent_graph_embeddings
        )
        dynamic_embeddings = torch.cat(
            [recent_node_embeddings, region_node_embeddings, graph_node_embeddings], dim=0
        )
        embeddings = self.lamb * self.static_embedding.weight + (
            1 - self.lamb
        ) * self.spatial_transform(
            self.embedding_transform(dynamic_embeddings)
            .reshape(3, self.n_nodes, self.memory_dimension)
            .permute(1, 0, 2)
            .reshape(self.n_nodes, -1)
        )
        return embeddings


class ContinuousWindowCMOD(nn.Module):
    def __init__(
        self,
        device: str,
        n_nodes: int = 263,
        message_dimension: int = 128,
        memory_dimension: int = 128,
        lambda_seconds: float = 3600.0,
        window_seconds: float = 12.0 * 3600.0,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.window_seconds = float(window_seconds)
        node_features = np.eye(n_nodes, dtype=np.float32)
        node_raw_features = torch.from_numpy(node_features).to(device)
        self.memory = WindowedNodeMemory(
            n_nodes=n_nodes,
            node_features=node_raw_features,
            embedding_dimension=memory_dimension,
            memory_dimension=memory_dimension,
            message_dimension=message_dimension,
            lambs=[1.0],
            device=device,
            output=lambda_seconds,
            init_lamb=0.1,
        )
        self.predict_od = PredictionLayer(memory_dimension, n_nodes)

    def encode_window(
        self, weight_matrix: torch.Tensor, last_event_times: torch.Tensor
    ) -> torch.Tensor:
        # Relative times preserve one-second precision in float32 throughout
        # the year.  Each independent sample is shifted to [0, window_seconds).
        self.memory.reset_window_memory(begin_time=0.0)
        return self.memory.forward_summary(
            weight_matrix=weight_matrix,
            last_event_times=last_event_times,
            now_time=self.window_seconds,
            begin_time=0.0,
        )

    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode one or more independent node-embedding matrices to full OD."""
        if embeddings.ndim == 2:
            return self.predict_od(embeddings)
        if embeddings.ndim != 3:
            raise ValueError(f"Expected [N,D] or [B,N,D], got {embeddings.shape}")
        batch, nodes, dimension = embeddings.shape
        if nodes != self.predict_od.n_nodes:
            raise ValueError(f"Expected {self.predict_od.n_nodes} nodes, got {nodes}")
        source = embeddings[:, :, None, :].expand(batch, nodes, nodes, dimension)
        destination = embeddings[:, None, :, :].expand(
            batch, nodes, nodes, dimension
        )
        pair = torch.cat([source, destination], dim=-1)
        return self.predict_od.w(pair).squeeze(-1)

    def forward(
        self, weight_matrix: torch.Tensor, last_event_times: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.encode_window(weight_matrix, last_event_times)
        return self.decode_embeddings(embeddings)
