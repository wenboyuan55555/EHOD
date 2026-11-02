from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


FEATURE_DIM = 23
HIERARCHICAL_CONTINUOUS_DIM = 18
CONTINUOUS_MODES = (
    "exact",
    "count_only",
    "quantized_15min",
    "hierarchical",
    "shuffled_time",
    "continuous_only",
)
FEATURE_ABLATIONS = (
    "none",
    "no_pair_history",
    "no_marginal_history",
    "no_statistical_priors",
    "no_spatial_relations",
    "pruned_redundant",
    "pruned_pair_marginal_stats",
    "no_continuous_timing",
    "no_subhour_state",
    "no_short_order_state",
    "no_stats_no_pair_history",
    "no_stats_no_marginal_history",
    "no_stats_no_spatial_relations",
    "no_stats_no_continuous_timing",
    "no_stats_no_subhour_state",
    "no_stats_no_short_order_state",
)
CONTEXT_ABLATIONS = ("none", "no_od_identity", "no_calendar")

# Indices refer to the documented 23-dimensional Count-Feature interface.
# A group is masked after construction so ablations retain identical model
# capacity and differ only in which information is observable.
FEATURE_ABLATION_INDICES = {
    "none": (),
    "no_pair_history": tuple(range(0, 10)),
    "no_marginal_history": tuple(range(10, 16)),
    "no_statistical_priors": tuple(range(16, 20)),
    "no_spatial_relations": tuple(range(20, 23)),
    # Keep pair history, exact recency and spatial relations.  This controlled
    # candidate tests whether hourly marginals remain useful once adaptive
    # short-term state is present; it is not the shared final profile.
    "pruned_redundant": tuple(range(10, 14)) + tuple(range(16, 20)),
    # Retain only continuous intensity/recency, spatial relations and (when
    # enabled) the adaptive short-term state.  Dimension 9 is also removed
    # because it normalizes intensity by the hourly pair-count sum.
    "pruned_pair_marginal_stats": (
        tuple(range(0, 8))
        + (9,)
        + tuple(range(10, 14))
        + tuple(range(16, 20))
    ),
    # Same-capacity short-order-state controls for hierarchical mode.
    "no_continuous_timing": (8, 9, 14, 15),
    "no_subhour_state": tuple(range(FEATURE_DIM, FEATURE_DIM + HIERARCHICAL_CONTINUOUS_DIM)),
    "no_short_order_state": (
        (8, 9, 14, 15)
        + tuple(range(FEATURE_DIM, FEATURE_DIM + HIERARCHICAL_CONTINUOUS_DIM))
    ),
    # Fair controls for the final profile, whose reference setting already
    # masks the four statistical-prior slots (16:20).  These unions prevent a
    # component ablation from accidentally restoring those priors.
    "no_stats_no_pair_history": tuple(range(0, 10)) + tuple(range(16, 20)),
    "no_stats_no_marginal_history": tuple(range(10, 20)),
    "no_stats_no_spatial_relations": tuple(range(16, 23)),
    "no_stats_no_continuous_timing": (8, 9, 14, 15) + tuple(range(16, 20)),
    "no_stats_no_subhour_state": (
        tuple(range(16, 20))
        + tuple(range(FEATURE_DIM, FEATURE_DIM + HIERARCHICAL_CONTINUOUS_DIM))
    ),
    "no_stats_no_short_order_state": (
        (8, 9, 14, 15)
        + tuple(range(16, 20))
        + tuple(range(FEATURE_DIM, FEATURE_DIM + HIERARCHICAL_CONTINUOUS_DIM))
    ),
}


def feature_dimension(continuous_mode: str) -> int:
    if continuous_mode not in CONTINUOUS_MODES:
        raise ValueError(f"Unknown continuous mode: {continuous_mode}")
    return FEATURE_DIM + (
        HIERARCHICAL_CONTINUOUS_DIM if continuous_mode == "hierarchical" else 0
    )


class ContinuousHybridOD(nn.Module):
    """Causal OD predictor with resolution-adaptive short-term order state.

    The short-term branch uses exact-time intensity/recency when timestamps
    permit it and observable sub-hour bins otherwise.  It predicts activation
    and positive magnitude jointly, but exposes the two heads so that the
    activation threshold remains validation-calibrated.
    """

    def __init__(
        self,
        nodes: int = 263,
        embedding_dim: int = 16,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        zero_init_residual: bool = True,
        feature_dim: int = FEATURE_DIM,
        context_ablation: str = "none",
    ) -> None:
        super().__init__()
        if context_ablation not in CONTEXT_ABLATIONS:
            raise ValueError(f"Unknown context ablation: {context_ablation}")
        self.feature_dim = int(feature_dim)
        self.context_ablation = context_ablation
        self.origin_embedding = nn.Embedding(nodes, embedding_dim)
        self.destination_embedding = nn.Embedding(nodes, embedding_dim)
        self.calendar = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.scalar = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        context_dim = hidden_dim + 3 * embedding_dim
        self.interaction = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.activation_head = nn.Linear(hidden_dim, 1)
        self.magnitude_residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if zero_init_residual:
            nn.init.zeros_(self.magnitude_residual[-1].weight)
            nn.init.zeros_(self.magnitude_residual[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        calendar: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if calendar.ndim == 2:
            calendar = calendar[:, None, :].expand(
                features.shape[0], features.shape[1], -1
            )
        hidden = self.scalar(features)
        origin_context = self.origin_embedding(origin)
        destination_context = self.destination_embedding(destination)
        calendar_context = self.calendar(calendar)
        if self.context_ablation == "no_od_identity":
            origin_context = torch.zeros_like(origin_context)
            destination_context = torch.zeros_like(destination_context)
        elif self.context_ablation == "no_calendar":
            calendar_context = torch.zeros_like(calendar_context)
        context = torch.cat(
            [
                hidden,
                origin_context,
                destination_context,
                calendar_context,
            ],
            dim=-1,
        )
        hidden = self.interaction(context)
        activation_logit = self.activation_head(hidden).squeeze(-1)
        residual = self.magnitude_residual(hidden).squeeze(-1)
        magnitude_log = torch.clamp(base_log_magnitude + residual, min=0.0)
        return activation_logit, magnitude_log


def hybrid_features(
    edge_history: torch.Tensor,
    origin_history: torch.Tensor,
    destination_history: torch.Tensor,
    continuous_intensity: torch.Tensor,
    origin_recency: torch.Tensor,
    destination_recency: torch.Tensor,
    starts: torch.Tensor,
    origin: torch.Tensor,
    destination: torch.Tensor,
    static: Dict[str, torch.Tensor],
    continuous_mode: str = "exact",
    feature_ablation: str = "none",
    quarter_edge: Optional[torch.Tensor] = None,
    quarter_origin: Optional[torch.Tensor] = None,
    quarter_destination: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if continuous_mode not in CONTINUOUS_MODES:
        raise ValueError(f"Unknown continuous mode: {continuous_mode}")
    if feature_ablation not in FEATURE_ABLATIONS:
        raise ValueError(f"Unknown feature ablation: {feature_ablation}")
    log_edge = torch.log1p(edge_history)
    edge_nonzero = edge_history > 0
    positive_count = edge_nonzero.sum(dim=-1).clamp_min(1)
    history_positive_mean = edge_history.sum(dim=-1) / positive_count
    last = log_edge[..., -1]
    first = log_edge[..., 0]
    edge_sum = edge_history.sum(dim=-1)
    slot = (starts % 168).long()
    seasonal_probability = static["seasonal_probability"][
        slot[:, None], origin, destination
    ].float()
    seasonal_positive_mean = static["seasonal_positive_mean"][
        slot[:, None], origin, destination
    ].float()
    global_positive_mean = static["edge_positive_mean"][origin, destination]
    base_magnitude = (
        0.45 * seasonal_positive_mean
        + 0.35 * history_positive_mean
        + 0.20 * global_positive_mean
    ).clamp_min(0.0)
    if continuous_mode == "count_only":
        modeled_intensity = torch.zeros_like(continuous_intensity)
        modeled_origin_recency = torch.zeros_like(origin_recency)
        modeled_destination_recency = torch.zeros_like(destination_recency)
    elif continuous_mode == "shuffled_time":
        decay = torch.exp(
            -torch.arange(
                edge_history.shape[-1] - 1,
                -1,
                -1,
                device=edge_history.device,
                dtype=edge_history.dtype,
            )
        )
        modeled_intensity = (
            edge_history * decay.view(*([1] * (edge_history.ndim - 1)), -1)
        ).sum(dim=-1) * (1.0 - math.exp(-1.0))

        def randomized_recency(count_history):
            active = count_history > 0
            positions = torch.arange(
                count_history.shape[-1],
                device=count_history.device,
                dtype=count_history.dtype,
            )
            latest = torch.where(
                active,
                positions.view(*([1] * (count_history.ndim - 1)), -1),
                torch.full_like(count_history, -1.0),
            ).max(dim=-1).values
            gather_index = latest.clamp_min(0).long().unsqueeze(-1)
            latest_count = count_history.gather(-1, gather_index).squeeze(-1)
            age_to_hour_start = count_history.shape[-1] - latest
            expected_fraction = latest_count / (latest_count + 1.0)
            value = torch.exp(-age_to_hour_start + expected_fraction)
            return torch.where(latest >= 0, value, torch.zeros_like(value))

        modeled_origin_recency = randomized_recency(origin_history)
        modeled_destination_recency = randomized_recency(destination_history)
    elif continuous_mode == "quantized_15min":
        # ContinuousFeatureStore has already replaced every pickup timestamp by
        # the center of its true 15-minute bin before computing these three
        # statistics.  No exact-time statistic is visible in this branch.
        modeled_intensity = continuous_intensity
        modeled_origin_recency = origin_recency
        modeled_destination_recency = destination_recency
    else:
        modeled_intensity = continuous_intensity
        modeled_origin_recency = origin_recency
        modeled_destination_recency = destination_recency

    features = torch.stack(
        [
            last,
            log_edge.mean(dim=-1),
            log_edge.max(dim=-1).values,
            log_edge.std(dim=-1, unbiased=False),
            edge_nonzero.float().mean(dim=-1),
            last - first,
            torch.log1p(edge_sum),
            torch.log1p(history_positive_mean),
            torch.log1p(modeled_intensity),
            torch.log1p(modeled_intensity) - torch.log1p(edge_sum),
            torch.log1p(origin_history[..., -1]),
            torch.log1p(origin_history.mean(dim=-1)),
            torch.log1p(destination_history[..., -1]),
            torch.log1p(destination_history.mean(dim=-1)),
            modeled_origin_recency,
            modeled_destination_recency,
            static["edge_nonzero_frequency"][origin, destination],
            torch.log1p(global_positive_mean),
            seasonal_probability,
            torch.log1p(seasonal_positive_mean),
            static["distance"][origin, destination],
            static["same_borough"][origin, destination],
            static["affinity"][origin, destination],
        ],
        dim=-1,
    )
    if continuous_mode == "hierarchical":
        if (
            quarter_edge is None
            or quarter_origin is None
            or quarter_destination is None
        ):
            raise ValueError(
                "hierarchical mode requires quarter_edge, quarter_origin and "
                "quarter_destination"
            )
        if not (
            quarter_edge.shape == quarter_origin.shape == quarter_destination.shape
            and quarter_edge.shape[:-1] == features.shape[:-1]
            and quarter_edge.shape[-1] == 4
        ):
            raise ValueError(
                "quarter-hour inputs must all have shape (*features, 4); got "
                f"{quarter_edge.shape}, {quarter_origin.shape}, "
                f"{quarter_destination.shape}"
            )
        edge_total = quarter_edge.sum(dim=-1, keepdim=True)
        edge_share = quarter_edge / (edge_total + 1.0)
        edge_trend = (quarter_edge[..., 3] - quarter_edge[..., 0]) / torch.sqrt(
            edge_total.squeeze(-1) + 1.0
        )
        edge_burst = (
            3.0 * quarter_edge[..., 3] - quarter_edge[..., :3].sum(dim=-1)
        ) / (edge_total.squeeze(-1) + 1.0)
        hierarchical = torch.cat(
            [
                torch.log1p(quarter_edge),
                edge_share,
                torch.log1p(quarter_origin),
                torch.log1p(quarter_destination),
                edge_trend.unsqueeze(-1),
                edge_burst.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = torch.cat([features, hierarchical], dim=-1)
    if continuous_mode == "continuous_only":
        keep = torch.zeros(FEATURE_DIM, dtype=features.dtype, device=features.device)
        keep[[8, 14, 15, 20, 21, 22]] = 1.0
        features = features * keep
        base_magnitude = torch.zeros_like(base_magnitude)
    removed = FEATURE_ABLATION_INDICES[feature_ablation]
    if removed:
        keep = torch.ones(
            features.shape[-1], dtype=features.dtype, device=features.device
        )
        keep[list(removed)] = 0.0
        features = features * keep
    return features, torch.log1p(base_magnitude)


__all__ = [
    "CONTINUOUS_MODES",
    "CONTEXT_ABLATIONS",
    "FEATURE_ABLATIONS",
    "FEATURE_DIM",
    "HIERARCHICAL_CONTINUOUS_DIM",
    "ContinuousHybridOD",
    "feature_dimension",
    "hybrid_features",
]
