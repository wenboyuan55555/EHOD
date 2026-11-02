from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ehod.data.od import ACTIVE_CITY, N_NODES


N_ROLES = 2
N_PHASES = 5
N_EVENT_TYPES = 8 if ACTIVE_CITY == "chicago" else 7
N_VENUES = 10 if ACTIVE_CITY == "chicago" else 8
STRUCTURED_DIM = N_ROLES * N_PHASES * N_EVENT_TYPES
VENUE_ROLE_DIM = N_ROLES * N_VENUES
TIME_ROLE_DIM = 4
EVENT_FEATURE_DIM = STRUCTURED_DIM + VENUE_ROLE_DIM + TIME_ROLE_DIM


class FactorizedEventPotential(nn.Module):
    """Encode a planned event as a low-rank role-stage transport potential.

    The 90-dimensional sparse input is not treated as an unstructured feature
    vector.  Its first block is a role x phase x event-type tensor and its
    second block is a role x venue tensor.  Low-rank factors turn those two
    tensors into a single event potential, while the four schedule coordinates
    retain continuous offsets to the published start and end times.
    """

    def __init__(
        self,
        rank: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.role_factor = nn.Parameter(torch.empty(N_ROLES, rank))
        self.phase_factor = nn.Parameter(torch.empty(N_PHASES, rank))
        self.type_factor = nn.Parameter(torch.empty(N_EVENT_TYPES, rank))
        self.venue_factor = nn.Parameter(torch.empty(N_VENUES, rank))
        self.time_projection = nn.Linear(TIME_ROLE_DIM, rank, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.role_factor,
            self.phase_factor,
            self.type_factor,
            self.venue_factor,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.25)
        nn.init.xavier_uniform_(self.time_projection.weight)

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        if event_features.shape[-1] != EVENT_FEATURE_DIM:
            raise ValueError(
                "Expected %d event features, received %d"
                % (EVENT_FEATURE_DIM, event_features.shape[-1])
            )
        typed = event_features[:, :STRUCTURED_DIM].reshape(
            -1, N_ROLES, N_PHASES, N_EVENT_TYPES
        )
        venue = event_features[
            :, STRUCTURED_DIM : STRUCTURED_DIM + VENUE_ROLE_DIM
        ].reshape(-1, N_ROLES, N_VENUES)
        schedule = event_features[:, -TIME_ROLE_DIM:]

        typed_basis = (
            self.role_factor[:, None, None, :]
            * self.phase_factor[None, :, None, :]
            * self.type_factor[None, None, :, :]
        )
        venue_basis = self.role_factor[:, None, :] * self.venue_factor[None, :, :]
        typed_potential = torch.einsum("brpt,rptk->bk", typed, typed_basis)
        venue_potential = torch.einsum("brv,rvk->bk", venue, venue_basis)
        schedule_potential = self.time_projection(schedule)
        return self.output(
            typed_potential + venue_potential + schedule_potential
        )


class EventAnchoredTransportAdapter(nn.Module):
    """Planned-event Hurdle intervention with no gate, Context, GRU, or peak head.

    A factorized event potential describes the scheduled intervention.  A
    separate OD susceptibility state is constructed only from the frozen
    Hurdle parameters and the identities of the two endpoint zones.  Their
    elementwise interaction is the event-on-OD mechanism.  Two bounded heads
    then update activation and positive magnitude directly.
    """

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 16,
        event_rank: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        activation_cap: float = 2.0,
        magnitude_cap: float = 1.25,
    ) -> None:
        super().__init__()
        self.activation_cap = float(activation_cap)
        self.magnitude_cap = float(magnitude_cap)
        self.event_potential = FactorizedEventPotential(
            rank=event_rank,
            hidden=hidden,
            dropout=dropout,
        )
        self.origin = nn.Embedding(nodes, node_dim)
        self.destination = nn.Embedding(nodes, node_dim)
        self.susceptibility = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.intervention = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.activation_delta = nn.Linear(hidden, 1)
        self.magnitude_delta = nn.Linear(hidden, 1)
        nn.init.zeros_(self.activation_delta.weight)
        nn.init.zeros_(self.activation_delta.bias)
        nn.init.zeros_(self.magnitude_delta.weight)
        nn.init.zeros_(self.magnitude_delta.bias)

    def forward(
        self,
        event_features: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_logit: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event = self.event_potential(event_features)
        traffic = self.susceptibility(
            torch.cat(
                [
                    base_logit[:, None],
                    base_log_magnitude[:, None],
                    self.origin(origin),
                    self.destination(destination),
                ],
                dim=-1,
            )
        )
        interaction = self.intervention(
            torch.cat([event, traffic, event * traffic], dim=-1)
        )
        activation_delta = self.activation_cap * torch.tanh(
            self.activation_delta(interaction).squeeze(-1)
        )
        magnitude_delta = self.magnitude_cap * torch.tanh(
            self.magnitude_delta(interaction).squeeze(-1)
        )
        return (
            base_logit + activation_delta,
            torch.clamp(base_log_magnitude + magnitude_delta, min=0.0),
            activation_delta,
            magnitude_delta,
        )


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(float(value))))


class DualAnchorEventLifecycle(nn.Module):
    """Robust continuous taxi response around event start and end.

    The rational-quadratic bell keeps the same two physical anchors as a
    Gaussian lifecycle, while retaining a small tail for heterogeneous arrival
    and departure times across passengers and event durations.
    """

    def __init__(
        self,
        arrival_lead_hours: float = 1.0,
        arrival_width_hours: float = 1.25,
        departure_lag_hours: float = 0.5,
        departure_width_hours: float = 1.50,
        anchor_mode: str = "dual",
        kernel_mode: str = "cauchy",
    ) -> None:
        super().__init__()
        if anchor_mode not in {"dual", "start_only"}:
            raise ValueError("Unknown anchor mode %r" % anchor_mode)
        if kernel_mode not in {"cauchy", "gaussian"}:
            raise ValueError("Unknown kernel mode %r" % kernel_mode)
        self.anchor_mode = anchor_mode
        self.kernel_mode = kernel_mode
        self.raw_arrival_lead = nn.Parameter(
            torch.tensor(inverse_softplus(arrival_lead_hours), dtype=torch.float32)
        )
        self.raw_arrival_width = nn.Parameter(
            torch.tensor(inverse_softplus(arrival_width_hours), dtype=torch.float32)
        )
        self.raw_departure_lag = nn.Parameter(
            torch.tensor(inverse_softplus(departure_lag_hours), dtype=torch.float32)
        )
        self.raw_departure_width = nn.Parameter(
            torch.tensor(inverse_softplus(departure_width_hours), dtype=torch.float32)
        )

    def parameters_dict(self) -> Dict[str, float]:
        return {
            "arrival_peak_hours_from_start": -float(
                F.softplus(self.raw_arrival_lead).detach().cpu()
            ),
            "arrival_width_hours": float(
                F.softplus(self.raw_arrival_width).detach().cpu()
            ),
            "departure_peak_hours_from_end": float(
                F.softplus(self.raw_departure_lag).detach().cpu()
            ),
            "departure_width_hours": float(
                F.softplus(self.raw_departure_width).detach().cpu()
            ),
            "kernel": self.kernel_mode,
            "anchor_mode": self.anchor_mode,
        }

    def forward(
        self,
        tau_start: torch.Tensor,
        tau_end: torch.Tensor,
        event_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del event_type
        arrival_center = -F.softplus(self.raw_arrival_lead)
        arrival_width = F.softplus(self.raw_arrival_width).clamp_min(0.25)
        departure_center = F.softplus(self.raw_departure_lag)
        departure_width = F.softplus(self.raw_departure_width).clamp_min(0.25)
        arrival_offset = (tau_start - arrival_center) / arrival_width
        departure_reference = tau_end if self.anchor_mode == "dual" else tau_start
        departure_offset = (departure_reference - departure_center) / departure_width
        if self.kernel_mode == "gaussian":
            arrival = torch.exp(-0.5 * arrival_offset.square())
            departure = torch.exp(-0.5 * departure_offset.square())
        else:
            arrival = 1.0 / (1.0 + arrival_offset.square())
            departure = 1.0 / (1.0 + departure_offset.square())
        return arrival, departure


class TaxiEventExposurePotential(nn.Module):
    """Venue-centered exposure field with an asymmetric taxi lifecycle."""

    def __init__(
        self,
        rank: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        arrival_length_km: float = 0.75,
        departure_length_km: float = 1.25,
        anchor_mode: str = "dual",
        kernel_mode: str = "cauchy",
        exposure_mode: str = "directional",
        type_mode: str = "id",
        venue_mode: str = "id",
        event_types: int = N_EVENT_TYPES,
        venues: int = N_VENUES,
    ) -> None:
        super().__init__()
        if exposure_mode not in {"directional", "symmetric"}:
            raise ValueError("Unknown exposure mode %r" % exposure_mode)
        if type_mode not in {"id", "shared"}:
            raise ValueError("Unknown event-type mode %r" % type_mode)
        if venue_mode not in {"id", "shared"}:
            raise ValueError("Unknown venue mode %r" % venue_mode)
        self.anchor_mode = anchor_mode
        self.exposure_mode = exposure_mode
        self.type_mode = type_mode
        self.venue_mode = venue_mode
        self.lifecycle = DualAnchorEventLifecycle(
            anchor_mode=anchor_mode, kernel_mode=kernel_mode
        )
        self.role_factor = nn.Parameter(torch.empty(N_ROLES, rank))
        self.event_types = int(event_types)
        self.venues = int(venues)
        self.type_factor = nn.Embedding(self.event_types, rank)
        self.venue_factor = nn.Embedding(self.venues, rank)
        self.arrival_shape = nn.Parameter(torch.empty(rank))
        self.departure_shape = nn.Parameter(torch.empty(rank))
        self.time_projection = nn.Linear(2, rank, bias=False)
        self.raw_arrival_length = nn.Parameter(
            torch.tensor(inverse_softplus(arrival_length_km), dtype=torch.float32)
        )
        self.raw_departure_length = nn.Parameter(
            torch.tensor(inverse_softplus(departure_length_km), dtype=torch.float32)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.role_factor, mean=0.0, std=0.25)
        nn.init.normal_(self.type_factor.weight, mean=0.0, std=0.25)
        nn.init.normal_(self.venue_factor.weight, mean=0.0, std=0.25)
        nn.init.normal_(self.arrival_shape, mean=1.0, std=0.10)
        nn.init.normal_(self.departure_shape, mean=1.0, std=0.10)
        nn.init.xavier_uniform_(self.time_projection.weight)

    def parameters_dict(self) -> Dict[str, float]:
        return {
            **self.lifecycle.parameters_dict(),
            "exposure_mode": self.exposure_mode,
            "type_mode": self.type_mode,
            "venue_mode": self.venue_mode,
            "arrival_spatial_length_km": float(
                F.softplus(self.raw_arrival_length).detach().cpu()
            ),
            "departure_spatial_length_km": float(
                F.softplus(self.raw_departure_length).detach().cpu()
            ),
        }

    def forward(
        self,
        event_type: torch.Tensor,
        venue: torch.Tensor,
        role: torch.Tensor,
        scale: torch.Tensor,
        tau_start: torch.Tensor,
        tau_end: torch.Tensor,
        arrival_distance_km: torch.Tensor,
        departure_distance_km: torch.Tensor,
    ) -> torch.Tensor:
        mask = event_type >= 0
        safe_type = event_type.clamp_min(0)
        safe_venue = venue.clamp_min(0)
        arrival_kernel, departure_kernel = self.lifecycle(
            tau_start, tau_end, safe_type
        )
        arrival_length = F.softplus(self.raw_arrival_length).clamp_min(0.10)
        departure_length = F.softplus(self.raw_departure_length).clamp_min(0.10)
        if self.exposure_mode == "directional":
            arrival_exposure = torch.exp(-arrival_distance_km / arrival_length)
            departure_exposure = torch.exp(-departure_distance_km / departure_length)
        else:
            arrival_exposure = 0.5 * (
                torch.exp(-arrival_distance_km / arrival_length)
                + torch.exp(-departure_distance_km / arrival_length)
            )
            departure_exposure = 0.5 * (
                torch.exp(-arrival_distance_km / departure_length)
                + torch.exp(-departure_distance_km / departure_length)
            )
        arrival_role = ((role == 0).float() + 0.5 * (role == 2).float())
        departure_role = ((role == 1).float() + 0.5 * (role == 2).float())
        role_state = (
            arrival_role.unsqueeze(-1) * self.role_factor[0]
            + departure_role.unsqueeze(-1) * self.role_factor[1]
        )
        if self.type_mode == "id":
            type_state = self.type_factor(safe_type)
        else:
            type_state = self.type_factor.weight.mean(dim=0).view(1, 1, -1)
        if self.venue_mode == "id":
            venue_state = self.venue_factor(safe_venue)
        else:
            venue_state = self.venue_factor.weight.mean(dim=0).view(1, 1, -1)
        semantics = role_state * (type_state + venue_state)
        arrival_intensity = (
            scale * arrival_role * arrival_kernel * arrival_exposure * mask.float()
        )
        departure_intensity = (
            scale
            * departure_role
            * departure_kernel
            * departure_exposure
            * mask.float()
        )
        second_anchor = tau_end if self.anchor_mode == "dual" else tau_start
        anchor_coordinates = torch.stack(
            [
                tau_start.clamp(-12.0, 12.0) / 6.0,
                second_anchor.clamp(-12.0, 12.0) / 6.0,
            ],
            dim=-1,
        )
        continuous_shape = (
            arrival_intensity.unsqueeze(-1) * self.arrival_shape
            + departure_intensity.unsqueeze(-1) * self.departure_shape
            + (arrival_intensity + departure_intensity).unsqueeze(-1)
            * torch.tanh(self.time_projection(anchor_coordinates))
        )
        potential = (semantics * continuous_shape).sum(dim=1)
        return self.output(potential)


class TaxiAwareEventTransitionAdapter(nn.Module):
    """Event-conditioned taxi-demand transition without phase bins or transport.

    Scheduled events create a continuous, venue-centered exposure field.  The
    frozen base Hurdle state determines how susceptible each OD pair is, and a
    zero-initialized residual performs the minimum activation/magnitude change.
    """

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 16,
        event_rank: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        initial_activation_scale: float = 2.0,
        initial_magnitude_scale: float = 1.25,
        anchor_mode: str = "dual",
        kernel_mode: str = "cauchy",
        exposure_mode: str = "directional",
        susceptibility_mode: str = "od_specific",
        head_mode: str = "both",
        base_state_mode: str = "full",
        type_mode: str = "id",
        venue_mode: str = "id",
        event_types: int = N_EVENT_TYPES,
        venues: int = N_VENUES,
    ) -> None:
        super().__init__()
        if susceptibility_mode not in {"od_specific", "base_only", "shared"}:
            raise ValueError(
                "Unknown susceptibility mode %r" % susceptibility_mode
            )
        if base_state_mode not in {"full", "no_base"}:
            raise ValueError("Unknown base-state mode %r" % base_state_mode)
        if head_mode not in {"both", "activation_only", "magnitude_only"}:
            raise ValueError("Unknown head mode %r" % head_mode)
        self.susceptibility_mode = susceptibility_mode
        self.base_state_mode = base_state_mode
        self.head_mode = head_mode
        self.event_exposure = TaxiEventExposurePotential(
            rank=event_rank,
            hidden=hidden,
            dropout=dropout,
            anchor_mode=anchor_mode,
            kernel_mode=kernel_mode,
            exposure_mode=exposure_mode,
            type_mode=type_mode,
            venue_mode=venue_mode,
            event_types=event_types,
            venues=venues,
        )
        self.node = nn.Embedding(nodes, node_dim)
        self.origin_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.destination_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.susceptibility = nn.Sequential(
            nn.Linear(2 + 2 * node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.response = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.activation_delta = nn.Linear(hidden, 1)
        self.magnitude_delta = nn.Linear(hidden, 1)
        self.raw_activation_scale = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_activation_scale), dtype=torch.float32
            )
        )
        self.raw_magnitude_scale = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_magnitude_scale), dtype=torch.float32
            )
        )
        nn.init.zeros_(self.activation_delta.weight)
        nn.init.zeros_(self.activation_delta.bias)
        nn.init.zeros_(self.magnitude_delta.weight)
        nn.init.zeros_(self.magnitude_delta.bias)

    def mechanism_parameters(self) -> Dict[str, float]:
        return {
            **self.event_exposure.parameters_dict(),
            "susceptibility_mode": self.susceptibility_mode,
            "base_state_mode": self.base_state_mode,
            "head_mode": self.head_mode,
            "activation_residual_scale": float(
                F.softplus(self.raw_activation_scale).detach().cpu()
            ),
            "magnitude_residual_scale": float(
                F.softplus(self.raw_magnitude_scale).detach().cpu()
            ),
        }

    def forward(
        self,
        event_type: torch.Tensor,
        venue: torch.Tensor,
        role: torch.Tensor,
        scale: torch.Tensor,
        tau_start: torch.Tensor,
        tau_end: torch.Tensor,
        arrival_distance_km: torch.Tensor,
        departure_distance_km: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_logit: torch.Tensor,
        base_log_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event = self.event_exposure(
            event_type,
            venue,
            role,
            scale,
            tau_start,
            tau_end,
            arrival_distance_km,
            departure_distance_km,
        )
        origin_state = self.origin_projection(self.node(origin))
        destination_state = self.destination_projection(self.node(destination))
        susceptibility_logit = base_logit
        susceptibility_log_magnitude = base_log_magnitude
        if self.base_state_mode == "no_base":
            susceptibility_logit = torch.zeros_like(base_logit)
            susceptibility_log_magnitude = torch.zeros_like(base_log_magnitude)
        if self.susceptibility_mode == "base_only":
            origin_state = torch.zeros_like(origin_state)
            destination_state = torch.zeros_like(destination_state)
        elif self.susceptibility_mode == "shared":
            susceptibility_logit = torch.zeros_like(base_logit)
            susceptibility_log_magnitude = torch.zeros_like(base_log_magnitude)
            origin_state = torch.zeros_like(origin_state)
            destination_state = torch.zeros_like(destination_state)
        traffic = self.susceptibility(
            torch.cat(
                [
                    susceptibility_logit[:, None],
                    susceptibility_log_magnitude[:, None],
                    origin_state,
                    destination_state,
                ],
                dim=-1,
            )
        )
        response = self.response(torch.cat([event, traffic, event * traffic], dim=-1))
        activation_scale = F.softplus(self.raw_activation_scale)
        magnitude_scale = F.softplus(self.raw_magnitude_scale)
        activation_delta = activation_scale * torch.tanh(
            self.activation_delta(response).squeeze(-1)
        )
        magnitude_delta = magnitude_scale * torch.tanh(
            self.magnitude_delta(response).squeeze(-1)
        )
        if self.head_mode == "activation_only":
            magnitude_delta = torch.zeros_like(magnitude_delta)
        elif self.head_mode == "magnitude_only":
            activation_delta = torch.zeros_like(activation_delta)
        return (
            base_logit + activation_delta,
            torch.clamp(base_log_magnitude + magnitude_delta, min=0.0),
            activation_delta,
            magnitude_delta,
        )


class DirectCountEventTransitionAdapter(nn.Module):
    """Single-stage count-regression control for the Hurdle EADT.

    The event lifecycle, venue exposure and OD susceptibility are unchanged.
    Unlike the formal model, this control has no activation classifier and no
    conditional positive-magnitude head: one bounded residual directly updates
    the final log-count prediction.
    """

    def __init__(
        self,
        nodes: int = N_NODES,
        node_dim: int = 16,
        event_rank: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        initial_count_scale: float = 1.25,
        anchor_mode: str = "dual",
        exposure_mode: str = "directional",
        susceptibility_mode: str = "od_specific",
        base_state_mode: str = "full",
        type_mode: str = "id",
        venue_mode: str = "id",
        event_types: int = N_EVENT_TYPES,
        venues: int = N_VENUES,
    ) -> None:
        super().__init__()
        if susceptibility_mode not in {"od_specific", "base_only", "shared"}:
            raise ValueError(
                "Unknown susceptibility mode %r" % susceptibility_mode
            )
        if base_state_mode not in {"full", "no_base"}:
            raise ValueError("Unknown base-state mode %r" % base_state_mode)
        self.susceptibility_mode = susceptibility_mode
        self.base_state_mode = base_state_mode
        self.event_exposure = TaxiEventExposurePotential(
            rank=event_rank,
            hidden=hidden,
            dropout=dropout,
            anchor_mode=anchor_mode,
            exposure_mode=exposure_mode,
            type_mode=type_mode,
            venue_mode=venue_mode,
            event_types=event_types,
            venues=venues,
        )
        self.node = nn.Embedding(nodes, node_dim)
        self.origin_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.destination_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.susceptibility = nn.Sequential(
            nn.Linear(1 + 2 * node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.response = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.count_delta = nn.Linear(hidden, 1)
        self.raw_count_scale = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_count_scale), dtype=torch.float32
            )
        )
        nn.init.zeros_(self.count_delta.weight)
        nn.init.zeros_(self.count_delta.bias)

    def mechanism_parameters(self) -> Dict[str, float]:
        return {
            **self.event_exposure.parameters_dict(),
            "susceptibility_mode": self.susceptibility_mode,
            "base_state_mode": self.base_state_mode,
            "count_residual_scale": float(
                F.softplus(self.raw_count_scale).detach().cpu()
            ),
            "prediction_mode": "single_stage_direct_count",
        }

    def forward(
        self,
        event_type: torch.Tensor,
        venue: torch.Tensor,
        role: torch.Tensor,
        scale: torch.Tensor,
        tau_start: torch.Tensor,
        tau_end: torch.Tensor,
        arrival_distance_km: torch.Tensor,
        departure_distance_km: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        base_count: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        event = self.event_exposure(
            event_type,
            venue,
            role,
            scale,
            tau_start,
            tau_end,
            arrival_distance_km,
            departure_distance_km,
        )
        base_log_count = torch.log1p(base_count.clamp_min(0.0))
        state_log_count = (
            torch.zeros_like(base_log_count)
            if self.base_state_mode == "no_base"
            else base_log_count
        )
        origin_state = self.origin_projection(self.node(origin))
        destination_state = self.destination_projection(self.node(destination))
        if self.susceptibility_mode == "base_only":
            origin_state = torch.zeros_like(origin_state)
            destination_state = torch.zeros_like(destination_state)
        elif self.susceptibility_mode == "shared":
            state_log_count = torch.zeros_like(state_log_count)
            origin_state = torch.zeros_like(origin_state)
            destination_state = torch.zeros_like(destination_state)
        traffic = self.susceptibility(
            torch.cat(
                [state_log_count[:, None], origin_state, destination_state],
                dim=-1,
            )
        )
        response = self.response(torch.cat([event, traffic, event * traffic], dim=-1))
        count_scale = F.softplus(self.raw_count_scale)
        count_delta = count_scale * torch.tanh(
            self.count_delta(response).squeeze(-1)
        )
        prediction = torch.expm1(
            torch.clamp(base_log_count + count_delta, min=0.0)
        )
        return prediction, count_delta


@dataclass(frozen=True)
class TransportElasticity:
    """Fraction of a direct event effect attributed to demand substitution."""

    arrival: float
    departure: float

    def validate(self) -> None:
        for name, value in (("arrival", self.arrival), ("departure", self.departure)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("%s elasticity must lie in [0, 1]" % name)


def marginal_transport_components(
    base_matrix: np.ndarray,
    direct_edges: np.ndarray,
    direct_deltas: np.ndarray,
    arrival_share: np.ndarray,
    departure_share: np.ndarray,
    node_similarity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build explicit arrival-row and departure-column demand transfers.

    Arrival effects compete inside an origin row: if an event increases
    ``o -> venue``, some of that flow may have moved from ``o -> other``.
    Departure effects compete inside a destination column: if an event
    increases ``venue -> d``, some may have moved from ``other -> d``.

    Event-direct cells are never changed by these two components.  The caller
    writes the learned direct intervention first and then adds each component
    multiplied by a validation-selected elasticity in ``[0, 1]``.
    """

    base = np.asarray(base_matrix, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("base_matrix must be square")
    nodes = base.shape[0]
    edge = np.asarray(direct_edges, dtype=np.int64).reshape(-1)
    delta = np.asarray(direct_deltas, dtype=np.float64).reshape(-1)
    arrival = np.asarray(arrival_share, dtype=np.float64).reshape(-1)
    departure = np.asarray(departure_share, dtype=np.float64).reshape(-1)
    if not (len(edge) == len(delta) == len(arrival) == len(departure)):
        raise ValueError("direct transport arrays must have equal length")
    if np.any(edge < 0) or np.any(edge >= nodes * nodes):
        raise IndexError("direct edge outside matrix")
    if np.any(arrival < 0.0) or np.any(departure < 0.0):
        raise ValueError("role shares must be non-negative")
    if np.any(arrival + departure > 1.0 + 1.0e-6):
        raise ValueError("arrival and departure shares cannot sum above one")
    similarity = None
    if node_similarity is not None:
        similarity = np.asarray(node_similarity, dtype=np.float64)
        if similarity.shape != (nodes, nodes):
            raise ValueError("node_similarity must match base_matrix")
        if np.any(similarity < 0.0):
            raise ValueError("node_similarity must be non-negative")

    direct_mask = np.zeros((nodes, nodes), dtype=bool)
    direct_mask.reshape(-1)[edge] = True
    origin = edge // nodes
    destination = edge % nodes
    arrival_change = np.zeros_like(base, dtype=np.float64)
    departure_change = np.zeros_like(base, dtype=np.float64)

    if similarity is None:
        arrival_groups = [(None, np.ones(nodes, dtype=np.float64))]
        departure_groups = [(None, np.ones(nodes, dtype=np.float64))]
    else:
        arrival_groups = [
            (venue, similarity[venue])
            for venue in np.unique(destination[arrival > 0.0])
        ]
        departure_groups = [
            (venue, similarity[venue])
            for venue in np.unique(origin[departure > 0.0])
        ]

    for venue, venue_similarity in arrival_groups:
        selected = arrival > 0.0
        if venue is not None:
            selected &= destination == venue
        row_mass = np.bincount(
            origin[selected],
            weights=delta[selected] * arrival[selected],
            minlength=nodes,
        ).astype(np.float64)
        candidate_base = np.where(~direct_mask, base, 0.0)
        weighted = candidate_base * venue_similarity[None, :]
        weight_total = weighted.sum(axis=1)
        available_total = candidate_base.sum(axis=1)
        row_transport = np.where(
            row_mass > 0.0, np.minimum(row_mass, available_total), row_mass
        )
        row_weights = np.divide(
            weighted,
            weight_total[:, None],
            out=np.zeros_like(weighted),
            where=weight_total[:, None] > 0.0,
        )
        arrival_change += -row_transport[:, None] * row_weights

    for venue, venue_similarity in departure_groups:
        selected = departure > 0.0
        if venue is not None:
            selected &= origin == venue
        column_mass = np.bincount(
            destination[selected],
            weights=delta[selected] * departure[selected],
            minlength=nodes,
        ).astype(np.float64)
        candidate_base = np.where(~direct_mask, base, 0.0)
        weighted = candidate_base * venue_similarity[:, None]
        weight_total = weighted.sum(axis=0)
        available_total = candidate_base.sum(axis=0)
        column_transport = np.where(
            column_mass > 0.0,
            np.minimum(column_mass, available_total),
            column_mass,
        )
        column_weights = np.divide(
            weighted,
            weight_total[None, :],
            out=np.zeros_like(weighted),
            where=weight_total[None, :] > 0.0,
        )
        departure_change += -column_transport[None, :] * column_weights

    return arrival_change.astype(np.float32), departure_change.astype(np.float32)


def redistribute_direct_effect(
    base_matrix: np.ndarray,
    direct_updates: Dict[Tuple[int, int], float],
    elasticity: TransportElasticity,
) -> np.ndarray:
    """Apply role-conditioned signed marginal transport to one OD matrix.

    ``direct_updates`` maps ``(origin, destination)`` to an additive event
    effect.  An arrival edge has competing alternatives in the same origin row;
    a departure edge has competing alternatives in the same destination
    column.  The caller marks a role by encoding arrival edges with a positive
    destination index and departure edges with ``-(destination + 1)``.  This
    compact representation keeps the operator independent of event metadata.

    The returned matrix is non-negative.  With elasticity one and sufficient
    competing mass, the direct change and its competing changes sum to zero;
    with elasticity below one, the remainder is interpreted as event-generated
    demand rather than substituted demand.
    """

    elasticity.validate()
    result = np.asarray(base_matrix, dtype=np.float64).copy()
    nodes = result.shape[0]
    if result.shape != (nodes, nodes):
        raise ValueError("base_matrix must be square")

    for (origin, encoded_destination), delta in direct_updates.items():
        is_departure = encoded_destination < 0
        destination = -encoded_destination - 1 if is_departure else encoded_destination
        if not (0 <= origin < nodes and 0 <= destination < nodes):
            raise IndexError("OD index outside matrix")
        delta = float(delta)
        result[origin, destination] = max(result[origin, destination] + delta, 0.0)
        fraction = elasticity.departure if is_departure else elasticity.arrival
        shifted_mass = fraction * delta
        if shifted_mass == 0.0:
            continue

        if is_departure:
            candidates = result[:, destination].copy()
            candidates[origin] = 0.0
        else:
            candidates = result[origin, :].copy()
            candidates[destination] = 0.0
        total = float(candidates.sum())
        if total <= 0.0:
            continue
        weights = candidates / total
        competing_change = -shifted_mass * weights
        if is_departure:
            result[:, destination] = np.maximum(
                result[:, destination] + competing_change, 0.0
            )
            result[origin, destination] = max(
                result[origin, destination], 0.0
            )
        else:
            result[origin, :] = np.maximum(
                result[origin, :] + competing_change, 0.0
            )
            result[origin, destination] = max(
                result[origin, destination], 0.0
            )
    return result.astype(np.float32)


__all__ = [
    "EVENT_FEATURE_DIM",
    "EventAnchoredTransportAdapter",
    "FactorizedEventPotential",
    "TransportElasticity",
    "marginal_transport_components",
    "redistribute_direct_effect",
]
