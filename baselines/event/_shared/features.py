from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ehod.data.od import ACTIVE_CITY, N_NODES


N_EVENT_TYPES = 8 if ACTIVE_CITY == "chicago" else 7
N_VENUES = 10 if ACTIVE_CITY == "chicago" else 8
STRUCTURED_DIM = 2 * 5 * N_EVENT_TYPES
VENUE_ROLE_DIM = 2 * N_VENUES
TIME_DIM = 4
EVENT_FEATURE_DIM = STRUCTURED_DIM + VENUE_ROLE_DIM + TIME_DIM


def event_feature_tokens(event_features: torch.Tensor) -> torch.Tensor:
    """Convert flat structured fields into ten O/D-role × phase tokens."""
    batch = event_features.shape[0]
    typed_phase = event_features[:, :STRUCTURED_DIM].reshape(
        batch, 2, 5, N_EVENT_TYPES
    )
    venue_role = event_features[
        :, STRUCTURED_DIM : STRUCTURED_DIM + VENUE_ROLE_DIM
    ].reshape(batch, 2, 1, N_VENUES).expand(-1, -1, 5, -1)
    time_role = event_features[:, -TIME_DIM:].reshape(
        batch, 1, 1, TIME_DIM
    ).expand(-1, 2, 5, -1)
    return torch.cat([typed_phase, venue_role, time_role], dim=-1).reshape(
        batch, 10, N_EVENT_TYPES + N_VENUES + TIME_DIM
    )


