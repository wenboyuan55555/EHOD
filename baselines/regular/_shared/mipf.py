"""Shared marginal reconciliation and OD reconstruction for task-adapted baselines.

The decoder uses only predicted marginals and an externally supplied OD prior.
Callers are responsible for constructing that prior from training hours only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class MIPFDiagnostics:
    iterations: int
    converged: bool
    max_row_error: float
    max_column_error: float
    zero_total_samples: int

    def as_dict(self) -> Dict[str, Union[int, bool, float]]:
        return asdict(self)


def _finite_nonnegative(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.maximum(array, 0.0)


def balance_marginals(
    marginal: np.ndarray, epsilon: float = 1.0e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconcile origin and destination totals without using ground truth.

    The common mass is the mean of the two predicted totals.  If exactly one
    predicted side has zero mass, its common mass is distributed uniformly;
    this makes the failure mode explicit and keeps IPF well-defined.  Samples
    with zero mass on both sides remain all-zero.
    """

    value = _finite_nonnegative(marginal, "marginal")
    if value.ndim < 2 or value.shape[-1] != 2:
        raise ValueError("marginal must have shape [..., num_nodes, 2]")
    outbound = value[..., 0].copy()
    inbound = value[..., 1].copy()
    number_of_nodes = outbound.shape[-1]
    outbound_total = outbound.sum(axis=-1, keepdims=True)
    inbound_total = inbound.sum(axis=-1, keepdims=True)
    common_total = 0.5 * (outbound_total + inbound_total)

    outbound_scale = np.divide(
        common_total,
        outbound_total,
        out=np.zeros_like(common_total),
        where=outbound_total > epsilon,
    )
    inbound_scale = np.divide(
        common_total,
        inbound_total,
        out=np.zeros_like(common_total),
        where=inbound_total > epsilon,
    )
    outbound *= outbound_scale
    inbound *= inbound_scale

    positive_total = common_total > epsilon
    missing_outbound = positive_total & (outbound_total <= epsilon)
    missing_inbound = positive_total & (inbound_total <= epsilon)
    outbound += missing_outbound * (common_total / number_of_nodes)
    inbound += missing_inbound * (common_total / number_of_nodes)
    return outbound, inbound


def reconstruct_od(
    prior: np.ndarray,
    marginal: np.ndarray,
    max_iterations: int = 500,
    tolerance: float = 1.0e-6,
    epsilon: float = 1.0e-10,
    support_smoothing: float = 2.0e-3,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, MIPFDiagnostics]]:
    """Reconstruct full OD matrices with converged iterative proportional fitting.

    `prior` must have shape ``[..., N, N]`` and `marginal` shape
    ``[..., N, 2]``.  Batch dimensions must match.  A small positive support is
    added before fitting so that a sparse seasonal prior cannot make a positive
    predicted marginal infeasible.  All-zero samples produce all-zero matrices.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0 or epsilon <= 0.0:
        raise ValueError("tolerance and epsilon must be positive")
    if not 0.0 <= support_smoothing <= 1.0:
        raise ValueError("support_smoothing must be between zero and one")
    matrix = _finite_nonnegative(prior, "prior").copy()
    value = np.asarray(marginal)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("prior must have shape [..., num_nodes, num_nodes]")
    if value.shape[:-2] != matrix.shape[:-2] or value.shape[-2] != matrix.shape[-1]:
        raise ValueError("prior and marginal batch/node dimensions do not match")
    outbound, inbound = balance_marginals(value, epsilon=epsilon)
    common_total = outbound.sum(axis=-1)
    zero_total = common_total <= epsilon

    # A sparse historical prior can make otherwise valid predicted marginals
    # infeasible or make Sinkhorn iterations arbitrarily slow.  Mix in a tiny
    # amount of the predicted independent table, which already has the desired
    # support and does not inspect the target OD matrix.
    denominator = np.maximum(common_total, epsilon)[..., None, None]
    independent = outbound[..., :, None] * inbound[..., None, :] / denominator
    matrix = (1.0 - support_smoothing) * matrix + support_smoothing * independent
    matrix += epsilon
    matrix[zero_total] = 0.0
    converged = False
    row_error = float("inf")
    column_error = float("inf")
    used_iterations = 0
    for used_iterations in range(1, max_iterations + 1):
        row_sum = matrix.sum(axis=-1)
        row_factor = np.divide(
            outbound,
            row_sum,
            out=np.zeros_like(outbound),
            where=row_sum > epsilon,
        )
        matrix *= row_factor[..., :, None]

        column_sum = matrix.sum(axis=-2)
        column_factor = np.divide(
            inbound,
            column_sum,
            out=np.zeros_like(inbound),
            where=column_sum > epsilon,
        )
        matrix *= column_factor[..., None, :]

        row_error = float(np.max(np.abs(matrix.sum(axis=-1) - outbound), initial=0.0))
        column_error = float(np.max(np.abs(matrix.sum(axis=-2) - inbound), initial=0.0))
        scale = max(float(np.max(common_total, initial=0.0)), 1.0)
        if max(row_error, column_error) <= tolerance * scale:
            converged = True
            break

    matrix[zero_total] = 0.0
    result = matrix.astype(np.float32)
    diagnostics = MIPFDiagnostics(
        iterations=used_iterations,
        converged=converged,
        max_row_error=row_error,
        max_column_error=column_error,
        zero_total_samples=int(np.count_nonzero(zero_total)),
    )
    if return_diagnostics:
        return result, diagnostics
    return result


__all__ = ["MIPFDiagnostics", "balance_marginals", "reconstruct_od"]
