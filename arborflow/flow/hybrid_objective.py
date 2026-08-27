"""Flow-matching objectives and physical-unit reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from arborflow.data.dynamic_batch import GeometryBatch


def masked_velocity_mse(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Task-book branch-average squared L2 velocity objective."""

    squared_l2 = (predicted_velocity - target_velocity).square().sum(dim=-1)
    weights = padding_mask.to(squared_l2.dtype)
    return (squared_l2 * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass(frozen=True)
class GeometryMetrics:
    loss: float
    control_rmse_um: float
    radius_rmse_um: float
    finite: bool


def geometry_metrics(
    predicted_velocity: torch.Tensor,
    batch: GeometryBatch,
    *,
    coordinate_scale_um: float,
    radius_scale_um: float,
) -> GeometryMetrics:
    """Extrapolate the current state to t=1 and compare with oracle geometry."""

    loss = masked_velocity_mse(
        predicted_velocity, batch.target_velocity, batch.padding_mask
    )
    remaining_time = 1.0 - batch.global_time[:, None, None]
    predicted_target = batch.current_geometry + remaining_time * predicted_velocity
    difference = predicted_target - batch.target_geometry
    mask = batch.padding_mask.to(difference.dtype)
    control_vectors = difference[..., :9].reshape(*difference.shape[:2], 3, 3)
    control_squared_distance = control_vectors.square().sum(dim=-1)
    control_denominator = (mask.sum() * 3.0).clamp_min(1.0)
    control_rmse = torch.sqrt(
        (control_squared_distance * mask.unsqueeze(-1)).sum() / control_denominator
    ) * coordinate_scale_um
    radius_squared = difference[..., 9:].square().sum(dim=-1)
    radius_denominator = (mask.sum() * 2.0).clamp_min(1.0)
    radius_rmse = torch.sqrt(
        (radius_squared * mask).sum() / radius_denominator
    ) * radius_scale_um
    finite = bool(
        torch.isfinite(loss)
        & torch.isfinite(control_rmse)
        & torch.isfinite(radius_rmse)
        & torch.isfinite(predicted_velocity).all()
    )
    return GeometryMetrics(
        float(loss.detach()),
        float(control_rmse.detach()),
        float(radius_rmse.detach()),
        finite,
    )
