"""Losses with explicit components for ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from genframes.models import InterpolationOutput


@dataclass(frozen=True)
class LossConfig:
    charbonnier_weight: float = 1.0
    edge_weight: float = 0.1
    bilateral_flow_weight: float = 0.01
    static_flow_weight: float = 0.1
    epsilon: float = 1e-3


class InterpolationLoss(nn.Module):
    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: InterpolationOutput | Tensor,
        target: Tensor,
        batch: dict[str, object] | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(output, InterpolationOutput):
            prediction = output.frame
            auxiliary = output.auxiliary
        else:
            prediction = output
            auxiliary = {}
        difference = prediction - target
        charbonnier = torch.sqrt(difference.square() + self.config.epsilon**2).mean()
        prediction_dx, prediction_dy = _gradient(prediction)
        target_dx, target_dy = _gradient(target)
        edge = (prediction_dx - target_dx).abs().mean() + (
            prediction_dy - target_dy
        ).abs().mean()
        flow = prediction.new_zeros(())
        if batch is not None and "flow_t0" in auxiliary and "flow_t1" in auxiliary:
            target_flow0 = _batch_tensor(batch, "flow_t0", prediction)
            target_flow1 = _batch_tensor(batch, "flow_t1", prediction)
            flow = balanced_flow_loss(
                auxiliary["flow_t0"], target_flow0, self.config.static_flow_weight
            ) + balanced_flow_loss(
                auxiliary["flow_t1"], target_flow1, self.config.static_flow_weight
            )
        total = (
            self.config.charbonnier_weight * charbonnier + self.config.edge_weight * edge
            + self.config.bilateral_flow_weight * flow
        )
        return {"total": total, "charbonnier": charbonnier, "edge": edge, "flow": flow}


def _gradient(frame: Tensor) -> tuple[Tensor, Tensor]:
    return frame[:, :, :, 1:] - frame[:, :, :, :-1], frame[:, :, 1:, :] - frame[:, :, :-1, :]


def balanced_flow_loss(prediction: Tensor, target: Tensor, static_weight: float = 0.1) -> Tensor:
    """Normalize moving and static regions separately to avoid zero-flow collapse."""
    endpoint_error = torch.sqrt((prediction - target).square().sum(dim=1, keepdim=True) + 1e-6)
    moving = target.square().sum(dim=1, keepdim=True) > 1e-8
    stationary = ~moving
    moving_loss = (endpoint_error * moving).sum() / moving.sum().clamp_min(1)
    stationary_loss = (endpoint_error * stationary).sum() / stationary.sum().clamp_min(1)
    return moving_loss + static_weight * stationary_loss


def _batch_tensor(batch: dict[str, object], key: str, reference: Tensor) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device=reference.device, dtype=reference.dtype, non_blocking=True)
