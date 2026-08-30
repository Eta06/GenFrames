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
    warp_oracle_weight: float = 0.0
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
        warp_oracle = prediction.new_zeros(())
        if batch is not None and "flow_t0" in auxiliary and "flow_t1" in auxiliary:
            target_flow0 = _batch_tensor(batch, "flow_t0", prediction)
            target_flow1 = _batch_tensor(batch, "flow_t1", prediction)
            valid = _optional_batch_tensor(batch, "flow_valid", prediction)
            mask0 = _optional_batch_tensor(batch, "flow_mask_t0", prediction)
            mask1 = _optional_batch_tensor(batch, "flow_mask_t1", prediction)
            flow = balanced_flow_loss(
                auxiliary["flow_t0"],
                target_flow0,
                self.config.static_flow_weight,
                valid,
                mask0,
            ) + balanced_flow_loss(
                auxiliary["flow_t1"],
                target_flow1,
                self.config.static_flow_weight,
                valid,
                mask1,
            )
        if "warped0" in auxiliary and "warped1" in auxiliary:
            warp_oracle = oracle_warp_loss(
                auxiliary["warped0"], auxiliary["warped1"], target, self.config.epsilon
            )
        total = (
            self.config.charbonnier_weight * charbonnier + self.config.edge_weight * edge
            + self.config.bilateral_flow_weight * flow
            + self.config.warp_oracle_weight * warp_oracle
        )
        return {
            "total": total,
            "charbonnier": charbonnier,
            "edge": edge,
            "flow": flow,
            "warp_oracle": warp_oracle,
        }


def _gradient(frame: Tensor) -> tuple[Tensor, Tensor]:
    return frame[:, :, :, 1:] - frame[:, :, :, :-1], frame[:, :, 1:, :] - frame[:, :, :-1, :]


def balanced_flow_loss(
    prediction: Tensor,
    target: Tensor,
    static_weight: float = 0.1,
    valid_samples: Tensor | None = None,
    spatial_mask: Tensor | None = None,
) -> Tensor:
    """Normalize moving and static regions separately to avoid zero-flow collapse."""
    if valid_samples is not None:
        valid_samples = valid_samples.bool().flatten()
        prediction = prediction[valid_samples]
        target = target[valid_samples]
        if prediction.shape[0] == 0:
            return prediction.new_zeros(())
        if spatial_mask is not None:
            spatial_mask = spatial_mask[valid_samples]
    endpoint_error = torch.sqrt((prediction - target).square().sum(dim=1, keepdim=True) + 1e-6)
    moving = target.square().sum(dim=1, keepdim=True) > 1e-8
    if spatial_mask is not None:
        moving = moving & spatial_mask.bool()
    stationary = ~moving
    if spatial_mask is not None:
        stationary = stationary & spatial_mask.bool()
    moving_loss = (endpoint_error * moving).sum() / moving.sum().clamp_min(1)
    stationary_loss = (endpoint_error * stationary).sum() / stationary.sum().clamp_min(1)
    return moving_loss + static_weight * stationary_loss


def oracle_warp_loss(
    warped0: Tensor, warped1: Tensor, target: Tensor, epsilon: float = 1e-3
) -> Tensor:
    """Require at least one endpoint warp to explain each target pixel."""
    error0 = torch.sqrt((warped0 - target).square() + epsilon**2).mean(dim=1)
    error1 = torch.sqrt((warped1 - target).square() + epsilon**2).mean(dim=1)
    return torch.minimum(error0, error1).mean()


def _batch_tensor(batch: dict[str, object], key: str, reference: Tensor) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device=reference.device, dtype=reference.dtype, non_blocking=True)


def _optional_batch_tensor(
    batch: dict[str, object], key: str, reference: Tensor
) -> Tensor | None:
    if key not in batch:
        return None
    return _batch_tensor(batch, key, reference)
