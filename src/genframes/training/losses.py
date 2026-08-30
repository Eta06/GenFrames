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
    visibility_weight: float = 0.0
    candidate_selection_weight: float = 0.0
    candidate_static_weight: float = 0.1
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
        visibility = prediction.new_zeros(())
        candidate_selection = prediction.new_zeros(())
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
        if batch is not None and "weight1" in auxiliary and "visibility0" in batch:
            visibility = visibility_blend_loss(auxiliary["weight1"], batch, prediction)
        if "candidate_logits" in auxiliary and "candidate_stack" in auxiliary:
            candidate_selection = candidate_selection_loss(
                auxiliary["candidate_logits"],
                auxiliary["candidate_stack"],
                target,
                static_weight=self.config.candidate_static_weight,
            )
        total = (
            self.config.charbonnier_weight * charbonnier + self.config.edge_weight * edge
            + self.config.bilateral_flow_weight * flow
            + self.config.warp_oracle_weight * warp_oracle
            + self.config.visibility_weight * visibility
            + self.config.candidate_selection_weight * candidate_selection
        )
        return {
            "total": total,
            "charbonnier": charbonnier,
            "edge": edge,
            "flow": flow,
            "warp_oracle": warp_oracle,
            "visibility": visibility,
            "candidate_selection": candidate_selection,
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


def visibility_blend_loss(
    prediction_weight1: Tensor, batch: dict[str, object], reference: Tensor
) -> Tensor:
    """Supervise fusion weights where synthetic endpoint visibility is exact."""
    visibility0 = _batch_tensor(batch, "visibility0", reference)
    visibility1 = _batch_tensor(batch, "visibility1", reference)
    valid = _optional_batch_tensor(batch, "visibility_valid", reference)
    if valid is not None:
        valid = valid.bool().flatten()
        prediction_weight1 = prediction_weight1[valid]
        visibility0 = visibility0[valid]
        visibility1 = visibility1[valid]
    if prediction_weight1.shape[0] == 0:
        return prediction_weight1.new_zeros(())
    target_time = _batch_tensor(batch, "time", reference).flatten()
    if valid is not None:
        target_time = target_time[valid]
    target_time = target_time[:, None, None, None]
    contribution0 = visibility0 * (1.0 - target_time)
    contribution1 = visibility1 * target_time
    denominator = contribution0 + contribution1
    usable = denominator > 1e-6
    target_weight1 = contribution1 / denominator.clamp_min(1e-6)
    error = torch.nn.functional.smooth_l1_loss(
        prediction_weight1, target_weight1, reduction="none"
    )
    return (error * usable).sum() / usable.sum().clamp_min(1)


def candidate_selection_loss(
    logits: Tensor,
    candidates: Tensor,
    target: Tensor,
    *,
    static_weight: float = 0.1,
) -> Tensor:
    """Teach a selector which fixed candidate best explains each target pixel."""
    if candidates.ndim != 5 or candidates.shape[2] != target.shape[1]:
        raise ValueError("candidates must have shape [B, K, C, H, W]")
    if logits.shape != (
        candidates.shape[0],
        candidates.shape[1],
        candidates.shape[3],
        candidates.shape[4],
    ):
        raise ValueError("logits must have shape [B, K, H, W] matching candidates")
    errors = (candidates.detach() - target[:, None]).abs().mean(dim=2)
    labels = errors.argmin(dim=1)
    per_pixel = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    conflict = (candidates[:, 0] - candidates[:, 1]).abs().mean(dim=1) > 0.05
    weights = torch.where(conflict, torch.ones_like(per_pixel), static_weight)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1e-6)


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
