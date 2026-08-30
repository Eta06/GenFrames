"""Training-only privileged optical-flow supervision for real triplets."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from genframes.ops import backward_warp


class PrivilegedRaftSmallTeacher:
    """Generate target-to-endpoint pseudo-flow without entering inference graphs."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_small(weights=self.weights).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def enrich_batch(
        self,
        batch: dict[str, object],
        frame0: Tensor,
        frame1: Tensor,
        target: Tensor,
    ) -> dict[str, object]:
        teacher0 = self._flow(target, frame0)
        teacher1 = self._flow(target, frame1)
        existing0 = _batch_tensor(batch, "flow_t0", frame0)
        existing1 = _batch_tensor(batch, "flow_t1", frame0)
        exact_valid = _batch_tensor(batch, "flow_valid", frame0).bool().view(-1, 1, 1, 1)
        flow0 = torch.where(exact_valid, existing0, teacher0)
        flow1 = torch.where(exact_valid, existing1, teacher1)
        teacher_mask0 = _useful_warp_mask(frame0, target, teacher0)
        teacher_mask1 = _useful_warp_mask(frame1, target, teacher1)
        ones = torch.ones_like(teacher_mask0)
        enriched = dict(batch)
        enriched.update(
            {
                "flow_t0": flow0,
                "flow_t1": flow1,
                "flow_valid": torch.ones(frame0.shape[0], device=frame0.device, dtype=torch.bool),
                "flow_mask_t0": torch.where(exact_valid, ones, teacher_mask0),
                "flow_mask_t1": torch.where(exact_valid, ones, teacher_mask1),
            }
        )
        return enriched

    def _flow(self, target: Tensor, endpoint: Tensor) -> Tensor:
        height, width = target.shape[-2:]
        padded_height = math.ceil(height / 8) * 8
        padded_width = math.ceil(width / 8) * 8
        padding = (0, padded_width - width, 0, padded_height - height)
        target = functional.pad(target.float(), padding, mode="replicate")
        endpoint = functional.pad(endpoint.float(), padding, mode="replicate")
        target, endpoint = self.transforms(target, endpoint)
        return self.model(target, endpoint)[-1][..., :height, :width]


def _useful_warp_mask(endpoint: Tensor, target: Tensor, flow: Tensor) -> Tensor:
    warped = backward_warp(endpoint.float(), flow.float())
    warped_error = (warped - target.float()).abs().mean(dim=1, keepdim=True)
    raw_error = (endpoint.float() - target.float()).abs().mean(dim=1, keepdim=True)
    return (warped_error < raw_error) & (warped_error < 0.15)


def _batch_tensor(batch: dict[str, object], key: str, reference: Tensor) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device=reference.device, dtype=reference.dtype, non_blocking=True)
