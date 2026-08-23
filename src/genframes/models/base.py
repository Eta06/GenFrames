"""Shared interpolation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class InterpolationOutput:
    frame: Tensor
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


def prepare_time(time: Tensor | float, reference: Tensor) -> Tensor:
    """Return target time as `[B, 1, 1, 1]` on the reference device/dtype."""
    batch = reference.shape[0]
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.ndim == 1 and value.shape[0] == 1 and batch != 1:
        value = value.expand(batch)
    if value.ndim != 1 or value.shape[0] != batch:
        raise ValueError(f"time must be scalar or shape [B], received {tuple(value.shape)}")
    if torch.any((value <= 0) | (value >= 1)):
        raise ValueError("target time must lie strictly inside (0, 1)")
    return value[:, None, None, None]


class FrameInterpolator(nn.Module, ABC):
    """Base class for all GenFrames candidates."""

    @abstractmethod
    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        raise NotImplementedError

    @staticmethod
    def validate_frames(frame0: Tensor, frame1: Tensor) -> None:
        if frame0.ndim != 4 or frame0.shape[1] != 3:
            raise ValueError("frames must have shape [B, 3, H, W]")
        if frame0.shape != frame1.shape:
            raise ValueError("input frames must have identical shapes")
        if not frame0.is_floating_point() or not frame1.is_floating_point():
            raise ValueError("input frames must be floating point")

