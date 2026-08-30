"""Wide-receptive-field coarse-to-fine independent endpoint-flow updates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from genframes.ops import backward_warp

from .blocks import ConvAct, ResidualBlock
from .pyramid_refiner import resize_flow


class _IndependentUpdateBlock(nn.Module):
    def __init__(self, channels: int, limit: float) -> None:
        super().__init__()
        self.limit = limit
        self.body = nn.Sequential(
            ConvAct(17, channels),
            ResidualBlock(channels),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4),
            ResidualBlock(channels, dilation=8),
        )
        self.head = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        frame0: Tensor,
        frame1: Tensor,
        warped0: Tensor,
        warped1: Tensor,
        flow0: Tensor,
        flow1: Tensor,
        time_map: Tensor,
    ) -> tuple[Tensor, Tensor]:
        evidence = torch.cat(
            (frame0, frame1, warped0, warped1, flow0, flow1, time_map), dim=1
        )
        delta = self.limit * self.head(self.body(evidence)).tanh()
        return delta[:, :2], delta[:, 2:]


class IndependentFlowPyramidRefiner(nn.Module):
    """Refine target-to-endpoint flows at 1/8, 1/4, and 1/2 resolution."""

    def __init__(
        self,
        channels: int = 32,
        factors: tuple[int, ...] = (8, 4, 2),
        limits: tuple[float, ...] = (32.0, 8.0, 4.0),
    ) -> None:
        super().__init__()
        if len(factors) != len(limits) or not factors:
            raise ValueError("factors and limits must have the same non-zero length")
        if any(factor <= 0 for factor in factors):
            raise ValueError("pyramid factors must be positive")
        self.factors = factors
        self.blocks = nn.ModuleList(
            _IndependentUpdateBlock(channels, limit) for limit in limits
        )

    def forward(
        self,
        frame0: Tensor,
        frame1: Tensor,
        target_time: Tensor,
        base_flow0: Tensor,
        base_flow1: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        height, width = frame0.shape[-2:]
        correction0: Tensor | None = None
        correction1: Tensor | None = None
        diagnostics = {}
        for level, (factor, block) in enumerate(zip(self.factors, self.blocks, strict=True)):
            size = (math.ceil(height / factor), math.ceil(width / factor))
            level_frame0 = functional.interpolate(
                frame0, size=size, mode="bilinear", align_corners=False
            )
            level_frame1 = functional.interpolate(
                frame1, size=size, mode="bilinear", align_corners=False
            )
            level_flow0 = resize_flow(base_flow0, size)
            level_flow1 = resize_flow(base_flow1, size)
            if correction0 is None:
                correction0 = torch.zeros_like(level_flow0)
                correction1 = torch.zeros_like(level_flow1)
            else:
                correction0 = resize_flow(correction0, size)
                assert correction1 is not None
                correction1 = resize_flow(correction1, size)
            current0 = level_flow0 + correction0
            current1 = level_flow1 + correction1
            warped0 = backward_warp(level_frame0, current0)
            warped1 = backward_warp(level_frame1, current1)
            time_map = target_time.expand(-1, -1, *size).to(level_frame0.dtype)
            delta0, delta1 = block(
                level_frame0,
                level_frame1,
                warped0,
                warped1,
                current0,
                current1,
                time_map,
            )
            correction0 = correction0 + delta0.to(correction0.dtype)
            correction1 = correction1 + delta1.to(correction1.dtype)
            diagnostics[f"independent_pyramid_delta0_{level}"] = delta0
            diagnostics[f"independent_pyramid_delta1_{level}"] = delta1
        assert correction0 is not None and correction1 is not None
        full0 = resize_flow(correction0, (height, width))
        full1 = resize_flow(correction1, (height, width))
        diagnostics["independent_pyramid_correction0"] = full0
        diagnostics["independent_pyramid_correction1"] = full1
        return base_flow0 + full0, base_flow1 + full1, diagnostics
