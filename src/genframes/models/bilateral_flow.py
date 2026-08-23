"""Portable, linearly constrained bilateral-flow GenFrames candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from genframes.ops import backward_warp

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock


@dataclass(frozen=True)
class BilateralFlowConfig:
    base_channels: int = 24
    max_flow: float = 20.0
    residual_limit: float = 0.1
    blend_logit_limit: float = 2.0
    coarse_velocity: bool = False


class GenFramesBilateralFlow(FrameInterpolator):
    """Estimate target-centric velocity, warp endpoints, and resolve visibility.

    A shared target-space velocity constrains bilateral flows to constant motion:
    `flow(t->0)=-t*v` and `flow(t->1)=(1-t)*v`. This is intentionally testable on
    the analytic dataset and will later serve as the local-motion branch in less
    restrictive multi-flow experiments.
    """

    def __init__(self, config: BilateralFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or BilateralFlowConfig()
        channels = self.config.base_channels
        self.stem = ConvAct(13, channels)
        self.encoder0 = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.down1 = ConvAct(channels, channels * 2, stride=2)
        self.encoder1 = nn.Sequential(
            ResidualBlock(channels * 2, dilation=2),
            ResidualBlock(channels * 2),
        )
        self.down2 = ConvAct(channels * 2, channels * 3, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels * 3, dilation=2),
            ResidualBlock(channels * 3, dilation=4),
            ResidualBlock(channels * 3),
        )
        self.decode1 = nn.Sequential(
            ConvAct(channels * 5, channels * 2),
            ResidualBlock(channels * 2),
        )
        self.decode0 = nn.Sequential(
            ConvAct(channels * 3, channels),
            ResidualBlock(channels),
        )
        self.head = nn.Conv2d(channels, 6, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.coarse_head: nn.Conv2d | None = None
        if self.config.coarse_velocity:
            self.coarse_head = nn.Conv2d(channels * 3, 2, kernel_size=3, padding=1)
            nn.init.zeros_(self.coarse_head.weight)
            nn.init.zeros_(self.coarse_head.bias)

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        time_map = target_time.expand(-1, -1, frame0.shape[2], frame0.shape[3])
        signed_difference = frame1 - frame0
        features = torch.cat(
            (frame0, frame1, signed_difference, signed_difference.abs(), time_map), dim=1
        )
        skip0 = self.encoder0(self.stem(features))
        skip1 = self.encoder1(self.down1(skip0))
        encoded = self.bottleneck(self.down2(skip1))
        decoded1 = functional.interpolate(
            encoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decode1(torch.cat((decoded1, skip1), dim=1))
        decoded0 = functional.interpolate(
            decoded1, size=skip0.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded0 = self.decode0(torch.cat((decoded0, skip0), dim=1))
        prediction = self.head(decoded0)

        velocity_logits = prediction[:, :2]
        coarse_logits = None
        if self.coarse_head is not None:
            coarse_logits = functional.interpolate(
                self.coarse_head(encoded),
                size=velocity_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            velocity_logits = velocity_logits + coarse_logits
        velocity = self.config.max_flow * velocity_logits.tanh()
        flow_t0 = -target_time * velocity
        flow_t1 = (1.0 - target_time) * velocity
        warped0 = backward_warp(frame0, flow_t0)
        warped1 = backward_warp(frame1, flow_t1)
        target_logit = torch.logit(target_time.clamp(1e-4, 1.0 - 1e-4))
        blend_delta = self.config.blend_logit_limit * prediction[:, 2:3].tanh()
        weight1 = (target_logit + blend_delta).sigmoid()
        residual = self.config.residual_limit * prediction[:, 3:6].tanh()
        frame = (1.0 - weight1) * warped0 + weight1 * warped1 + residual
        auxiliary = {
            "velocity": velocity,
            "flow_t0": flow_t0,
            "flow_t1": flow_t1,
            "weight1": weight1,
            "residual": residual,
            "warped0": warped0,
            "warped1": warped1,
        }
        if coarse_logits is not None:
            auxiliary["coarse_velocity_logits"] = coarse_logits
        return InterpolationOutput(frame=frame, auxiliary=auxiliary)
