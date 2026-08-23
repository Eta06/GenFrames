"""First original, portable GenFrames learning candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock


@dataclass(frozen=True)
class MiniConfig:
    base_channels: int = 24
    residual_limit: float = 0.25


class GenFramesMini(FrameInterpolator):
    """Time-conditioned U-Net that learns a bounded correction to linear blend.

    The model intentionally uses only convolution, SiLU, concatenation, sigmoid,
    tanh, and bilinear resize. It is a correctness milestone and a flow-free
    experiment direction, not the final GenFrames architecture.
    """

    def __init__(self, config: MiniConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniConfig()
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
        self.head = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        time_map = target_time.expand(-1, -1, frame0.shape[2], frame0.shape[3])
        blend = (1.0 - target_time) * frame0 + target_time * frame1
        features = torch.cat((frame0, frame1, blend, frame1 - frame0, time_map), dim=1)

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
        residual = self.config.residual_limit * prediction[:, :3].tanh()
        confidence = prediction[:, 3:].sigmoid()
        frame = blend + confidence * residual
        return InterpolationOutput(
            frame=frame,
            auxiliary={
                "blend": blend,
                "residual": residual,
                "confidence": confidence,
            },
        )
