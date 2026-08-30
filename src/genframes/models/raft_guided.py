"""HQ RAFT-guided interpolation with a small trainable occlusion/fusion head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from genframes.ops import backward_warp

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock


@dataclass(frozen=True)
class RaftGuidedConfig:
    fusion_channels: int = 24
    blend_logit_limit: float = 4.0
    residual_limit: float = 0.1
    freeze_raft: bool = True


class GenFramesRaftGuided(FrameInterpolator):
    """Use global endpoint flow for motion and learn only target visibility/fusion."""

    def __init__(
        self,
        config: RaftGuidedConfig | None = None,
        *,
        flow_estimator: nn.Module | None = None,
        pair_transform=None,
    ) -> None:
        super().__init__()
        self.config = config or RaftGuidedConfig()
        if flow_estimator is None:
            weights = Raft_Small_Weights.DEFAULT
            self.flow_estimator = raft_small(weights=weights)
            self.pair_transform = weights.transforms()
            self.flow_weights_id = str(weights)
            self.flow_weights_url = weights.url
        else:
            self.flow_estimator = flow_estimator
            self.pair_transform = pair_transform or _identity_pair
            self.flow_weights_id = "injected"
            self.flow_weights_url = None
        if self.config.freeze_raft:
            for parameter in self.flow_estimator.parameters():
                parameter.requires_grad_(False)
        channels = self.config.fusion_channels
        self.stem = ConvAct(20, channels)
        self.encoder0 = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.down1 = ConvAct(channels, channels * 2, stride=2)
        self.encoder1 = nn.Sequential(
            ResidualBlock(channels * 2, dilation=2), ResidualBlock(channels * 2)
        )
        self.down2 = ConvAct(channels * 2, channels * 3, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels * 3, dilation=2),
            ResidualBlock(channels * 3, dilation=4),
            ResidualBlock(channels * 3),
        )
        self.decode1 = nn.Sequential(
            ConvAct(channels * 5, channels * 2), ResidualBlock(channels * 2)
        )
        self.decode0 = nn.Sequential(
            ConvAct(channels * 3, channels), ResidualBlock(channels)
        )
        self.head = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_raft:
            self.flow_estimator.eval()
        return self

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        flow01, flow10 = self._endpoint_flows(frame0, frame1)
        flow_t0 = -(1.0 - target_time) * target_time * flow01 + target_time.square() * flow10
        flow_t1 = (1.0 - target_time).square() * flow01 - (
            target_time * (1.0 - target_time) * flow10
        )
        warped0 = backward_warp(frame0, flow_t0.to(frame0.dtype))
        warped1 = backward_warp(frame1, flow_t1.to(frame1.dtype))
        time_map = target_time.expand(-1, -1, frame0.shape[-2], frame0.shape[-1])
        evidence = torch.cat(
            (
                frame0,
                frame1,
                warped0,
                warped1,
                (warped1 - warped0).abs(),
                flow_t0.to(frame0.dtype),
                flow_t1.to(frame0.dtype),
                time_map,
            ),
            dim=1,
        )
        skip0 = self.encoder0(self.stem(evidence))
        skip1 = self.encoder1(self.down1(skip0))
        encoded = self.bottleneck(self.down2(skip1))
        decoded1 = functional.interpolate(
            encoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decode1(torch.cat((decoded1, skip1), dim=1))
        decoded0 = functional.interpolate(
            decoded1, size=skip0.shape[-2:], mode="bilinear", align_corners=False
        )
        prediction = self.head(self.decode0(torch.cat((decoded0, skip0), dim=1)))
        target_logit = torch.logit(target_time.clamp(1e-4, 1.0 - 1e-4))
        blend_delta = self.config.blend_logit_limit * prediction[:, :1].tanh()
        weight1 = (target_logit + blend_delta).sigmoid()
        residual = self.config.residual_limit * prediction[:, 1:].tanh()
        frame = (1.0 - weight1) * warped0 + weight1 * warped1 + residual
        return InterpolationOutput(
            frame=frame,
            auxiliary={
                "velocity": flow_t1 - flow_t0,
                "flow01": flow01,
                "flow10": flow10,
                "flow_t0": flow_t0,
                "flow_t1": flow_t1,
                "warped0": warped0,
                "warped1": warped1,
                "weight1": weight1,
                "residual": residual,
            },
        )

    def _endpoint_flows(self, frame0: Tensor, frame1: Tensor) -> tuple[Tensor, Tensor]:
        height, width = frame0.shape[-2:]
        padded_size = (math.ceil(height / 8) * 8, math.ceil(width / 8) * 8)
        padding = (0, padded_size[1] - width, 0, padded_size[0] - height)
        input0 = functional.pad(frame0.float(), padding, mode="replicate")
        input1 = functional.pad(frame1.float(), padding, mode="replicate")
        input0, input1 = self.pair_transform(input0, input1)
        context = torch.no_grad() if self.config.freeze_raft else torch.enable_grad()
        with context:
            flow01 = self.flow_estimator(input0, input1)[-1][..., :height, :width]
            flow10 = self.flow_estimator(input1, input0)[-1][..., :height, :width]
        return flow01, flow10


def _identity_pair(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    return first, second
