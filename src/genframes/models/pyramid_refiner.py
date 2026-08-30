"""Target-centric coarse-to-fine single-velocity correspondence refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from genframes.ops import backward_warp, correlation_soft_argmax, local_correlation

from .blocks import ConvAct, ResidualBlock


def resize_flow(flow: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a pixel-unit flow while preserving its displacement in image space."""
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = size
    resized = functional.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    scale = flow.new_tensor((new_width / old_width, new_height / old_height)).view(1, 2, 1, 1)
    return resized * scale


class _FeaturePyramid(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            stages.append(
                nn.Sequential(
                    ConvAct(input_channels, output_channels, stride=2),
                    ResidualBlock(output_channels),
                )
            )
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)

    def forward(self, frame: Tensor) -> list[Tensor]:
        features = []
        for stage in self.stages:
            frame = stage(frame)
            features.append(frame)
        return features


class _VelocityUpdate(nn.Module):
    def __init__(self, feature_channels: int, radius: int) -> None:
        super().__init__()
        window_channels = (2 * radius + 1) ** 2
        input_channels = 2 * feature_channels + 2 * window_channels + 7
        hidden_channels = max(24, feature_channels)
        self.radius = radius
        self.body = nn.Sequential(
            ConvAct(input_channels, hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.head = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        self.moment_scale = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        warped0: Tensor,
        warped1: Tensor,
        velocity: Tensor,
        time_map: Tensor,
        temperature: float,
    ) -> Tensor:
        correlation01 = local_correlation(warped0, warped1, self.radius)
        correlation10 = local_correlation(warped1, warped0, self.radius)
        moment01 = correlation_soft_argmax(correlation01, self.radius, temperature)
        moment10 = correlation_soft_argmax(correlation10, self.radius, temperature)
        moment = 0.5 * (moment01 - moment10)
        evidence = torch.cat(
            (
                warped0,
                warped1,
                correlation01,
                correlation10,
                moment01,
                moment10,
                velocity,
                time_map,
            ),
            dim=1,
        )
        learned = self.radius * self.head(self.body(evidence)).tanh()
        return learned + self.moment_scale.tanh() * moment


class PyramidVelocityRefiner(nn.Module):
    """Refine one interval velocity from 1/16 through 1/2 endpoint features."""

    def __init__(
        self,
        channels: tuple[int, ...] = (12, 16, 24, 32),
        radii: tuple[int, ...] = (2, 2, 2, 4),
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != len(radii) or not channels:
            raise ValueError("channels and radii must have the same non-zero length")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.encoder = _FeaturePyramid(channels)
        self.updates = nn.ModuleList(
            _VelocityUpdate(feature_channels, radius)
            for feature_channels, radius in zip(channels, radii, strict=True)
        )

    def forward(
        self,
        frame0: Tensor,
        frame1: Tensor,
        target_time: Tensor,
        base_velocity: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pyramid0 = self.encoder(frame0)
        pyramid1 = self.encoder(frame1)
        correction: Tensor | None = None
        diagnostics: dict[str, Tensor] = {}
        level_indices = range(len(pyramid0) - 1, -1, -1)
        for level in level_indices:
            feature0 = pyramid0[level]
            feature1 = pyramid1[level]
            size = feature0.shape[-2:]
            base_at_level = resize_flow(base_velocity, size)
            if correction is None:
                correction = torch.zeros_like(base_at_level)
            else:
                correction = resize_flow(correction, size)
            current_velocity = base_at_level + correction
            level_time = target_time.expand(-1, -1, *size)
            warped0 = backward_warp(feature0, -level_time * current_velocity)
            warped1 = backward_warp(feature1, (1.0 - level_time) * current_velocity)
            delta = self.updates[level](
                warped0,
                warped1,
                current_velocity,
                level_time,
                self.temperature,
            )
            correction = correction + delta
            diagnostics[f"pyramid_delta_{level}"] = delta
        assert correction is not None
        full_correction = resize_flow(correction, base_velocity.shape[-2:])
        diagnostics["pyramid_velocity_correction"] = full_correction
        return base_velocity + full_correction, diagnostics
