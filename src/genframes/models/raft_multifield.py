"""RAFT-oracle multi-field candidates with a compact convex spatial selector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from genframes.ops import backward_warp, inverse_flow_hypotheses

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock


@dataclass(frozen=True)
class RaftMultiFieldConfig:
    selector_channels: int = 32
    refinement_steps: tuple[int, ...] = (0, 1, 2, 4)
    damping: float = 0.75
    logit_delta_limit: float = 6.0
    freeze_raft: bool = True


class GenFramesRaftMultiField(FrameInterpolator):
    """Keep motion hypotheses fixed and learn only their convex spatial selection."""

    def __init__(
        self,
        config: RaftMultiFieldConfig | None = None,
        *,
        flow_estimator: nn.Module | None = None,
        pair_transform=None,
    ) -> None:
        super().__init__()
        self.config = config or RaftMultiFieldConfig()
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
        self.candidate_count = 2 + 2 * len(self.config.refinement_steps)
        input_channels = self.candidate_count * 3 + 3 + 3 + 1
        channels = self.config.selector_channels
        self.selector = nn.Sequential(
            ConvAct(input_channels, channels),
            ResidualBlock(channels),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4),
            ResidualBlock(channels),
        )
        self.selector_head = nn.Conv2d(channels, self.candidate_count, 3, padding=1)
        nn.init.zeros_(self.selector_head.weight)
        nn.init.zeros_(self.selector_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_raft:
            self.flow_estimator.eval()
        return self

    def selector_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith("flow_estimator.")
        }

    def load_selector_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            name for name in self.state_dict() if name.startswith("flow_estimator.")
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"incompatible selector checkpoint: {incompatible}")

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        flow01, flow10 = self._endpoint_flows(frame0, frame1)
        quadratic0 = -(1.0 - target_time) * target_time * flow01 + target_time.square() * flow10
        quadratic1 = (1.0 - target_time).square() * flow01 - (
            target_time * (1.0 - target_time) * flow10
        )
        fields0 = inverse_flow_hypotheses(
            flow01,
            target_time.flatten(),
            refinement_steps=self.config.refinement_steps,
            damping=self.config.damping,
        )
        fields1 = inverse_flow_hypotheses(
            flow10,
            1.0 - target_time.flatten(),
            refinement_steps=self.config.refinement_steps,
            damping=self.config.damping,
        )
        candidates = (
            backward_warp(frame0, quadratic0.to(frame0.dtype)),
            backward_warp(frame1, quadratic1.to(frame1.dtype)),
        ) + tuple(backward_warp(frame0, field.to(frame0.dtype)) for field in fields0) + tuple(
            backward_warp(frame1, field.to(frame1.dtype)) for field in fields1
        )
        candidate_stack = torch.stack(candidates, dim=1)
        consensus_mean = candidate_stack.mean(dim=1)
        consensus_std = candidate_stack.std(dim=1)
        time_map = target_time.expand(-1, -1, frame0.shape[-2], frame0.shape[-1])
        evidence = torch.cat(
            (candidate_stack.flatten(1, 2), consensus_mean, consensus_std, time_map), dim=1
        )
        delta = self.config.logit_delta_limit * self.selector_head(self.selector(evidence)).tanh()
        prior = delta.new_full(delta.shape, -8.0)
        prior[:, 0] = torch.log((1.0 - time_map[:, 0]).clamp_min(1e-4))
        prior[:, 1] = torch.log(time_map[:, 0].clamp_min(1e-4))
        logits = prior + delta
        weights = logits.softmax(dim=1)
        frame = (weights[:, :, None] * candidate_stack).sum(dim=1)
        return InterpolationOutput(
            frame=frame,
            auxiliary={
                "velocity": quadratic1 - quadratic0,
                "flow01": flow01,
                "flow10": flow10,
                "flow_t0": quadratic0,
                "flow_t1": quadratic1,
                "warped0": candidates[0],
                "warped1": candidates[1],
                "candidate_stack": candidate_stack,
                "candidate_logits": logits,
                "candidate_weights": weights,
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
