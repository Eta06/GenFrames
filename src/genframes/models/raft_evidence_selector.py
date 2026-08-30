"""Confidence and cycle-aware selector over the immutable Parallax fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from genframes.ops import backward_warp

from .base import InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock
from .raft_multifield import (
    GenFramesRaftMultiField,
    RaftMultiFieldConfig,
    build_multifield_candidates,
)


@dataclass(frozen=True)
class RaftEvidenceSelectorConfig:
    selector_channels: int = 32
    refinement_steps: tuple[int, ...] = (0, 1, 2, 4)
    damping: float = 0.75
    logit_delta_limit: float = 4.0
    confidence_prior_scale: float = 1.0
    freeze_raft: bool = True


class GenFramesRaftEvidenceSelector(GenFramesRaftMultiField):
    """Select fields using cycle, inverse-residual, validity, and appearance evidence."""

    evidence_names = (
        "photometric_cycle",
        "inverse_residual",
        "flow_cycle",
        "source_valid",
        "cross_valid",
        "consensus_deviation",
    )

    def __init__(
        self,
        config: RaftEvidenceSelectorConfig | None = None,
        *,
        flow_estimator: nn.Module | None = None,
        pair_transform=None,
    ) -> None:
        self.evidence_config = config or RaftEvidenceSelectorConfig()
        super().__init__(
            RaftMultiFieldConfig(
                selector_channels=self.evidence_config.selector_channels,
                refinement_steps=self.evidence_config.refinement_steps,
                damping=self.evidence_config.damping,
                logit_delta_limit=self.evidence_config.logit_delta_limit,
                freeze_raft=self.evidence_config.freeze_raft,
            ),
            flow_estimator=flow_estimator,
            pair_transform=pair_transform,
        )
        del self.selector
        del self.selector_head
        evidence_channels = self.candidate_count * len(self.evidence_names)
        input_channels = self.candidate_count * 3 + evidence_channels + 3 + 3 + 1
        channels = self.evidence_config.selector_channels
        self.evidence_selector = nn.Sequential(
            ConvAct(input_channels, channels),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4),
            ResidualBlock(channels, dilation=8),
            ResidualBlock(channels),
        )
        self.evidence_head = nn.Conv2d(channels, self.candidate_count, 3, padding=1)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        flow01, flow10 = self._endpoint_flows(frame0, frame1)
        candidates, fields, source_ids = build_multifield_candidates(
            frame0,
            frame1,
            target_time,
            flow01,
            flow10,
            refinement_steps=self.evidence_config.refinement_steps,
            damping=self.evidence_config.damping,
        )
        evidence, confidence_prior = _build_evidence(
            frame0,
            frame1,
            target_time,
            flow01,
            flow10,
            candidates,
            fields,
            source_ids,
        )
        consensus_mean = candidates.mean(1)
        consensus_std = candidates.std(1)
        time_map = target_time.expand(-1, -1, frame0.shape[-2], frame0.shape[-1])
        selector_input = torch.cat(
            (
                candidates.flatten(1, 2),
                evidence.flatten(1, 2),
                consensus_mean,
                consensus_std,
                time_map,
            ),
            dim=1,
        )
        coarse = functional.avg_pool2d(selector_input, kernel_size=2, stride=2)
        delta = self.evidence_head(self.evidence_selector(coarse))
        delta = functional.interpolate(
            delta, size=frame0.shape[-2:], mode="bilinear", align_corners=False
        )
        delta = self.evidence_config.logit_delta_limit * delta.tanh()
        logits = self.evidence_config.confidence_prior_scale * confidence_prior + delta
        weights = logits.softmax(1)
        frame = (weights[:, :, None] * candidates).sum(1)
        return InterpolationOutput(
            frame=frame,
            auxiliary={
                "velocity": fields[1] - fields[0],
                "flow01": flow01,
                "flow10": flow10,
                "flow_t0": fields[0],
                "flow_t1": fields[1],
                "warped0": candidates[:, 0],
                "warped1": candidates[:, 1],
                "candidate_stack": candidates,
                "candidate_fields": torch.stack(fields, dim=1),
                "candidate_evidence": evidence,
                "confidence_prior": confidence_prior,
                "candidate_logits": logits,
                "candidate_weights": weights,
            },
        )


def _build_evidence(
    frame0,
    frame1,
    target_time,
    flow01,
    flow10,
    candidates,
    fields,
    source_ids,
):
    consensus = candidates.median(dim=1).values
    evidence_items = []
    confidence_items = []
    for index, (field, source_id) in enumerate(zip(fields, source_ids, strict=True)):
        endpoint_flow = flow01 if source_id == 0 else flow10
        reverse_flow = flow10 if source_id == 0 else flow01
        opposite = frame1 if source_id == 0 else frame0
        fraction = target_time if source_id == 0 else 1.0 - target_time
        sampled_endpoint = backward_warp(endpoint_flow, field)
        inverse_residual = torch.linalg.vector_norm(
            field + fraction * sampled_endpoint, dim=1, keepdim=True
        )
        cross_field = field + sampled_endpoint
        cross_appearance = backward_warp(opposite, cross_field.to(opposite.dtype))
        photometric = (candidates[:, index] - cross_appearance).abs().mean(1, keepdim=True)
        sampled_reverse = backward_warp(reverse_flow, cross_field)
        flow_cycle = torch.linalg.vector_norm(
            sampled_endpoint + sampled_reverse, dim=1, keepdim=True
        )
        source_valid = _flow_valid(field)
        cross_valid = _flow_valid(cross_field)
        consensus_deviation = (candidates[:, index] - consensus).abs().mean(1, keepdim=True)
        inverse_scaled = torch.log1p(inverse_residual) / 4.0
        cycle_scaled = torch.log1p(flow_cycle) / 4.0
        item = torch.cat(
            (
                photometric,
                inverse_scaled,
                cycle_scaled,
                source_valid,
                cross_valid,
                consensus_deviation,
            ),
            dim=1,
        )
        evidence_items.append(item)
        confidence_items.append(
            -4.0 * photometric[:, 0]
            - 2.0 * inverse_scaled[:, 0]
            - cycle_scaled[:, 0]
            - 4.0 * (1.0 - source_valid[:, 0])
            - 2.0 * (1.0 - cross_valid[:, 0])
        )
    return torch.stack(evidence_items, dim=1), torch.stack(confidence_items, dim=1)


def _flow_valid(flow: Tensor) -> Tensor:
    height, width = flow.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x[None] + flow[:, 0]
    source_y = y[None] + flow[:, 1]
    return (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    ).to(flow.dtype).unsqueeze(1)
