"""Region-coherent hard ownership over the immutable Parallax candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .base import InterpolationOutput, prepare_time
from .blocks import ConvAct, ResidualBlock
from .raft_evidence_selector import _build_evidence
from .raft_multifield import (
    GenFramesRaftMultiField,
    RaftMultiFieldConfig,
    build_multifield_candidates,
)


@dataclass(frozen=True)
class RaftRegionAssignmentConfig:
    selector_channels: int = 32
    refinement_steps: tuple[int, ...] = (0, 1, 2, 4)
    damping: float = 0.75
    region_stride: int = 8
    logit_delta_limit: float = 4.0
    confidence_prior_scale: float = 1.0
    abstain_margin: float = 1.5
    freeze_raft: bool = True


class GenFramesRaftRegionAssignment(GenFramesRaftMultiField):
    """Make one coherent field/ownership decision per region, with abstention."""

    def __init__(
        self,
        config: RaftRegionAssignmentConfig | None = None,
        *,
        flow_estimator: nn.Module | None = None,
        pair_transform=None,
    ) -> None:
        self.assignment_config = config or RaftRegionAssignmentConfig()
        super().__init__(
            RaftMultiFieldConfig(
                selector_channels=self.assignment_config.selector_channels,
                refinement_steps=self.assignment_config.refinement_steps,
                damping=self.assignment_config.damping,
                logit_delta_limit=self.assignment_config.logit_delta_limit,
                freeze_raft=self.assignment_config.freeze_raft,
            ),
            flow_estimator=flow_estimator,
            pair_transform=pair_transform,
        )
        del self.selector
        del self.selector_head
        evidence_channels = self.candidate_count * 6
        input_channels = self.candidate_count * 3 + evidence_channels + 3 + 3 + 1
        channels = self.assignment_config.selector_channels
        self.region_encoder = nn.Sequential(
            ConvAct(input_channels, channels),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4),
            ResidualBlock(channels),
        )
        self.assignment_head = nn.Conv2d(channels, self.candidate_count + 1, 3, padding=1)
        nn.init.zeros_(self.assignment_head.weight)
        nn.init.zeros_(self.assignment_head.bias)

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
            refinement_steps=self.assignment_config.refinement_steps,
            damping=self.assignment_config.damping,
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
        stride = self.assignment_config.region_stride
        region_input = functional.avg_pool2d(selector_input, kernel_size=stride, stride=stride)
        region_delta = self.assignment_head(self.region_encoder(region_input))
        candidate_prior = functional.avg_pool2d(
            confidence_prior, kernel_size=stride, stride=stride
        )
        abstain_prior = candidate_prior.max(1, keepdim=True).values - (
            self.assignment_config.abstain_margin
        )
        region_prior = torch.cat((candidate_prior, abstain_prior), dim=1)
        region_logits = region_prior + self.assignment_config.logit_delta_limit * (
            region_delta.tanh()
        )
        logits = functional.interpolate(
            region_logits, size=frame0.shape[-2:], mode="bilinear", align_corners=False
        )
        probabilities = logits.softmax(1)
        assignment_index = logits.argmax(1)
        hard_assignment = functional.one_hot(
            assignment_index, num_classes=self.candidate_count + 1
        ).permute(0, 3, 1, 2).to(candidates.dtype)
        # The forward pass is genuinely discrete; probabilities only carry gradients.
        straight_through = hard_assignment + probabilities - probabilities.detach()
        fallback_index = confidence_prior.argmax(1)
        fallback = candidates.gather(
            1, fallback_index[:, None, None].expand(-1, 1, 3, -1, -1)
        ).squeeze(1)
        appearances = torch.cat((candidates, fallback[:, None]), dim=1)
        frame = (straight_through[:, :, None] * appearances).sum(1)

        source_lookup = candidates.new_tensor(source_ids)
        fallback_source = source_lookup[fallback_index]
        source0_candidates = (source_lookup == 0).to(candidates.dtype)
        source0_probability = (
            probabilities[:, : self.candidate_count]
            * source0_candidates[None, :, None, None]
        ).sum(1, keepdim=True)
        source0_probability = source0_probability + probabilities[:, -1:] * (
            fallback_source == 0
        ).to(candidates.dtype)[:, None]
        chosen_source = source_lookup[fallback_index]
        regular_choice = assignment_index < self.candidate_count
        chosen_source = torch.where(
            regular_choice,
            source_lookup[assignment_index.clamp_max(self.candidate_count - 1)],
            chosen_source,
        )
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
                "assignment_logits": logits,
                "assignment_probabilities": probabilities,
                "assignment_index": assignment_index,
                "fallback_index": fallback_index,
                "fallback_frame": fallback,
                "abstain_mask": assignment_index == self.candidate_count,
                "source0_probability": source0_probability,
                "chosen_source": chosen_source,
            },
        )
