"""Losses with explicit components for ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossConfig:
    charbonnier_weight: float = 1.0
    edge_weight: float = 0.1
    epsilon: float = 1e-3


class InterpolationLoss(nn.Module):
    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        difference = prediction - target
        charbonnier = torch.sqrt(difference.square() + self.config.epsilon**2).mean()
        prediction_dx, prediction_dy = _gradient(prediction)
        target_dx, target_dy = _gradient(target)
        edge = (prediction_dx - target_dx).abs().mean() + (
            prediction_dy - target_dy
        ).abs().mean()
        total = (
            self.config.charbonnier_weight * charbonnier + self.config.edge_weight * edge
        )
        return {"total": total, "charbonnier": charbonnier, "edge": edge}


def _gradient(frame: Tensor) -> tuple[Tensor, Tensor]:
    return frame[:, :, :, 1:] - frame[:, :, :, :-1], frame[:, :, 1:, :] - frame[:, :, :-1, :]

