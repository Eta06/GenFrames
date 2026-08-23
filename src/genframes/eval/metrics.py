"""Precisely defined lightweight metrics used in smoke experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have identical [B, C, H, W] shapes")


def batch_mae(prediction: Tensor, target: Tensor) -> Tensor:
    _validate_pair(prediction, target)
    return (prediction - target).abs().flatten(1).mean(1)


def batch_psnr(prediction: Tensor, target: Tensor, *, data_range: float = 1.0) -> Tensor:
    _validate_pair(prediction, target)
    mse = (prediction - target).square().flatten(1).mean(1)
    maximum = torch.as_tensor(data_range, device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10(maximum.square() / mse.clamp_min(1e-12))

