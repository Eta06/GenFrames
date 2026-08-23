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


def batch_masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    _validate_pair(prediction, target)
    mask = _validate_mask(mask, prediction)
    error = (prediction - target).abs().mean(dim=1)
    counts = mask.flatten(1).sum(1)
    values = (error * mask).flatten(1).sum(1) / counts.clamp_min(1)
    return values, counts > 0


def batch_masked_psnr(
    prediction: Tensor, target: Tensor, mask: Tensor, *, data_range: float = 1.0
) -> tuple[Tensor, Tensor]:
    _validate_pair(prediction, target)
    mask = _validate_mask(mask, prediction)
    squared_error = (prediction - target).square().mean(dim=1)
    counts = mask.flatten(1).sum(1)
    mse = (squared_error * mask).flatten(1).sum(1) / counts.clamp_min(1)
    maximum = torch.as_tensor(data_range, device=mse.device, dtype=mse.dtype)
    values = 10.0 * torch.log10(maximum.square() / mse.clamp_min(1e-12))
    return values, counts > 0


def _validate_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or mask.shape != (reference.shape[0], *reference.shape[-2:]):
        raise ValueError("mask must have shape [B, 1, H, W] or [B, H, W]")
    return mask.to(device=reference.device, dtype=reference.dtype)
