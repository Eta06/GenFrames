"""Framework-neutral-in-spirit PyTorch evaluation loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from genframes.models import FrameInterpolator

from .metrics import batch_mae, batch_psnr


@dataclass(frozen=True)
class EvaluationResult:
    samples: int
    mae: float
    psnr: float
    parameters: int


@torch.inference_mode()
def evaluate_model(
    model: FrameInterpolator,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device | str,
) -> EvaluationResult:
    model = model.to(device).eval()
    mae_total = torch.zeros((), dtype=torch.float64)
    psnr_total = torch.zeros((), dtype=torch.float64)
    sample_count = 0
    for batch in batches:
        frame0 = _tensor(batch, "frame0", device)
        frame1 = _tensor(batch, "frame1", device)
        target = _tensor(batch, "target", device)
        time = _tensor(batch, "time", device)
        prediction = model(frame0, frame1, time).frame.clamp(0.0, 1.0)
        mae_values = batch_mae(prediction, target)
        psnr_values = batch_psnr(prediction, target)
        mae_total += mae_values.double().cpu().sum()
        psnr_total += psnr_values.double().cpu().sum()
        sample_count += frame0.shape[0]
    if sample_count == 0:
        raise ValueError("evaluation received no samples")
    return EvaluationResult(
        samples=sample_count,
        mae=float(mae_total / sample_count),
        psnr=float(psnr_total / sample_count),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def _tensor(batch: dict[str, Any], key: str, device: torch.device | str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device)
