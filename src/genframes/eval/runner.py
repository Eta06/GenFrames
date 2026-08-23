"""Framework-neutral-in-spirit PyTorch evaluation loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from genframes.models import FrameInterpolator

from .metrics import batch_mae, batch_masked_mae, batch_masked_psnr, batch_psnr


@dataclass(frozen=True)
class EvaluationResult:
    samples: int
    mae: float
    psnr: float
    parameters: int
    moving_mae: float | None = None
    moving_psnr: float | None = None
    occlusion_mae: float | None = None
    occlusion_psnr: float | None = None
    object_mae: float | None = None
    object_psnr: float | None = None
    flow_epe: float | None = None
    moving_flow_epe: float | None = None


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
    regional: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "moving_mae",
            "moving_psnr",
            "occlusion_mae",
            "occlusion_psnr",
            "object_mae",
            "object_psnr",
            "flow_epe",
            "moving_flow_epe",
        )
    }
    for batch in batches:
        frame0 = _tensor(batch, "frame0", device)
        frame1 = _tensor(batch, "frame1", device)
        target = _tensor(batch, "target", device)
        time = _tensor(batch, "time", device)
        output = model(frame0, frame1, time)
        prediction = output.frame.clamp(0.0, 1.0)
        mae_values = batch_mae(prediction, target)
        psnr_values = batch_psnr(prediction, target)
        mae_total += mae_values.double().cpu().sum()
        psnr_total += psnr_values.double().cpu().sum()
        sample_count += frame0.shape[0]
        moving_mask = _moving_mask(batch, prediction)
        if moving_mask is not None:
            _append_masked(regional, "moving", prediction, target, moving_mask)
        occlusion_mask = _occlusion_mask(batch, prediction)
        if occlusion_mask is not None:
            _append_masked(regional, "occlusion", prediction, target, occlusion_mask)
        object_mask = _optional_tensor(batch, "object_mask", device)
        if object_mask is not None:
            _append_masked(regional, "object", prediction, target, object_mask.bool())
        _append_flow_metrics(regional, output.auxiliary, batch, prediction)
    if sample_count == 0:
        raise ValueError("evaluation received no samples")
    return EvaluationResult(
        samples=sample_count,
        mae=float(mae_total / sample_count),
        psnr=float(psnr_total / sample_count),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        **{name: _mean_or_none(values) for name, values in regional.items()},
    )


def _tensor(batch: dict[str, Any], key: str, device: torch.device | str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device)


def _optional_tensor(
    batch: dict[str, Any], key: str, device: torch.device | str
) -> Tensor | None:
    if key not in batch:
        return None
    return _tensor(batch, key, device)


def _moving_mask(batch: dict[str, Any], reference: Tensor) -> Tensor | None:
    flow0 = _optional_tensor(batch, "flow_t0", reference.device)
    flow1 = _optional_tensor(batch, "flow_t1", reference.device)
    if flow0 is None or flow1 is None:
        return None
    valid = _optional_tensor(batch, "flow_valid", reference.device)
    moving = (flow0.square().sum(1) + flow1.square().sum(1)) > 1e-8
    if valid is not None:
        moving &= valid.bool().flatten()[:, None, None]
    return moving


def _occlusion_mask(batch: dict[str, Any], reference: Tensor) -> Tensor | None:
    visibility0 = _optional_tensor(batch, "visibility0", reference.device)
    visibility1 = _optional_tensor(batch, "visibility1", reference.device)
    if visibility0 is None or visibility1 is None:
        return None
    return ~(visibility0.bool() & visibility1.bool())


def _append_masked(
    metrics: dict[str, list[Tensor]],
    prefix: str,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> None:
    mae, valid = batch_masked_mae(prediction, target, mask)
    psnr, _ = batch_masked_psnr(prediction, target, mask)
    metrics[f"{prefix}_mae"].append(mae[valid].detach().cpu())
    metrics[f"{prefix}_psnr"].append(psnr[valid].detach().cpu())


def _append_flow_metrics(
    metrics: dict[str, list[Tensor]],
    auxiliary: dict[str, Tensor],
    batch: dict[str, Any],
    reference: Tensor,
) -> None:
    if "flow_t0" not in auxiliary or "flow_t1" not in auxiliary:
        return
    target0 = _optional_tensor(batch, "flow_t0", reference.device)
    target1 = _optional_tensor(batch, "flow_t1", reference.device)
    if target0 is None or target1 is None:
        return
    valid = _optional_tensor(batch, "flow_valid", reference.device)
    valid_samples = (
        torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
        if valid is None
        else valid.bool().flatten()
    )
    epe0 = torch.linalg.vector_norm(auxiliary["flow_t0"] - target0, dim=1)
    epe1 = torch.linalg.vector_norm(auxiliary["flow_t1"] - target1, dim=1)
    per_sample = 0.5 * (epe0.flatten(1).mean(1) + epe1.flatten(1).mean(1))
    metrics["flow_epe"].append(per_sample[valid_samples].detach().cpu())
    moving = (target0.square().sum(1) + target1.square().sum(1)) > 1e-8
    moving_counts = moving.flatten(1).sum(1)
    moving_epe = 0.5 * (
        (epe0 * moving).flatten(1).sum(1) + (epe1 * moving).flatten(1).sum(1)
    ) / moving_counts.clamp_min(1)
    usable = valid_samples & (moving_counts > 0)
    metrics["moving_flow_epe"].append(moving_epe[usable].detach().cpu())


def _mean_or_none(values: list[Tensor]) -> float | None:
    nonempty = [value.flatten() for value in values if value.numel()]
    if not nonempty:
        return None
    return float(torch.cat(nonempty).double().mean())
