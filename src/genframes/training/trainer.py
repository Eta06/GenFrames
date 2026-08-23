"""Small explicit trainer used before introducing a larger orchestration stack."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from genframes.models import FrameInterpolator

from .losses import InterpolationLoss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    amp: bool = True
    seed: int = 123
    log_every: int = 50


@dataclass(frozen=True)
class TrainResult:
    steps: int
    initial_loss: float
    final_loss: float
    minimum_loss: float


def train_steps(
    model: FrameInterpolator,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device | str,
    config: TrainConfig,
    criterion: InterpolationLoss | None = None,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    if config.steps <= 0:
        raise ValueError("training steps must be positive")
    _seed_everything(config.seed)
    device = torch.device(device)
    model.to(device).train()
    criterion = (criterion or InterpolationLoss()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    iterator = iter(batches)
    losses: list[float] = []

    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        frame0 = _tensor(batch, "frame0", device)
        frame1 = _tensor(batch, "frame1", device)
        target = _tensor(batch, "target", device)
        time = _tensor(batch, "time", device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(frame0, frame1, time).frame
            components = criterion(prediction, target)
        scaler.scale(components["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        value = float(components["total"].detach())
        losses.append(value)
        if callback is not None and (step == 1 or step % config.log_every == 0):
            callback(
                step,
                {name: float(component.detach()) for name, component in components.items()},
            )

    return TrainResult(
        steps=config.steps,
        initial_loss=losses[0],
        final_loss=losses[-1],
        minimum_loss=min(losses),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor(batch: dict[str, Any], key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device, non_blocking=True)

