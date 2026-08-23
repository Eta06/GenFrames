"""Deterministic analytic motion data for pipeline correctness tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _MovingLayer:
    shape: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    velocity_x: float
    velocity_y: float
    color: Tensor


class AnalyticMotionDataset(Dataset[dict[str, Tensor | str]]):
    """Render simple layered motion with exact bilateral target flows.

    This dataset is deliberately small and synthetic. Its purpose is to detect
    broken temporal conditioning, warping conventions, losses, and training
    loops before expensive real data is introduced.
    """

    def __init__(
        self,
        *,
        length: int = 1_024,
        height: int = 64,
        width: int = 64,
        seed: int = 17,
        max_translation: float = 14.0,
        midpoint_only: bool = False,
    ) -> None:
        if length <= 0 or height < 16 or width < 16:
            raise ValueError("length must be positive and spatial dimensions at least 16")
        self.length = length
        self.height = height
        self.width = width
        self.seed = seed
        self.max_translation = max_translation
        self.midpoint_only = midpoint_only
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index * 7_919)
        time = 0.5 if self.midpoint_only else self._sample_time(generator)
        background = self._make_background(generator)
        layers = self._make_layers(generator)

        frame0, owner0 = self._render(background, layers, 0.0)
        target, owner_target = self._render(background, layers, time)
        frame1, owner1 = self._render(background, layers, 1.0)
        flow_t0, flow_t1 = self._target_flows(owner_target, layers, time)
        visibility0 = self._visibility(owner_target, owner0, flow_t0)
        visibility1 = self._visibility(owner_target, owner1, flow_t1)

        return {
            "frame0": frame0,
            "frame1": frame1,
            "target": target,
            "time": torch.tensor(time, dtype=torch.float32),
            "flow_t0": flow_t0,
            "flow_t1": flow_t1,
            "visibility0": visibility0,
            "visibility1": visibility1,
            "sequence_id": f"analytic-{self.seed}-{index}",
        }

    @staticmethod
    def _sample_time(generator: torch.Generator) -> float:
        choices = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        choice = torch.randint(len(choices), (), generator=generator).item()
        return float(choices[choice])

    def _make_background(self, generator: torch.Generator) -> Tensor:
        low_height = max(2, self.height // 8)
        low_width = max(2, self.width // 8)
        noise = torch.rand((1, 3, low_height, low_width), generator=generator)
        background = functional.interpolate(
            noise,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )[0]
        horizontal = self._x / max(self.width - 1, 1)
        vertical = self._y / max(self.height - 1, 1)
        gradient = torch.stack((horizontal, vertical, 1.0 - horizontal))
        return (0.45 * background + 0.35 * gradient + 0.1).clamp(0.0, 1.0)

    def _make_layers(self, generator: torch.Generator) -> tuple[_MovingLayer, ...]:
        count = int(torch.randint(1, 4, (), generator=generator).item())
        layers: list[_MovingLayer] = []
        for _ in range(count):
            minimum_size = min(self.height, self.width) * 0.10
            maximum_size = min(self.height, self.width) * 0.26
            size_x = self._uniform(generator, minimum_size, maximum_size)
            size_y = self._uniform(generator, minimum_size, maximum_size)
            layers.append(
                _MovingLayer(
                    shape="circle" if torch.rand((), generator=generator) < 0.5 else "box",
                    center_x=self._uniform(generator, size_x, self.width - size_x),
                    center_y=self._uniform(generator, size_y, self.height - size_y),
                    size_x=size_x,
                    size_y=size_y,
                    velocity_x=self._uniform(
                        generator, -self.max_translation, self.max_translation
                    ),
                    velocity_y=self._uniform(
                        generator, -self.max_translation, self.max_translation
                    ),
                    color=0.15 + 0.8 * torch.rand((3,), generator=generator),
                )
            )
        return tuple(layers)

    @staticmethod
    def _uniform(generator: torch.Generator, minimum: float, maximum: float) -> float:
        return minimum + (maximum - minimum) * float(torch.rand((), generator=generator))

    def _render(
        self, background: Tensor, layers: tuple[_MovingLayer, ...], time: float
    ) -> tuple[Tensor, Tensor]:
        frame = background.clone()
        owner = torch.zeros((self.height, self.width), dtype=torch.long)
        for layer_id, layer in enumerate(layers, start=1):
            center_x = layer.center_x + time * layer.velocity_x
            center_y = layer.center_y + time * layer.velocity_y
            normalized_x = (self._x - center_x) / layer.size_x
            normalized_y = (self._y - center_y) / layer.size_y
            if layer.shape == "circle":
                mask = normalized_x.square() + normalized_y.square() <= 1.0
            else:
                mask = (normalized_x.abs() <= 1.0) & (normalized_y.abs() <= 1.0)
            frame[:, mask] = layer.color[:, None]
            owner[mask] = layer_id
        return frame, owner

    def _target_flows(
        self, owner: Tensor, layers: tuple[_MovingLayer, ...], time: float
    ) -> tuple[Tensor, Tensor]:
        velocity = torch.zeros((2, self.height, self.width), dtype=torch.float32)
        for layer_id, layer in enumerate(layers, start=1):
            mask = owner == layer_id
            velocity[0, mask] = layer.velocity_x
            velocity[1, mask] = layer.velocity_y
        return -time * velocity, (1.0 - time) * velocity

    def _visibility(self, target_owner: Tensor, source_owner: Tensor, flow: Tensor) -> Tensor:
        source_x = (self._x + flow[0]).round().long()
        source_y = (self._y + flow[1]).round().long()
        inside = (
            (source_x >= 0)
            & (source_x < self.width)
            & (source_y >= 0)
            & (source_y < self.height)
        )
        sampled_owner = torch.full_like(target_owner, -1)
        sampled_owner[inside] = source_owner[source_y[inside], source_x[inside]]
        return (inside & (sampled_owner == target_owner)).float().unsqueeze(0)

