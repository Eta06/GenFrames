"""Deterministic real/synthetic sample mixing with a common training contract."""

from __future__ import annotations

import random
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MixedFrameDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        datasets: Mapping[str, Dataset[dict[str, Tensor | str]]],
        *,
        weights: Mapping[str, float],
        length: int,
        seed: int,
    ) -> None:
        if length <= 0 or not datasets or set(datasets) != set(weights):
            raise ValueError("datasets and weights must have matching non-empty keys")
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("mixture weights must be positive")
        self.datasets = dict(datasets)
        total = sum(weights.values())
        cumulative = 0.0
        self.thresholds: list[tuple[float, str]] = []
        for name, weight in weights.items():
            cumulative += weight / total
            self.thresholds.append((cumulative, name))
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        generator = random.Random(self.seed + index * 15_485_863)
        draw = generator.random()
        source = next(name for threshold, name in self.thresholds if draw <= threshold)
        dataset = self.datasets[source]
        sample_index = generator.randrange(len(dataset))
        sample = dict(dataset[sample_index])
        reference = _tensor(sample, "target")
        height, width = reference.shape[-2:]
        sample.setdefault("flow_t0", torch.zeros((2, height, width), dtype=reference.dtype))
        sample.setdefault("flow_t1", torch.zeros((2, height, width), dtype=reference.dtype))
        sample.setdefault("flow_valid", torch.tensor(source == "synthetic"))
        sample.setdefault("object_mask", torch.zeros((1, height, width), dtype=torch.bool))
        sample.setdefault("moving_mask", torch.zeros((1, height, width), dtype=torch.bool))
        sample["source_kind"] = source
        common_fields = (
            "frame0",
            "frame1",
            "target",
            "time",
            "flow_t0",
            "flow_t1",
            "flow_valid",
            "object_mask",
            "moving_mask",
            "sequence_id",
            "source_kind",
        )
        return {key: sample[key] for key in common_fields}


def _tensor(sample: dict[str, Tensor | str], key: str) -> Tensor:
    value = sample[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"sample field {key!r} must be a tensor")
    return value
