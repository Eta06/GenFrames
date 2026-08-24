"""Portable local endpoint correlation built from padding, slicing, and reduction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def local_correlation(first: Tensor, second: Tensor, radius: int) -> Tensor:
    """Return dot-product costs for shifts ordered by `(dy, dx)`."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("features must have identical [B, C, H, W] shapes")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    first = functional.normalize(first, dim=1, eps=1e-6)
    second = functional.normalize(second, dim=1, eps=1e-6)
    height, width = first.shape[-2:]
    padded = functional.pad(second, (radius, radius, radius, radius))
    costs = []
    for delta_y in range(-radius, radius + 1):
        top = radius + delta_y
        for delta_x in range(-radius, radius + 1):
            left = radius + delta_x
            shifted = padded[:, :, top : top + height, left : left + width]
            costs.append((first * shifted).sum(dim=1, keepdim=True))
    return torch.cat(costs, dim=1)


def correlation_shift_index(delta_y: int, delta_x: int, radius: int) -> int:
    if not (-radius <= delta_y <= radius and -radius <= delta_x <= radius):
        raise ValueError("shift lies outside correlation radius")
    diameter = 2 * radius + 1
    return (delta_y + radius) * diameter + delta_x + radius
