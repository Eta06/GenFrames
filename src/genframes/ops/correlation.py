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


def correlation_soft_argmax(costs: Tensor, radius: int, temperature: float = 0.1) -> Tensor:
    """Convert correlation channels into expected `(dx, dy)` feature-pixel shifts."""
    diameter = 2 * radius + 1
    if costs.ndim != 4 or costs.shape[1] != diameter**2:
        raise ValueError("cost channels do not match the requested radius")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifts = torch.arange(-radius, radius + 1, device=costs.device, dtype=costs.dtype)
    delta_y, delta_x = torch.meshgrid(shifts, shifts, indexing="ij")
    coordinates = torch.stack((delta_x.flatten(), delta_y.flatten()), dim=0)
    probabilities = torch.softmax(costs / temperature, dim=1)
    return torch.einsum("bkhw,ck->bchw", probabilities, coordinates)
