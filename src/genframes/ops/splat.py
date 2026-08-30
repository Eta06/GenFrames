"""Reference forward bilinear splatting in pixel displacement units."""

from __future__ import annotations

import torch
from torch import Tensor


def forward_splat(
    source: Tensor, flow: Tensor, *, importance: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Project source pixels along flow and return normalized values plus mass."""
    if source.ndim != 4:
        raise ValueError("source must have shape [B, C, H, W]")
    if flow.shape != (source.shape[0], 2, source.shape[2], source.shape[3]):
        raise ValueError("flow must have shape [B, 2, H, W] matching source")
    if source.device != flow.device or source.dtype != flow.dtype:
        raise ValueError("source and flow must share device and dtype")
    batch, channels, height, width = source.shape
    if importance is None:
        importance = source.new_ones((batch, 1, height, width))
    if importance.shape != (batch, 1, height, width):
        raise ValueError("importance must have shape [B, 1, H, W]")

    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    target_x = x[None] + flow[:, 0]
    target_y = y[None] + flow[:, 1]
    x0 = target_x.floor()
    y0 = target_y.floor()
    output = source.new_zeros((batch, channels, height * width))
    mass = source.new_zeros((batch, 1, height * width))
    source_flat = source.flatten(2)
    importance_flat = importance.flatten(2)
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        destination_x = x0 + offset_x
        destination_y = y0 + offset_y
        weight_x = 1.0 - (target_x - destination_x).abs()
        weight_y = 1.0 - (target_y - destination_y).abs()
        weight = (weight_x * weight_y).clamp_min(0.0)
        valid = (
            (destination_x >= 0)
            & (destination_x < width)
            & (destination_y >= 0)
            & (destination_y < height)
        )
        weight = (weight * valid).flatten(1).unsqueeze(1) * importance_flat
        index = (
            destination_y.clamp(0, height - 1).long() * width
            + destination_x.clamp(0, width - 1).long()
        ).flatten(1).unsqueeze(1)
        mass.scatter_add_(2, index, weight)
        output.scatter_add_(2, index.expand(-1, channels, -1), source_flat * weight)
    mass = mass.view(batch, 1, height, width)
    output = output.view(batch, channels, height, width) / mass.clamp_min(1e-6)
    return output, mass
