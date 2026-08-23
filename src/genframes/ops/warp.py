"""Reference backward bilinear warping in pixel displacement units."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def backward_warp(
    source: Tensor,
    flow: Tensor,
    *,
    padding_mode: str = "border",
) -> Tensor:
    """Sample `source` at target coordinates displaced by `flow`.

    `flow[:, 0]` is horizontal displacement in pixels and `flow[:, 1]` is
    vertical displacement. At target coordinate `(x, y)`, the source is sampled
    at `(x + flow_x, y + flow_y)`. The reference uses `align_corners=False`.
    """
    if source.ndim != 4:
        raise ValueError("source must have shape [B, C, H, W]")
    if flow.shape != (source.shape[0], 2, source.shape[2], source.shape[3]):
        raise ValueError("flow must have shape [B, 2, H, W] matching source")
    if source.device != flow.device or source.dtype != flow.dtype:
        raise ValueError("source and flow must share device and dtype")
    batch, _, height, width = source.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    sample_x = x[None] + flow[:, 0]
    sample_y = y[None] + flow[:, 1]
    normalized_x = (2.0 * sample_x + 1.0) / width - 1.0
    normalized_y = (2.0 * sample_y + 1.0) / height - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    if grid.shape[0] != batch:
        raise AssertionError("constructed grid batch mismatch")
    return functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )

