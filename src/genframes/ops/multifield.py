"""Target-time flow hypotheses derived from a source-to-source correspondence field."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from .warp import backward_warp


def inverse_flow_hypotheses(
    endpoint_flow: Tensor,
    fraction: Tensor | float,
    *,
    refinement_steps: Sequence[int] = (0, 1, 2, 4),
    damping: float = 0.75,
) -> tuple[Tensor, ...]:
    """Approximate multiple target-to-source fields along fixed-point refinement.

    The endpoint flow maps a source pixel to the opposite endpoint. ``fraction``
    specifies how far that source pixel travels toward target time. Each retained
    iterate is an independently warpable field, rather than being fused here.
    """
    if endpoint_flow.ndim != 4 or endpoint_flow.shape[1] != 2:
        raise ValueError("endpoint_flow must have shape [B, 2, H, W]")
    steps = tuple(sorted(set(refinement_steps)))
    if not steps or steps[0] < 0:
        raise ValueError("refinement_steps must contain non-negative integers")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie inside (0, 1]")
    if not isinstance(fraction, Tensor):
        fraction = endpoint_flow.new_tensor(float(fraction))
    fraction = fraction.to(device=endpoint_flow.device, dtype=endpoint_flow.dtype)
    if fraction.ndim == 0:
        fraction = fraction.view(1, 1, 1, 1)
    elif fraction.ndim == 1:
        fraction = fraction[:, None, None, None]
    state = -fraction * endpoint_flow
    hypotheses = []
    if 0 in steps:
        hypotheses.append(state)
    for step in range(1, steps[-1] + 1):
        sampled = backward_warp(endpoint_flow, state)
        proposal = -fraction * sampled
        state = (1.0 - damping) * state + damping * proposal
        if step in steps:
            hypotheses.append(state)
    return tuple(hypotheses)
