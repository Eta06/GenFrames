# ADR 0001: Phase 0 experiment directions

Status: accepted for experimentation

Date: 2026-08-24

## Context

GenFrames must become an original model rather than a wrapper or renamed fork.
The literature supports several incompatible directions: explicit flow, adaptive
kernels, flow-free spatio-temporal convolution, multi-hypothesis flow, selective
global matching, and generative/diffusion interpolation. Selecting one before a
shared data and evaluation harness would turn literature preference into an
unmeasured product decision.

## Decision

Phase 0 does not select a final architecture. Phase 2/3 will create comparable
small implementations for:

1. direct blend-residual prediction;
2. portable bilateral intermediate flow;
3. multi-hypothesis flow with visibility/confidence fusion;
4. coarse selective global matching over a local-flow backbone.

The minimal model proves training correctness before teacher supervision,
specialized operators, perceptual losses, or a large backbone are introduced.

## Shared contracts

- Inputs are two RGB frames in `[0, 1]` plus target time `t` in `(0, 1)`.
- Output is RGB at the same spatial resolution and a structured dictionary of
  optional motion, mask, confidence, and residual tensors.
- `t` is batched and explicit; midpoint-only hidden assumptions are forbidden.
- Spatial dimensions may be odd. Padding and crop-back behavior are model-owned.
- Modules with backend risk (warp, correlation, splat) use reference interfaces.
- Parameter names and shapes must support deterministic framework conversion.

## Consequences

This adds early implementation work but allows evidence-based elimination and
keeps MLX portability measurable. Diffusion remains a possible teacher/HQ branch,
not a dependency of the first correctness proof.

