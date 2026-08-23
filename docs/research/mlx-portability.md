# MLX portability constraints

Last updated: 2026-08-24

MLX is a real target, not a post-hoc checkbox. PyTorch remains the primary
training environment on the current NVIDIA workstation, while model math and
weight layouts must have a documented MLX path.

## Current platform facts

- MLX 0.32 documents `Conv2d`, `Conv3d`, grouped convolution, transpose
  convolution, pooling, normalization, and common tensor operations.
- MLX convolution uses channels-last (`NHWC`) input, unlike the conventional
  PyTorch VFI implementation layout (`NCHW`). Weight conversion and parity tests
  are therefore mandatory.
- MLX documents bilinear `grid_sample` as a custom Metal-kernel example. This
  proves feasibility, but also means warping is an explicit portability surface,
  not an assumed identical built-in primitive.
- The present workstation is Windows/NVIDIA. Apple Metal execution cannot be
  validated here; MLX parity tests require an Apple Silicon runner later.

Sources: [MLX operations](https://ml-explore.github.io/mlx/build/html/python/ops.html),
[MLX Conv2d](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html),
[custom Metal grid sampling](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
[MLX installation targets](https://ml-explore.github.io/mlx/build/html/install.html).

## Architecture rules

1. The minimal model must have a pure tensor/standard-convolution path.
2. Custom CUDA/CuPy operators cannot be required for correctness.
3. Warping, correlation, and splatting live behind narrow interfaces with
   reference implementations and numerical tests.
4. A forward-splat path may be explored on CUDA, but a backward-gather or
   standard-op alternative must be retained until Metal parity is measured.
5. Full-resolution all-pairs correlation is prohibited. Correlation is local,
   coarse-scale, chunked, or sparse.
6. Shapes and padding behavior are explicit; hidden framework defaults are not
   accepted in checkpoint conversion.
7. Activation functions, normalization, resize modes, and align-corners
   semantics are recorded in the architecture specification.
8. Checkpoints use a framework-neutral name map and `safetensors` where possible.

## Operator risk register

| Operation | PyTorch/CUDA | MLX direction | Risk |
|---|---|---|---|
| 2D/grouped convolution | native | native NHWC | low; weight transpose required |
| bilinear resize | native | standard tensor ops/native resize | low-medium; coordinate parity |
| backward bilinear warp | `grid_sample` | reference ops then custom Metal kernel | medium |
| local correlation | unfold/einsum/custom kernel | reshape/einsum or Metal kernel | medium |
| forward soft splat | custom scatter kernel | custom Metal atomics or fallback | high |
| deformable convolution | extension/operator dependent | custom kernel | high |
| full all-pairs volume | native tensor math but memory heavy | tensor math but memory heavy | high |
| sparse global selection | indexing/top-k | indexing/top-k | medium; benchmark required |

## Planned parity gates

- fixed synthetic inputs and serialized PyTorch outputs;
- layer-level maximum/mean absolute error;
- end-to-end RGB, flow, mask, and residual comparisons;
- FP32 first, then FP16;
- odd sizes and padding boundaries;
- gradient comparison for training-capable MLX modules;
- latency and peak unified-memory measurements on at least two Apple SoCs.

