# Helix: GenFramesMini analytic experiment 001

Decision:

- training/checkpoint pipeline: **continue**
- flow-free blend-residual candidate: **revise; do not promote**

## Hypothesis

A small time-conditioned encoder/decoder may learn a useful bounded correction
to linear blending without explicit flow. The same run tests CUDA AMP, arbitrary
time conditioning, validation isolation, and safetensors checkpoint output.

## Reproduction

- Git commit: `3e57f14c9c7a134a2fb1cfed0d7e17bd2b06bd6d`
- Config: `configs/phase3/mini-analytic.yaml`
- Device: RTX 3070, PyTorch 2.8.0+cu126
- Model parameters: 549,076
- Training: 1,000 steps, batch 16, FP16 autocast, AdamW, LR 0.0002
- Training data: 2,048 analytic sequences, seed 123, 64x64
- Validation: 256 disjoint analytic sequences, seed 1,000,123, 64x64
- Wall time including baseline/final evaluation and on-the-fly CPU rendering:
  approximately 75.8 seconds
- Checkpoint size: 2,200,616 bytes
- Checkpoint SHA-256:
  `2baf31ba31042b5b4a179e87729273d631f92cf0d020ab10f77fde5eef723562`

## Generalization result

| Candidate | Parameters | RGB MAE | RGB PSNR |
|---|---:|---:|---:|
| linear blend | 0 | 0.0181509 | 26.0698 dB |
| GenFramesMini | 549,076 | 0.0179704 | 26.0829 dB |

The 0.0131 dB gain is too small to justify the model. Batch loss is not directly
monotonic because batches have different motion difficulty; initial/final batch
loss therefore does not establish convergence. Validation is authoritative.

## Pipeline-separation overfit test

A one-sample overfit run used seed 321, 1,000 steps, LR 0.0005, and the same
architecture:

| State | RGB MAE | RGB PSNR |
|---|---:|---:|
| linear blend | 0.00371870 | 33.5490 dB |
| trained model | 0.000353699 | 49.4733 dB |

Training loss fell from 0.0051105 to 0.00131865 (minimum 0.00127457). This proves
the optimizer, gradient, target-time, and checkpoint-era model path can learn.
It does not prove useful distribution-level interpolation.

## Interpretation and next action

The direct residual model can memorize a motion case but does not generalize
enough to replace explicit alignment. Static background also dominates global
metrics. The next candidate will add a portable backward warp and learned
bilateral target flows. Evaluation will add masks for moving and occlusion pixels
before comparing architectures.

