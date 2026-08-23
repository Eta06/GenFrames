# Orbit: bilateral-flow analytic experiments 001-003

Decision:

- portable warp and target-centric bilateral constraint: **continue**
- unbalanced dense flow supervision: **eliminate**
- current single-velocity field: **revise before real-data promotion**

## Hypothesis

Explicit endpoint alignment should generalize better than the Helix direct
residual candidate. GenFramesBilateralFlow predicts a target-centric velocity
field, constrains target-to-endpoint flows to `-t*v` and `(1-t)*v`, backward
warps both frames, predicts a visibility blend, and adds a bounded RGB residual.
The model uses portable convolutions, bilinear resize, elementwise operations,
and the project's pixel-unit backward-warp contract.

## Reproduction

- Candidate commit: `8e5bf9f`
- Final loss commit and checkpoint code state: `214532f`
- Config: `configs/phase4/bilateral-flow-analytic.yaml`
- Device: RTX 3070, PyTorch 2.8.0+cu126
- Parameters: 549,510
- Training per run: 1,000 steps, batch 16, FP16 autocast, AdamW, LR 0.0002
- Training: 2,048 analytic sequences, seed 123, 64x64
- Validation: 256 disjoint analytic sequences, seed 1,000,123, 64x64
- Final checkpoint: 2,202,352 bytes
- Final checkpoint SHA-256:
  `4289af9ee2478d9047aff7f0a0f07fbb4db7eabf14650d8c6cf995d6fd2dcdd7`

Generated checkpoints live under ignored `artifacts/`; this record contains the
reproduction identity but does not commit binary weights.

## Ablation result

| Run | Flow objective | RGB MAE | RGB PSNR | Decision |
|---|---|---:|---:|---|
| linear blend | none | 0.0181509 | 26.0698 dB | lower bound |
| Helix Mini | none | 0.0179704 | 26.0829 dB | revise |
| Orbit 001 | dense, weight 0.01 | 0.0146128 | 25.3598 dB | eliminate objective |
| Orbit 002 | dense, weight 0.10 | 0.0142499 | 25.7078 dB | eliminate objective |
| Orbit 003 | motion-balanced, weight 0.01 | **0.0129784** | **27.1978 dB** | continue |

Orbit 003 improves over linear blend by 1.1280 dB and reduces MAE by 28.50%.
The result also materially exceeds the 0.0131 dB gain of Helix Mini.

## Why dense flow supervision failed

Only 20.47% of evaluated endpoint-flow pixels are moving. Averaging flow loss
over every pixel rewards the trivial all-zero field. Orbit 001's validation EPE
was 1.0827 px versus 1.0804 px for an all-zero predictor, confirming collapse.
Increasing its scalar weight did not repair the class imbalance.

Orbit 003 normalizes moving and stationary regions separately. Its moving-region
EPE is 4.1460 px versus 5.2792 px for all-zero flow, a 21.47% improvement. It
does leak 0.3878 px EPE into static regions, so aggregate EPE (1.1569 px) is not
yet better than all-zero. The improved rendered-frame metrics show useful
alignment, while the leakage identifies the next ablation target.

## Limits and next action

This synthetic result does not establish real-video quality. The single shared
velocity field assumes locally linear motion and cannot fully represent
occlusion, acceleration, deformation, or multiple hypotheses. Next work adds
mask-aware evaluation, a portable coarse-to-fine correspondence/multi-flow
candidate, then licensed real-data manifests and blinded visual comparisons.
