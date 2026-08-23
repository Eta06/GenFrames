# Prism: coarse velocity refinement 001

Decision: **keep and promote to current best**.

## Hypothesis and isolated change

The full-resolution flow head may lack a direct low-resolution route for large
motion. Add one zero-initialized 1/4-resolution, two-channel velocity correction
head to the existing model. The initial function is exactly the preceding Prism
checkpoint; data order, mix, optimizer, crop, step count, and loss are unchanged.

No correlation volume or multi-flow fusion is included in this experiment.

## Setup and cost

- Initial checkpoint: `prism-bilateral-flow-davis-mixed-001`
- Experiment: `prism-bilateral-flow-coarse-davis-mixed-001`
- Checkpoint SHA-256:
  `716e9c0c50acfdd06d0c5799f84fd57bd89acba058071608b02b527005f83aee`
- Parameters: 550,808 (+1,298 / +0.24%)
- Checkpoint bytes: 2,207,712
- Training: 1,500 steps, batch 4, 256x256, AMP, AdamW, LR 0.0001
- Data: same 75% DAVIS / 25% analytic mix, seed 5,101
- Wall time: 212.25 seconds
- Peak allocated training VRAM: 593,927,680 bytes

## Fixed 300-sample full-resolution DAVIS result

| Candidate | RGB MAE | RGB PSNR | Motion-proxy MAE | Motion-proxy PSNR | Object MAE | Object PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear blend | 0.0721960 | 20.1207 | 0.0934461 | 18.2591 | 0.109902 | 17.1942 |
| Prism mixed baseline | 0.0686962 | 20.1624 | 0.0873207 | 18.4653 | 0.105291 | 17.3307 |
| Prism coarse | **0.0671891** | **20.4075** | **0.0852318** | **18.7135** | **0.102859** | **17.5970** |

The coarse branch gains 0.2450 dB over the frozen real baseline and 0.2868 dB
over linear blend. All reported regional metrics improve, rather than only the
global average.

Full-resolution model-only latency is 20.77 ms median, 21.95 ms p90, and
21.99 ms p99 at batch 1. Peak allocated inference memory is 468,385,792 bytes.
The median cost increase over the preceding checkpoint is about 0.16 ms.

## Synthetic result

Analytic PSNR rises from 33.9081 to 34.4979 dB. Moving-region PSNR rises from
30.1372 to 30.9391 dB, occlusion-region PSNR from 20.9906 to 21.4281 dB, and
moving-flow EPE falls from 3.9495 to 3.1605 px.

## Interpretation and next step

The controlled result supports a coarse-to-fine direction at negligible model
cost and with low MLX operator risk. The gain is still modest in perceptual
terms, so a blinded A/B hard-motion package is required. Multi-flow should be a
separate next ablation, not silently folded into this promoted checkpoint.
