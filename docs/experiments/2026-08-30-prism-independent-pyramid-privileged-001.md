# Prism independent pyramid privileged 001

## Decision

Rejected for perceptual quality. Do not promote and do not generate blind A/B.
End the small direct-regression matcher line.

## Hypothesis and change

Replace the shallow independent-flow head with 1/8, 1/4, and 1/2 update blocks.
Dilations 1/2/4/8 provide a wide coarse receptive field; feature-pixel update
limits 32/8/4 allow large displacement. Independent target-to-endpoint flows,
training-only RAFT-small supervision, and the unchanged fusion path were kept.

## Setup

- Initial checkpoint: `prism-coarse-oracle-warp-davis-mixed-001`
- 791,012 parameters
- 75% DAVIS / 25% analytic, 256 crop, batch 4
- 1500 steps, LR 1e-4, seed 5101, AMP
- Wall time 347.84 s; peak training VRAM 667.6 MB

## Results

- DAVIS-300: 20.8225 dB, 0.06379 MAE
- Synthetic-256: 33.9286 dB, moving flow EPE 3.1753 px
- Median inference latency: 36.01 ms
- Hard-case aggregate alignment: **+0.1241 dB**
- Hard cases improved over oracle-warp baseline: 6/6
- Absolute positive alignment: 2/6

The numeric alignment gate passed for the first time, but the visual gate did
not. Predictions still contain obvious translucent duplicate people, vehicles,
wheels, and backgrounds. Four cases retain negative absolute alignment. Global
DAVIS-300 quality also regressed relative to the shallower independent-flow
student.

This is not meaningful perceptual progress and will not be sent to human A/B.
Wider direct convolutional regression is no longer justified. The teacher audit
shows that correct correspondence exists, but the student requires stronger
pretrained/global matching and substantially broader real-triplet training data.

## Artifacts

- Checkpoint: `D:/GenFrames/checkpoints/prism-independent-pyramid-privileged-davis-mixed-001/model.safetensors`
- SHA-256: `a7c5319a544f5fab5fc487cbcd08bb62739258983e64c187ada1c8871206a214`
- DAVIS-300: `D:/GenFrames/benchmarks/prism-independent-pyramid-privileged-davis-300-001/result.json`
- Diagnostics: `D:/GenFrames/benchmarks/prism-human-ab-explicit-displacement-001/diagnostics-independent-pyramid-privileged-001/analysis.json`
