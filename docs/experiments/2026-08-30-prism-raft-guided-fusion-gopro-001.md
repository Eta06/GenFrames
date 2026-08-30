# Prism RAFT-guided GoPro fusion 001

## Decision

Keep as an HQ correspondence reference, but do not promote and do not generate a
blind A/B. Global matching passed; the quadratic target-flow and fusion path did not
produce enough perceptual quality.

## Hypothesis and change

The rejected direct-regression students were replaced by frozen TorchVision
RAFT-small global endpoint correspondence. Endpoint flow is converted to two
target-to-source flows with the quadratic Super SloMo approximation. Motion is
fixed; a 550,588-parameter GenFrames U-Net learns only blend visibility and a bounded
RGB residual. Only GenFrames fusion tensors are stored in the checkpoint; RAFT
weights remain an external versioned dependency.

## Setup

- 85% real / 15% analytic; real is 80% GoPro and 20% DAVIS
- GoPro_Large_all sharp frames, CC BY 4.0, 240 FPS, gaps 2/4/8/16
- Crop 256, batch 8, 1500 steps, LR 1e-4, AMP, seed 6101
- Charbonnier + edge; exact synthetic visibility supervises fusion weights
- Wall time 612.07 s; peak training allocation 1,257,854,464 bytes
- Total parameters 1,540,750; trainable/published parameters 550,588

## Results

- DAVIS fixed 64 crops: 26.4441 dB vs linear 24.5860 dB
- Moving-region: 22.0362 dB vs linear 20.0898 dB
- Object-region: 20.9872 dB vs linear 18.8277 dB
- Analytic fixed 64: 35.0468 dB vs linear 32.2033 dB
- Analytic occlusion: 21.0335 dB vs linear 19.0325 dB
- Hard-case prediction: 18.2133 dB vs BilateralFlow 14.4143 dB
- Hard-case alignment: **+2.9618 dB**, positive in 5/6 cases
- Hard cases better than BilateralFlow prediction: 6/6

The global matcher is a real structural gain, not a small metric fluctuation. However,
the trained result is effectively unchanged from the untrained RAFT quadratic time
blend (18.2136 dB in the prior audit). Mean blend-weight spatial standard deviation is
only 0.0126 and the bounded residual changes PSNR by -0.0040 dB. Diagnostic panels
still show severe doubled edges or rubber-sheet artifacts in shooting, drift-straight,
motocross-jump, and bmx-trees. Fusion did not learn useful occlusion selection, while
quadratic reversal itself distorts fast non-linear motion.

The next ablation should change the target motion representation: forward-project
endpoint pixels/flows into target time with collision and hole evidence, rather than
continuing to tune this backward quadratic single-field fusion head.

## Artifacts

- Fusion checkpoint: `D:/GenFrames/checkpoints/prism-raft-guided-fusion-gopro-mixed-001/fusion.safetensors`
- SHA-256: `591f63a66a68607c00aced83237f5b651f95a17eb6cd6cef30ad888903d9fbdc`
- Training result: sibling `result.json`
- Diagnostics: `D:/GenFrames/benchmarks/prism-human-ab-002/diagnostics-raft-guided-fusion-gopro-001/analysis.json`
