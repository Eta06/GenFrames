# Parallax RAFT region assignment 001

## Decision

Reject and do not promote. No blind A/B is warranted. The ablation answers the
ownership questions, but hard assignment replaces soft ghosting with more severe
field-switch seams and fragmented fallback artifacts on real motion.

## Hypothesis and controlled change

Keep the exact accepted ten-field representation, RAFT oracle, training data,
crop, seed, and training length. Replace dense ten-way RGB blending with an
8x8-region assignment over ten hypotheses plus one abstain class. The deployed
forward pass selects exactly one candidate; soft probabilities exist only for a
straight-through gradient. Abstention selects the pixelwise field with the best
cycle/confidence evidence. No RGB residual or flow update is permitted.

## Training setup

- 85% real / 15% analytic; real is 80% GoPro and 20% DAVIS
- 256 crop, batch 4, 1500 steps, LR 1e-4, seed 7201, AMP
- Hard best-field/unsupported supervision, edge-aware assignment coherence, and
  exact synthetic endpoint-ownership supervision
- Wall time 705.53 s; peak training allocation 633,798,144 bytes
- 86,635 trainable parameters; checkpoint excludes frozen RAFT weights

## Results

| Set/region | Single | Dense evidence | Hard region |
|---|---:|---:|---:|
| DAVIS 64 | 25.4438 dB | 25.7970 dB | 24.7526 dB |
| GoPro test 24 | 39.4153 dB | 39.5137 dB | 37.9422 dB |
| Hard DAVIS 6 | 18.2136 dB | 18.4820 dB | 17.7561 dB |
| Hard high-disagreement ghost proxy | 0.05847 | 0.05754 | 0.10951 |
| Analytic occlusion MAE | 0.06800 | 0.06496 | 0.05811 |
| Analytic occlusion ghost proxy | 0.10950 | 0.11066 | 0.12889 |

Hard-case aggregate MAE oracle-gap closure is -2.72%; none of the six hard cases
beats the single-field PSNR. Motocross falls from 17.8649 dB single-field and
17.5913 dB dense to 17.2970 dB. Its high-disagreement ghost proxy is 0.07323.

## Assignment and ownership diagnosis

- Hard output entropy is exactly zero, but pre-decision entropy remains 1.527.
- Low-image-edge neighbor consistency is 0.839, worse than dense-selector argmax
  consistency at 0.896. Thus the same object/region did not become more coherent.
- Mean dominant assignment coverage inside DAVIS objects is only 60.25%.
- Abstain is used on 92.66% of hard DAVIS pixels and 91.65% of its
  high-disagreement pixels. The fallback improves MAE over forced no-abstain by
  only 0.00226, far too little to offset its fragmented field switching.
- Exact analytic disocclusion ownership accuracy reaches 67.57%, and occlusion MAE
  improves. However, boundary ghosting worsens because independently chosen hard
  warps are not composited continuously.

The key failure is structural, not a temperature/channel issue. The region head
mostly learns to abstain; pixelwise fallback then bypasses region coherence and
creates discontinuities wherever its winning field changes. Forced region choices
also show blocky or torn surfaces, so merely lowering abstain usage is not enough.

Visual inspection of all six panels shows severe tearing on the shooting subject,
car, motocross bike, and cyclist. The car case is substantially worse than both
single and dense outputs. Metric and visual gates both fail.

## Cost and artifacts

- 256x256 latency: 95.74 ms hard region, 91.54 ms dense, 72.23 ms single
- Peak 256x256 allocation: 141,666,816 bytes
- Checkpoint: `D:/GenFrames/checkpoints/parallax-raft-region-assignment-gopro-001/selector.safetensors`
- SHA-256: `6bb987e262a74f0b0ae62f5731bb86688098b24b7650d65129987adfa8f8a634`
- Audit: `D:/GenFrames/benchmarks/parallax-region-assignment-audit-001/result.json`
- Visual panels: `D:/GenFrames/benchmarks/parallax-region-assignment-audit-001/case-01.png`
  through `case-06.png`

## Next implication

Keep the representation, but do not repeat raw K-way hard field assignment or tune
its stride/threshold. The next structural candidate should separate endpoint
ownership from within-surface correspondence: a layered source-ownership mask with
a narrow learned boundary/compositing band, plus a region-scoped fallback that
cannot reintroduce pixelwise field fragmentation. This retains discrete ownership
away from boundaries while allowing continuity where two warped surfaces meet.
