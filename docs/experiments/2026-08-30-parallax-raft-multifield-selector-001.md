# Parallax RAFT multi-field selector 001

## Decision

Selector rejected and not promoted. Retain the multi-field representation result.
Do not create a blind A/B.

## Hypothesis and change

Freeze the ten fields proven by the representation audit. Train only an
87,562-parameter spatial CNN that produces a convex softmax over candidate images.
No flow update, RGB residual, or generative refinement is allowed. Zero
initialization reproduces the prior quadratic time blend.

## Setup

- 85% real / 15% analytic; real is 80% GoPro and 20% DAVIS
- 256 crop, batch 4, 1500 steps, LR 1e-4, seed 7201, AMP
- Charbonnier + edge + conflict-balanced best-candidate classification
- Wall time 269.16 s; peak training allocation 639,750,656 bytes
- Selector checkpoint excludes RAFT weights

## Validation

| Set/region | Single field | Multi selector | Delta |
|---|---:|---:|---:|
| DAVIS 64 | 25.4438 dB | 25.7445 dB | +0.3007 dB |
| DAVIS moving | 21.4943 dB | 21.8315 dB | +0.3372 dB |
| DAVIS object | 20.8956 dB | 21.0956 dB | +0.2000 dB |
| GoPro test 24 | 39.4153 dB | 39.3859 dB | -0.0294 dB |
| Analytic global 64 | 34.1638 dB | 33.9311 dB | -0.2327 dB |
| Analytic occlusion | 20.5917 dB | 21.4005 dB | +0.8088 dB |

On the six sealed hard cases, aggregate PSNR improves 18.2136 -> 18.4208 dB
and five of six cases improve. High-disagreement MAE improves 0.13648 -> 0.13197;
the Laplacian edge-ghosting proxy improves 0.05831 -> 0.05428. Motocross regresses
by 0.4381 dB, however, and visual ghosting remains obvious.

The selector assigns 55.4% mean weight to extra fields but retains high entropy
(1.962 vs a ten-way maximum of 2.303). A post-hoc temperature/hard-selection audit
rejects the idea that simple sparsification fixes this: temperature 0.5 and 0.25
both regress, while hard argmax collapses aggregate PSNR to 16.9157 dB and worsens
the edge proxy to 0.07833.

The candidate set is demonstrably better, but local RGB-only class selection cannot
recover enough of its oracle margin. The next selector must receive explicit
candidate confidence/cycle-consistency evidence and impose spatially coherent
selection. Repeating temperature, entropy, or channel-count tweaks is not justified.

## Cost and artifacts

- Native 480x1152 hard-case latency: 145.72 ms median
- Peak native inference allocation: 1,622,820,864 bytes
- Checkpoint: `D:/GenFrames/checkpoints/parallax-raft-multifield-selector-gopro-001/selector.safetensors`
- SHA-256: `1a4dc0695f67a306153aef3c5cec29c51f1505d7ef92bf58b12c5b63320e38e3`
- Hard diagnostics: `D:/GenFrames/benchmarks/parallax-multifield-selector-hard-001/result.json`
- Sparsity audit: `D:/GenFrames/benchmarks/parallax-selector-sparsity-audit-001/result.json`
