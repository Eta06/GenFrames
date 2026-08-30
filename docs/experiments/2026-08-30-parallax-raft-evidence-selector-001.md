# Parallax RAFT evidence selector 001

## Decision

Selector rejected and not promoted. Keep the ten-field representation and do not
create a blind A/B: metric gains do not produce a clear visual ghosting reduction.

## Hypothesis and change

Keep the exact ten fixed-point candidate fields from the accepted representation
audit. Add per-candidate photometric cycle, inverse residual, bidirectional flow
cycle, source/cross validity, and consensus-deviation evidence. Predict a convex
selector at half resolution with dilated context and edge-aware spatial coherence.
RAFT remains a frozen research oracle; there is no RGB residual or flow update.

## Training setup

- 85% real / 15% analytic; real is 80% GoPro and 20% DAVIS
- 256 crop, batch 4, 1500 steps, LR 1e-4, seed 7201, AMP
- Soft best-candidate supervision plus edge-aware spatial selector loss
- Wall time 782.35 s; peak training allocation 636,454,400 bytes
- 104,842 trainable selector parameters; selector checkpoint excludes RAFT

## Results

| Set/region | Single field | Evidence selector | Delta |
|---|---:|---:|---:|
| DAVIS 64 | 25.4438 dB | 25.7970 dB | +0.3531 dB |
| DAVIS moving | 21.4943 dB | 21.9006 dB | +0.4063 dB |
| DAVIS object | 20.8956 dB | 21.1045 dB | +0.2088 dB |
| GoPro test 24 | 39.4153 dB | 39.5137 dB | +0.0984 dB |
| Analytic global 64 | 34.1638 dB | 34.4066 dB | +0.2428 dB |
| Analytic occlusion | 20.5917 dB | 21.6111 dB | +1.0194 dB |

On the six sealed DAVIS hard cases, PSNR improves 18.2136 -> 18.4820 dB
and five of six cases improve. However, only 8.86% of the aggregate MAE oracle
gap and 7.03% of the PSNR oracle gap are closed. High-disagreement MAE improves
0.13179 -> 0.12174, but the Laplacian ghosting proxy changes only
0.05847 -> 0.05754 (1.59%). Motocross still regresses.

The learned exact top-1 accuracy is 11.95% on hard DAVIS, versus 11.25% for the
raw confidence prior. With tied/near-equivalent candidates accepted within
1/255 MAE of the oracle, learned accuracy is 33.45%, slightly worse than the
prior's 34.12%. DAVIS and GoPro show the same global pattern: the network often
improves a soft blend without reliably choosing the best candidate.

Exact analytic occlusion exposes a real but narrow learned effect. Tolerance
accuracy improves from 21.03% (confidence prior) to 57.25%; disocclusion improves
20.57% -> 57.51%, and MAE improves 0.06800 -> 0.06496. Yet the occlusion edge
proxy worsens 0.10950 -> 0.11066, analytic high-disagreement error regresses,
and aggregate analytic MAE oracle-gap closure is -20.69%.

Visual inspection of all six evidence panels confirms the gate failure. Evidence
maps respond to object boundaries and invalid warps, but field maps remain noisy,
entropy is high (2.20 vs ten-way maximum 2.303), and drift/motocross retain strong
double edges and smeared structure. The improvement is not obvious enough for a
blind A/B.

## Cost and artifacts

- 256x256 latency: 89.64 ms selector vs 72.32 ms single-field
- Peak 256x256 inference allocation: 122,864,128 bytes vs 112,514,048 bytes
- Checkpoint: `D:/GenFrames/checkpoints/parallax-raft-evidence-selector-gopro-001/selector.safetensors`
- SHA-256: `3ec3bc1232840e1892f0b48557a167ee06da91437587bd193d4b1e6590c238fc`
- Audit: `D:/GenFrames/benchmarks/parallax-evidence-selector-audit-003/result.json`
- Evidence panels: `D:/GenFrames/benchmarks/parallax-evidence-selector-audit-003/case-01.png` through `case-06.png`

## Next implication

Do not spend another run on temperature/channel tweaks. The frozen representation
still has substantial oracle margin, but a dense ten-way soft blend is not turning
it into coherent visible surfaces. The next ablation should preserve the same
candidates while changing the selector target to layered visibility/ownership or
region-level assignment, with explicit boundary supervision and an abstain/fallback
path for unsupported pixels.
