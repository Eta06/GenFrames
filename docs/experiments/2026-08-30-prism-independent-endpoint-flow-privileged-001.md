# Prism independent endpoint flow privileged 001

## Decision

Not promoted. The representation is retained as a research direction, but the
single full-resolution prediction head is rejected. No blind A/B was created.

## Hypothesis and change

Replace the shared constant-velocity constraint with independent target-to-input
flows while keeping RGB warp, blend, and residual fusion unchanged. Train the
551,676-parameter student with exact analytic flow and training-only RAFT-small
target-to-endpoint pseudo-flow on real triplets. Task-oriented masks retain only
teacher pixels whose warp is better than the raw endpoint and below 0.15 MAE.

## Training setup

- Initial checkpoint: `prism-coarse-oracle-warp-davis-mixed-001`
- Data: 75% DAVIS / 25% analytic, 256 crops, batch 4
- Steps / LR / seed: 1500 / 1e-4 / 5101
- Loss: flow 0.01, warp oracle 0.2, unchanged RGB/edge losses
- Teacher: TorchVision RAFT-small C_T_V2, training only
- Wall time / peak training VRAM: 326.07 s / 646.4 MB

## Result

- DAVIS-300: 20.9132 dB PSNR, 0.06412 MAE
- Synthetic-256: 34.8704 dB PSNR, moving flow EPE 3.0037 px
- Median inference latency: 26.30 ms
- Hard-case aggregate alignment: -0.1514 dB
- Checkpoint SHA-256: `c95d8d6c0462992675c7fdae7396a97102bba47932a2df06a1f09f2c563127a9`

Compared with the oracle-warp baseline, alignment improved on lab-coat,
drift-straight, motocross-jump, and loading, but regressed slightly on shooting
and bmx-trees. Loading reached +1.00 dB absolute alignment. The acceptance gate
still failed because aggregate alignment remained negative and visible ghosting
remained severe.

Student hard-case motion p95 stayed near 21-25 px while the privileged teacher
required up to 231 px. The independent representation helped, but a shallow
decoder head did not learn large correspondence. The next controlled ablation
must change the student matcher to wide-receptive-field coarse-to-fine updates;
changing flow bounds, loss weights, or head channels is not justified.

## Artifacts

- Checkpoint: `D:/GenFrames/checkpoints/prism-independent-endpoint-flow-privileged-davis-mixed-001/model.safetensors`
- DAVIS-300: `D:/GenFrames/benchmarks/prism-independent-endpoint-flow-privileged-davis-300-001/result.json`
- Diagnostics: `D:/GenFrames/benchmarks/prism-human-ab-explicit-displacement-001/diagnostics-independent-endpoint-privileged-001/analysis.json`
