# Prism: oracle-warp supervision 001

Decision: **keep the loss; do not claim perceptual adequacy; proceed to explicit
single-flow correspondence, not multi-flow**.

## Hypothesis and isolated change

The hard-case diagnosis showed that predicted warps are worse than raw endpoints.
Add a per-pixel minimum endpoint-warp reconstruction loss so at least one warp
must explain the real GT. Architecture, initialization, data order, mix, crop,
optimizer, and training length remain fixed. Loss weight is 0.2.

## Setup

- Initial checkpoint: `prism-bilateral-flow-coarse-davis-mixed-001`
- Experiment: `prism-coarse-oracle-warp-davis-mixed-001`
- Checkpoint SHA-256:
  `c6ad55d210ff33a824fa87c3f8bb80228d36756641b9d56fc3b05243e53f491b`
- Parameters/checkpoint bytes: 550,808 / 2,207,712
- Training: 1,500 steps, batch 4, 256x256, AMP, LR 0.0001
- Data: 75% DAVIS / 25% analytic, seed 5,101
- Wall time: 253.79 seconds
- Peak allocated training VRAM: 593,928,192 bytes

## Fixed full-resolution DAVIS-300

| Candidate | RGB MAE | RGB PSNR | Motion PSNR | Object PSNR |
|---|---:|---:|---:|---:|
| coarse baseline | 0.0671891 | 20.4075 | 18.7135 | 17.5970 |
| oracle-warp loss | **0.0656737** | **20.6103** | **18.9005** | **17.8482** |

The loss adds 0.2028 dB globally with no parameter or measurable latency cost.
Analytic PSNR rises from 34.4979 to 35.1296 dB and moving-flow EPE falls from
3.1605 to 2.6707 px.

## Six prior both-bad cases

| Diagnostic | Coarse baseline | Oracle-warp loss |
|---|---:|---:|
| final PSNR | 14.5274 dB | 14.6532 dB |
| warp-oracle PSNR | 16.6061 dB | 16.6699 dB |
| flow alignment gain | -0.5186 dB | -0.4548 dB |
| fusion opportunity | +2.0787 dB | +2.0166 dB |
| high warp disagreement | 54.90% | 54.59% |

The diagnostic-targeted loss moves every aggregate in the intended direction,
but flow alignment remains negative and the outputs remain far below the oracle.
It therefore helps without resolving the primary failure.

## Next architecture decision

Do not add multiple bad flow hypotheses. The next controlled candidate must add
an explicit endpoint correspondence signal at coarse resolution while retaining
one bilateral motion hypothesis and oracle-warp supervision. A small local
correlation/cost-volume route is preferred before full all-pairs correlation;
its displacement coverage, VRAM, latency, and MLX operator mapping must be
reported. Multi-flow becomes justified only after the single-flow warp oracle
beats the unwarped oracle on the hard slice.
