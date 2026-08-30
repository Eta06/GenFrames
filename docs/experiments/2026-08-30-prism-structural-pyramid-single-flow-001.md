# Prism structural pyramid single-flow 001

## Decision

Rejected. Do not promote and do not continue with radius, channel-count, or
single-flow pyramid tweaks. No blind A/B was created because the hard-case
alignment gate failed and visible ghosting remained severe.

## Hypothesis

An explicit 1/16 -> 1/8 -> 1/4 -> 1/2 feature pyramid could correct the frozen
baseline's target-centric interval velocity over larger displacements. Warping
both endpoint features toward the target before each local correlation was
expected to make the remaining displacement locally searchable.

## Change

- Added a shared endpoint feature pyramid with channels `[12, 16, 24, 32]`.
- Used bidirectional local cost volumes with radii `[2, 2, 2, 4]`.
- Propagated one target-centric interval-velocity correction coarse-to-fine.
- Zero-initialized every update head and moment gate so the initial function
  exactly matched `prism-coarse-oracle-warp-davis-mixed-001`.
- Kept RGB warp, visibility/blend, and residual fusion architecture unchanged.

## Training setup

- Initial checkpoint: `prism-coarse-oracle-warp-davis-mixed-001`
- Data: 75% DAVIS 2017 / 25% analytic exact-flow
- Crop / batch: 256 x 256 / 4
- Steps / LR / seed: 1500 / 1e-4 / 5101
- Loss: bilateral flow 0.01, static flow 0.1, warp oracle 0.2
- GPU: NVIDIA RTX 3070, AMP
- Wall time: 582.24 seconds

## Results

| Measure | Oracle-warp baseline | Structural pyramid |
|---|---:|---:|
| Parameters | 550,808 | 777,716 |
| DAVIS-300 PSNR | 20.6103 dB | 21.2305 dB |
| DAVIS-300 MAE | 0.06567 | 0.06207 |
| Median latency, batch 1 | 20.76 ms | 53.02 ms |
| Peak evaluation VRAM | 468.4 MB | 477.8 MB |
| Synthetic-256 PSNR | 35.1296 dB | 37.5781 dB |
| Synthetic moving flow EPE | not recorded | 2.2260 px |
| Hard-case alignment gain | -0.4548 dB | **-0.8940 dB** |

The global metric improved, but correspondence became worse on every sealed
hard case:

| Case | Sequence | Alignment delta vs baseline |
|---|---|---:|
| 01 | shooting | -0.2790 dB |
| 02 | lab-coat | -1.0024 dB |
| 03 | drift-straight | -0.2290 dB |
| 04 | motocross-jump | -0.4451 dB |
| 05 | loading | -0.4463 dB |
| 06 | bmx-trees | -0.2314 dB |

## Failure interpretation

The velocity visualization is spatially noisy rather than piecewise coherent.
Endpoint warps rubber-sheet the scene and create double contours or translucent
copies around people, vehicles, wheels, tree trunks, and high-parallax
backgrounds. Lower warp disagreement does not represent better alignment here;
both endpoint warps are often distorted in a similarly wrong direction.

The shared constant-velocity field is an inadequate representation at
occlusion boundaries, non-rigid motion, high parallax, and independently moving
layers. Local matching trained from scratch on this data also lacks a strong
global correspondence prior. Fusion is not the first bottleneck: the GT oracle
cannot recover a good target from the malformed warps.

## Next action

End this single target-centric flow line. Re-evaluate research around independent
target-to-endpoint bilateral flows, global/transformer matching, and teacher
flow distillation. The next ablation must change the motion representation or
matching supervision, not make a small tweak to this pyramid.

## Artifact

- Checkpoint: `D:/GenFrames/checkpoints/prism-structural-pyramid-single-flow-davis-mixed-001/model.safetensors`
- SHA-256: `202eddefdf60bc3e4a8664004fb7de0a8b3f37d0c2da28accd4247990b7513f5`
- DAVIS-300: `D:/GenFrames/benchmarks/prism-structural-pyramid-single-flow-davis-300-001/result.json`
- Diagnostics: `D:/GenFrames/benchmarks/prism-human-ab-explicit-displacement-001/diagnostics-structural-pyramid-single-flow-001/analysis.json`
