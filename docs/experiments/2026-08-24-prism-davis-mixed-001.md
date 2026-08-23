# Prism: DAVIS mixed fine-tune 001

Decision: **keep as current real-video baseline; architecture still requires revision**.

## Hypothesis and change

The Orbit checkpoint learned useful alignment on exact-flow synthetic motion but
failed to transfer to natural video. Fine-tuning the same architecture on a
75% DAVIS / 25% analytic mixture tests domain adaptation without conflating it
with an architecture change. Synthetic samples retain exact bilateral-flow
supervision; real samples contribute image and edge losses only.

## Setup

- Initial checkpoint: `orbit-bilateral-flow-analytic-003`
- Experiment/checkpoint: `prism-bilateral-flow-davis-mixed-001`
- Checkpoint SHA-256:
  `974b1402c33938617ccbf4c13bbb4ad031425cadc524c699d8facef19676e7f2`
- Parameters/checkpoint bytes: 549,510 / 2,202,352
- Training: 1,500 steps, batch 4, 256x256, AMP, AdamW, LR 0.0001
- Mix: 75% DAVIS train, 25% analytic exact-flow, seed 5,101
- Device: RTX 3070 8 GB
- Wall time: 230.43 seconds
- Peak allocated VRAM: 589,190,656 bytes

## Fixed real-video result

The authoritative subset is 300 deterministic full-resolution DAVIS validation
triplets selected with seed 2,405. It is identical for all rows.

| Candidate | RGB MAE | RGB PSNR | Motion-proxy MAE | Motion-proxy PSNR | Object MAE | Object PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear blend | 0.0721960 | 20.1207 | 0.0934461 | 18.2591 | 0.109902 | 17.1942 |
| Orbit analytic checkpoint | 0.0744357 | 19.3869 | 0.0958532 | 17.6084 | 0.111753 | 16.6704 |
| Prism mixed 001 | **0.0686962** | **20.1624** | **0.0873207** | **18.4653** | **0.105291** | **17.3307** |

Prism improves over the frozen Orbit checkpoint by 0.7756 dB and over linear
blend by 0.0418 dB. The latter margin is real but too small to finish the phase.

Runtime at full DAVIS resolution, batch 1, model-only timing: median 20.60 ms,
p90 21.79 ms, p99 21.86 ms; peak allocated inference memory 468,379,648 bytes.

## Synthetic retention

On 256 analytic validation samples, Prism reaches 33.9081 dB versus 31.9556 dB
for linear blend. Moving-region PSNR is 30.1372 dB, occlusion-region PSNR is
20.9906 dB, and moving-flow EPE is 3.9495 px. Exact-flow supervision therefore
remains active after real-data adaptation.

## Interpretation

Real data corrects most of the synthetic domain gap without increasing model
size or latency. Remaining gain over linear blend is insufficient, especially
for object regions. The next controlled change should add coarse correspondence
and/or multiple motion hypotheses while keeping this training mix and fixed
DAVIS subset unchanged.
