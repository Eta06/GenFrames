# Prism: explicit displacement moments 001

Decision: **passes aggregate alignment threshold; await blinded human A/B before
promotion or further architectural branching**.

## Change

Convert bidirectional radius-4 correlation distributions into fixed `(dx, dy)`
soft-argmax moments. Feed these moments into the decoder and through a
zero-initialized direct velocity gate. Everything else matches the prior local
correlation ablation.

## Cost and result

- Parameters/checkpoint: 669,051 / 2,682,892 bytes
- SHA-256: `668751060c520010ebbe3134cfb9b211c34d7508206a5845393a33d8b5821ff7`
- Training: 1,500 steps, 294.52 s, peak VRAM 615,156,224 bytes
- Full-resolution median latency: 24.33 ms

| Metric | Shallow correlation | Explicit displacement |
|---|---:|---:|
| DAVIS-300 PSNR | 20.7239 | **21.0971** |
| motion PSNR | 19.0077 | **19.3981** |
| object PSNR | 17.9297 | **18.3048** |
| analytic PSNR | 35.3464 | **36.0072** |
| analytic moving EPE | 2.4352 | **1.9903** |
| hard-case final PSNR | 14.7109 | **15.0222** |
| hard-case flow alignment | -0.4082 | **+0.0311** |

The aggregate acceptance threshold is positive for the first time. However,
five of six cases remain individually negative; the aggregate is lifted by a
large +1.8038 dB alignment gain on `loading`. A blinded A/B on the unchanged six
cases is mandatory before claiming meaningful perceptual progress.
