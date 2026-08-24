# Prism: single-flow local correlation 001

Decision: **revise; do not promote and do not start multi-flow**.

## Hypothesis and isolated change

Add shared endpoint features at 1/8 resolution, bidirectional radius-4 local
dot-product cost volumes, and a zero-initialized velocity correction head. This
covers approximately +/-32 input pixels per component. Fusion and the single
motion hypothesis remain unchanged; oracle-warp supervision is retained.

## Cost

- Parameters: 667,898 (+117,090)
- Checkpoint bytes: 2,678,192
- SHA-256: `f53abc6cc6ede17cf3a0b842d82f3c825c0cd903763e7839b2e4beb4c31a506d`
- Training wall time / peak VRAM: 334.12 s / 597,446,144 bytes
- Full-resolution median latency: 24.04 ms versus 20.76 ms

## Results

| Metric | Oracle-warp baseline | Local correlation |
|---|---:|---:|
| DAVIS-300 PSNR | 20.6103 | **20.7239** |
| motion-proxy PSNR | 18.9005 | **19.0077** |
| object PSNR | 17.8482 | **17.9297** |
| analytic PSNR | 35.1296 | **35.3464** |
| analytic moving EPE | 2.6707 | **2.4352** |
| hard-case final PSNR | 14.6532 | **14.7109** |
| hard-case flow alignment | -0.4548 | **-0.4082** |

All aggregate metrics move in the intended direction, but the acceptance test
fails: five of six prior both-bad cases still have negative flow alignment. Only
`loading` crosses zero (+0.060 dB). The gain does not justify another blind A/B.

## Interpretation

Explicit correspondence evidence helps, but a shallow convolution over cost
channels does not reliably decode displacement. The next single-flow ablation
should expose fixed displacement coordinates/soft correspondence moments to the
velocity updater, rather than adding more flow hypotheses. Multi-flow stays
deferred until aggregate hard-case alignment is positive.
