# Parallax fixed-point multi-field audit 002

## Decision

Representation passed. Proceed to a small selector/fusion ablation while freezing
candidate generation. This result is not a final model and is not eligible for A/B.

## Hypothesis and change

Keep RAFT endpoint correspondence fixed. In addition to the two quadratic endpoint
warps, retain target-to-source fixed-point inversion iterates 0/1/2/4 from each
endpoint. This creates ten independently warpable appearance hypotheses. No selector,
fusion, or learned parameter is introduced; GT oracle selection is diagnostic only.

## Results

| Dataset/region | Single oracle | Multi oracle | Gain |
|---|---:|---:|---:|
| DAVIS six hard cases | 20.0750 dB | 22.0303 dB | +1.9553 dB |
| DAVIS validation, 24 crops | 24.7363 dB | 26.2681 dB | +1.5318 dB |
| GoPro official test, 24 crops | 31.6812 dB | 32.0915 dB | +0.4103 dB |
| Analytic exact motion, 24 | 44.8466 dB | 49.8171 dB | +4.9705 dB |

- All six DAVIS hard cases improve: +0.729 to +4.087 dB.
- Hard-case high-disagreement MAE: 0.07775 -> 0.05005.
- A better candidate exists on 54.96% of high-disagreement pixels.
- Analytic exact occlusion MAE: 0.01373 -> 0.00508.
- A better candidate exists on 36.14% of exact occlusion pixels.
- GoPro gains are smaller because the fixed subset contains only 2.56% high-
  disagreement pixels, but remain positive.

Visual oracle panels confirm that the gain is not isolated noise. Drift-straight,
motocross-jump, shooting, and BMX contain substantially better aligned object parts
than the quadratic pair. The oracle still contains seams because per-pixel GT
selection is not spatially regularized; it demonstrates candidate availability, not
a deployable fusion result.

## Cost

- Added learned parameters: 0
- Representation time at 256 square: 0.718 ms single vs 5.877 ms multi
- End-to-end with common RAFT oracle: 77.17 ms single vs 82.35 ms multi
- End-to-end peak CUDA allocation: 118,004,736 bytes
- Research-oracle total parameters: 1,540,750

The next experiment must keep these fields fixed and train only a compact spatial
selector. It should report how much of the +1.96 dB hard-case oracle margin is
recoverable, especially in high-disagreement and exact-occlusion regions. A blind A/B
remains blocked until the actual selected output visibly reduces ghosting.

## Artifacts

- Audit: `D:/GenFrames/benchmarks/parallax-fixed-point-multifield-audit-002/result.json`
- Visual panels: sibling `case-01.png` through `case-06.png`
