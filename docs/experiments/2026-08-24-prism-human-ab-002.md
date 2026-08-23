# Prism: blinded hard-motion human A/B 002

Decision: **metric gain is not perceptually sufficient; diagnose before selecting
the next architecture**.

## Protocol

- Six unique DAVIS validation sequences were chosen by the highest hard-motion
  score inside the frozen DAVIS-300 subset.
- Panel order: input 0, candidate A, ground truth, candidate B, input 1.
- Candidate identity was sealed until all preferences were received.
- Compared checkpoints: `prism-bilateral-flow-davis-mixed-001` and
  `prism-bilateral-flow-coarse-davis-mixed-001`.

## Human result

| Case | Sequence | Preference |
|---|---|---|
| 01 | shooting | both-bad |
| 02 | lab-coat | both-bad |
| 03 | drift-straight | both-bad |
| 04 | motocross-jump | both-bad |
| 05 | loading | both-bad |
| 06 | bmx-trees | both-bad |

Aggregate: A wins 0, B wins 0, ties 0, **both-bad 6/6**.

After sealing the response, the key showed the coarse challenger as A in cases
01, 05, and 06, and as B in cases 02, 03, and 04. Because every response is
both-bad, the random mapping has no effect on the conclusion.

## Interpretation

The coarse checkpoint's PSNR/MAE improvement does not translate into acceptable
visual quality on selected hard-motion examples. Neither checkpoint is promoted
as perceptually adequate. Multi-flow is not automatically authorized by this
result; the six cases require component-level flow, warp, fusion, residual, and
detail diagnostics first.
