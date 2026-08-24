# Prism: explicit displacement blinded human A/B

Decision: **reject perceptual promotion; stop small tweaks in this architecture**.

## Protocol

The unchanged six hard-motion DAVIS cases compared
`prism-coarse-explicit-displacement-davis-mixed-001` against
`prism-coarse-oracle-warp-davis-mixed-001`. Identities were randomly assigned
per case and sealed until the project owner answered.

## Result

All six responses were `both-bad`. The reviewer additionally reported that the
two candidates were not meaningfully distinguishable.

| Case | Challenger position | Preference |
|---|---|---|
| 01 | A | both-bad |
| 02 | B | both-bad |
| 03 | B | both-bad |
| 04 | B | both-bad |
| 05 | A | both-bad |
| 06 | A | both-bad |

Aggregate: challenger wins 0, baseline wins 0, ties 0, **both-bad 6/6**.

## Interpretation

The positive aggregate flow-alignment result (+0.0311 dB) and DAVIS-300 metric
gain do not produce a visible quality improvement. The acceptance criterion was
therefore necessary but not sufficient. Radius, temperature, channel count, or
loss-weight tuning inside this same shallow architecture is stopped.

The next candidate must be a structural single-flow correspondence system with
iterative/coarse-to-fine flow updates and feature warping. Multi-flow remains
deferred until that system creates an obvious single-flow alignment improvement.
