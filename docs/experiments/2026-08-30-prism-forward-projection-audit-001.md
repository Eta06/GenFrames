# Prism forward projection audit 001

## Decision

Rejected before training. Do not build a fusion model on this representation.

## Hypothesis and change

Replace quadratic target-to-source backward warps with bilinear forward projection
of both RAFT endpoint flows to target time. Track splat mass, collisions, and holes;
use endpoint photometric consistency as fixed visibility importance. This tests a
different motion representation without changing or training matching/fusion.

## Setup

- Frozen TorchVision RAFT-small C_T_V2
- The same six sealed DAVIS hard-motion cases
- Importance alpha sweep: 0, 10, 20, 40
- Quadratic RAFT time blend used only to fill projection holes
- No learned parameters and no checkpoint

## Results

- Best aggregate: alpha 10, 18.2770 dB
- Delta over quadratic RAFT blend: **+0.0634 dB**
- Per-case delta: +0.624, +0.029, +0.063, -0.271, -0.071, +0.007 dB
- Mean hole fraction at alpha 10: 1.48%
- Mean disagreement-region MAE at alpha 10: 0.1005

The small aggregate gain is not spread robustly and the visual gate fails. Shooting
improves, but drift and motocross retain severe rubber-sheet/collision artifacts;
BMX retains a doubled cyclist. Higher consistency weighting converts ghosting into
holes and broken object fragments rather than solving it.

Global endpoint matching remains useful, but neither one quadratic backward field
nor one forward-projected field can represent competing target-time motion and
visibility hypotheses. The single-field acceptance proof has now been obtained for
correspondence (the prior +2.9618 dB audit), and its representation limit has also
been isolated. A controlled multi-hypothesis/field target representation is now
justified; further scalar tweaks to these single-field paths are not.

## Artifacts

- Audit: `D:/GenFrames/benchmarks/prism-forward-projection-audit-001/result.json`
- Panels: sibling `case-01.png` through `case-06.png`
