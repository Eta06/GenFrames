# Motion representation reassessment after single-flow rejection

## Trigger

The structural coarse-to-fine single target-centric velocity model improved
DAVIS-300 PSNR but regressed alignment on all six sealed hard cases. This ends
the single-flow pyramid line.

## Evidence from prior work

- RIFE directly estimates two intermediate flows, `F(t->0)` and `F(t->1)`, and
  reports that privileged distillation stabilizes training. Its ablation also
  shows that endpoint RAFT plus flow reversal is weaker than direct intermediate
  flow estimation. Paper: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740608.pdf>
- IFRNet jointly refines bilateral intermediate flows and target features, then
  uses task-oriented optical-flow distillation to transfer only knowledge useful
  to interpolation. Paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.html>
- RAFT uses all-pairs correlation and recurrent updates, avoiding the early-error
  and small-fast-object weaknesses of classic coarse-to-fine local matching.
  Paper: <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf>
- AMT combines all-pairs matching with multiple flow fields for occlusion, but
  its reference code is CC BY-NC 4.0. It is a research reference, not a code base
  for GenFrames. Paper: <https://openaccess.thecvf.com/content/CVPR2023/html/Li_AMT_All-Pairs_Multi-Field_Transforms_for_Efficient_Frame_Interpolation_CVPR_2023_paper.html>

## Controlled teacher ceiling

`prism-privileged-raft-small-audit-001` estimates training-only target-to-endpoint
flows with TorchVision RAFT-small. The ground-truth middle frame is intentionally
available to the teacher; it will never be an inference input.

| Case | Sequence | Alignment gain |
|---|---|---:|
| 01 | shooting | +10.4993 dB |
| 02 | lab-coat | +10.7890 dB |
| 03 | drift-straight | +6.7193 dB |
| 04 | motocross-jump | +8.1599 dB |
| 05 | loading | +16.3901 dB |
| 06 | bmx-trees | +10.1098 dB |

Aggregate is +10.4446 dB, with 6/6 positive cases. Visual panels show coherent
piecewise motion and sharply reduced ghosting. This demonstrates that endpoint
warping has a high ceiling when correspondence is supervised correctly.

## Next ablation

Train a small student that predicts independent `F(t->0)` and `F(t->1)` while
keeping the existing RGB warp and fusion architecture fixed. Preserve exact
analytic flow labels for synthetic samples. For real samples, use RAFT-small
target-to-endpoint flows only during training, with task-oriented reliability
masks. The RAFT implementation and weights are training provenance, not an
inference or redistribution dependency.

Acceptance remains unchanged: aggregate hard-case alignment must be positive,
gains must span multiple cases, and visible ghosting must materially decrease
before a blind A/B is generated.
