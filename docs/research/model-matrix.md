# Video frame interpolation model matrix

Last updated: 2026-08-24

This is a decision aid, not a leaderboard. Roles are intentionally non-exclusive:
a system can be a baseline, an architectural inspiration, and a supervision
source. No entry authorizes copying code or redistributing weights. Code,
checkpoint, and dataset terms are audited separately before an artifact is used.

## Role definitions

- **Baseline:** run in the common harness when reproducible within the budget.
- **Inspiration:** reimplement and ablate a paper-level idea in GenFrames.
- **Potential teacher:** evaluate outputs or internal supervision for distillation.
- **Supervision source:** generate pseudo-targets, flows, masks, or confidence cues.
- **Unsuitable:** not a primary path because of cost, operators, scope, or fidelity.

## Matrix

| System | Core direction | Candidate roles | GenFrames takeaway | Main concern |
|---|---|---|---|---|
| SepConv (2017) | adaptive separable kernels | inspiration, baseline | learned local resampling without explicit flow | kernel support grows poorly with large motion |
| Super SloMo (2018) | bidirectional flow and visibility | baseline, inspiration | arbitrary-time formulation and multi-frame supervision | older flow stack and weak large-motion handling |
| DAIN (2019) | depth-aware flow projection | inspiration, supervision source | explicit ordering cue for occlusion | heavy multi-network pipeline; custom projection |
| SoftSplat (2020) | differentiable forward warping | inspiration, baseline | collision-aware feature propagation | scatter/splat kernels complicate MLX and edge export |
| RIFE (2020/2022) | direct intermediate-flow estimation | baseline, potential teacher | task-oriented intermediate flow and privileged distillation | common implementations/checkpoints have separate provenance |
| IFRNet (2022) | coarse-to-fine flow and feature refinement | baseline, inspiration, potential teacher | compact pyramid, joint feature/flow refinement, distillation | local refinement can miss extreme displacement |
| FILM (2022) | scale-agnostic shared feature pyramid | baseline, potential teacher, inspiration | weight sharing across scales for large motion | recursive multi-frame inference and archived TF implementation |
| VFIformer (2022) | local spatio-temporal attention | baseline, inspiration | context beyond convolutional receptive fields | attention cost and MLX parity risk |
| FLAVR (2021/2023) | flow-free 3D convolution | baseline, alternate direction | tests whether explicit warping is necessary | fixed temporal windows; exact arbitrary-time control is not native |
| EBME (2023) | enhanced bidirectional motion | baseline, inspiration | small motion estimator with competitive reconstruction | paper/runtime numbers require local reproduction |
| UPR-Net (2023) | recurrent shared pyramid | baseline, inspiration | resolution-adaptive pyramid depth and parameter reuse | recurrent scheduling/export complexity |
| AMT (2023) | all-pairs correlation and multi-field transforms | baseline, potential teacher, inspiration | multiple motion hypotheses for occlusion and large motion | full correlation memory at high resolution |
| EMA-VFI (2023) | CNN/attention motion-appearance extraction | baseline, potential teacher | reuse attention evidence for appearance and correspondence | hybrid attention portability and memory |
| SGM-VFI (2024) | sparse global matching for bad-flow regions | inspiration, potential teacher | pay global-matching cost only where local flow is unreliable | sparse indexing and selection need portable implementation |
| IQ-VFI (2024) | implicit quadratic motion | inspiration, potential teacher | model non-linear motion rather than assuming constant velocity | extra temporal context and training complexity |
| PerVFI (2024) | asymmetric blending and normalizing flow | potential teacher, HQ inspiration | perception-oriented visibility selection | likelihood/generative objective may trade fidelity for texture |
| BiM-VFI (2025) | bidirectional motion-field representation | potential teacher, inspiration | reduce time-to-location ambiguity for acceleration and direction changes | implementation and checkpoint availability require audit |
| HFD (2025) | hierarchical diffusion over bilateral flow | potential teacher, alternate HQ direction | diffuse in compact motion space rather than image latent space | iterative sampling remains expensive |
| EDEN (2025) | latent diffusion transformer | potential teacher, qualitative baseline | semantic prior for ambiguous large motion | hallucination, latency, memory, and difficult MLX deployment |
| TLB-VFI (2025) | temporal latent Brownian bridge | research reference | explicit temporal diffusion trajectory | unsuitable for the first deterministic/minimal model |
| LDF-VFI (2026) | autoregressive sequence-level diffusion | potential teacher, sequence-consistency inspiration | evaluate entire-video coherence, tiled high-resolution processing | billion-scale dependency chain and autoregressive cost |
| ARVFI (2026) | bidirectional autoregressive diffusion with semantic motion | research reference, possible teacher | semantic motion features may resolve complex paths | excessive scope for initial hardware budget |
| ANVIL (2026) | accelerator-native interpolation with codec MVs | inspiration, later lightweight baseline | design for supported operators and exploit codec motion priors | codec dependence must remain optional for raw-frame use |
| SPEED (2026 preprint) | one-step pixel diffusion | monitor, possible teacher | test whether one-step generative refinement improves perception | too new for immediate architectural commitment |

## Primary sources

- [SepConv](https://arxiv.org/abs/1708.01692)
- [Super SloMo](https://arxiv.org/abs/1712.00080)
- [DAIN](https://arxiv.org/abs/1904.00830)
- [SoftSplat](https://openaccess.thecvf.com/content_CVPR_2020/html/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.html)
- [RIFE](https://arxiv.org/abs/2011.06294)
- [IFRNet](https://openaccess.thecvf.com/content/CVPR2022/html/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.html)
- [FILM](https://arxiv.org/abs/2202.04901)
- [VFIformer](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Video_Frame_Interpolation_With_Transformer_CVPR_2022_paper.html)
- [FLAVR](https://arxiv.org/abs/2012.08512)
- [EBME](https://openaccess.thecvf.com/content/WACV2023/html/Jin_Enhanced_Bi-Directional_Motion_Estimation_for_Video_Frame_Interpolation_WACV_2023_paper.html)
- [UPR-Net](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_A_Unified_Pyramid_Recurrent_Network_for_Video_Frame_Interpolation_CVPR_2023_paper.html)
- [AMT](https://openaccess.thecvf.com/content/CVPR2023/html/Li_AMT_All-Pairs_Multi-Field_Transforms_for_Efficient_Frame_Interpolation_CVPR_2023_paper.html)
- [EMA-VFI](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Extracting_Motion_and_Appearance_via_Inter-Frame_Attention_for_Efficient_Video_CVPR_2023_paper.html)
- [SGM-VFI](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Sparse_Global_Matching_for_Video_Frame_Interpolation_with_Large_Motion_CVPR_2024_paper.html)
- [IQ-VFI](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_IQ-VFI_Implicit_Quadratic_Motion_Estimation_for_Video_Frame_Interpolation_CVPR_2024_paper.html)
- [PerVFI](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Perception-Oriented_Video_Frame_Interpolation_via_Asymmetric_Blending_CVPR_2024_paper.html)
- [BiM-VFI](https://openaccess.thecvf.com/content/CVPR2025/html/Seo_BiM-VFI_Bidirectional_Motion_Field-Guided_Frame_Interpolation_for_Video_with_Non-uniform_CVPR_2025_paper.html)
- [HFD](https://openaccess.thecvf.com/content/CVPR2025/html/Hai_Hierarchical_Flow_Diffusion_for_Efficient_Frame_Interpolation_CVPR_2025_paper.html)
- [EDEN](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_EDEN_Enhanced_Diffusion_for_High-quality_Large-motion_Video_Frame_Interpolation_CVPR_2025_paper.html)
- [TLB-VFI](https://openaccess.thecvf.com/content/ICCV2025/html/Lyu_TLB-VFI_Temporal-Aware_Latent_Brownian_Bridge_Diffusion_for_Video_Frame_Interpolation_ICCV_2025_paper.html)
- [LDF-VFI](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Towards_Holistic_Modeling_for_Video_Frame_Interpolation_with_Auto-regressive_Diffusion_CVPR_2026_paper.html)
- [ARVFI](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Bi-directional_Autoregressive_Diffusion_for_Large_Complex_Motion_Interpolation_CVPR_2026_paper.html)
- [ANVIL](https://arxiv.org/abs/2603.26835)
- [SPEED](https://arxiv.org/abs/2607.15585)

## Initial experiment directions

No winner is selected in Phase 0. The first matrix will compare four deliberately
different directions under the same parameter and memory reporting:

1. **Direct blend residual:** a small flow-free sanity model proving data and loss wiring.
2. **Portable bilateral flow:** direct arbitrary-time intermediate flow with backward warp.
3. **Multi-hypothesis hybrid:** two or more flows plus learned visibility/confidence fusion.
4. **Selective global motion:** local pyramid by default, global matching only at coarse scale.

Diffusion and sequence-level models remain teacher/HQ research paths until the
minimal deterministic model and benchmark harness are trustworthy.

