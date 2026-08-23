# Prism stronger-motion decision note

The controlled sequence is based on primary-source mechanisms, not copied code:

1. **Coarse velocity correction first.** IFRNet jointly refines bilateral flows
   and intermediate features through a feature pyramid. GenFrames first tests a
   much smaller change: a zero-initialized 1/4-resolution velocity head added to
   the existing full-resolution prediction. This isolates whether coarse context
   helps before introducing correlation memory.
2. **Multi-field only after coarse ablation.** AMT reports that all-pairs
   bidirectional correlations aid large motion and that multiple fine-grained
   flow groups reduce occlusion ambiguity. If coarse correction is insufficient,
   GenFrames will next test two motion hypotheses with shared features.
3. **Selective global matching later.** SGM-VFI applies global matching only to
   locations where local flow is flawed. This is preferable to full-resolution
   all-pairs memory for a portable/mobile path, but sparse selection/indexing has
   higher MLX implementation risk and should be justified by a hard-motion slice.

Primary sources:

- [IFRNet (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.html)
- [AMT (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Li_AMT_All-Pairs_Multi-Field_Transforms_for_Efficient_Frame_Interpolation_CVPR_2023_paper.html)
- [SGM-VFI (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Sparse_Global_Matching_for_Video_Frame_Interpolation_with_Large_Motion_CVPR_2024_paper.html)

The coarse candidate uses convolution, elementwise addition, tanh, and bilinear
resize only. These already exist in the MLX portability plan. It adds no custom
correlation, scatter, deformable convolution, or dynamic sparse-index operator.
