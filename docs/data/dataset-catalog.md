# Dataset catalog and proposed roles

Last updated: 2026-08-24

Licensing notes are operational risk notes, not legal conclusions. A repository
license does not automatically license its downloaded videos or checkpoints.
`Unknown` means the primary source inspected did not state sufficiently clear
terms; it does not mean unrestricted use.

## Catalog

| Dataset | Content / motion | FPS and resolution | Approximate scale | License state | Proposed role |
|---|---|---|---|---|---|
| Vimeo90K Triplet | broad web video, mostly local/moderate motion | 3 frames, 448x256; source FPS varies | 73,171 triplets, about 33 GB full archive | **Unknown for dataset**; official code is MIT, source clips came from Vimeo | research train/validation; release-weight use requires review |
| Vimeo90K Septuplet | longer local temporal context from the same source family | 7 frames, 448x256 | 91,701 septuplets; original archive reported around 82 GB | **Unknown for dataset** | temporal-context research, not default VFI benchmark |
| X4K1000FPS | extreme motion and high spatial detail | 1000 FPS; source 4096x2160; train crops 768x768 | train: 4,408 clips from 110 scenes, 65 frames each; test: 15 clips, 33 frames each | **Unknown** in inspected primary repository | large-motion train/benchmark; especially x4/x8 gaps |
| SNU-FILM | GoPro and YouTube sequences stratified by temporal gap | about 1280x720 | 4 difficulty sets (easy/medium/hard/extreme), 310 triplets each commonly reported | **Unknown for dataset**; CAIN code is MIT | benchmark only |
| Middlebury Flow/Interpolation | hidden-texture, synthetic, stereo, high-speed real scenes | mixed, small legacy resolutions; usually 8-frame sequences | 12 evaluation + 12 public “other” sequences | **Unknown**; official page provides benchmark downloads | tiny diagnostic benchmark; never primary training data |
| DAVIS 2016/2017 | high-quality natural videos, deformable objects, masks | 24 FPS; originally 1080p with 480p distribution | DAVIS 2016: 50 sequences/3,455 frames; DAVIS 2017: 60 train + 30 validation examples in TFDS | source-specific Creative Commons terms; verify exact files | perceptual/temporal benchmark and segmentation-aware diagnostics |
| UCF101 | compressed YouTube action clips, humans, sports, camera motion | commonly 25 FPS, 320x240 | 13,320 clips, 101 action classes, about 27 hours | official page does not state sufficiently clear reuse terms; third-party summaries call it academic/non-commercial | legacy benchmark only |
| GoPro Large / GoPro Large all | dynamic scenes, sharp high-rate frames and integrated blur pairs | captured at 240 FPS; HD frames | public train/test sharp and blur pairs; exact count depends on archive variant | CC BY 4.0 stated by dataset owner | promising real-data train source; retain attribution/provenance |
| Adobe240 | natural 240 FPS sequences used by Super SloMo/DeMFI | 240 FPS; primarily 720p sequences in released variants | DeMFI mirror reports 49.7 GB split archive | original terms need verification before training release weights | research train and blur/interpolation evaluation |
| SportsSloMo | human-centric sports, deformable bodies and frequent occlusion | high-resolution (at least 720p), slow-motion source videos | official paper reports over 130K clips / over 1M frames; repo distributes over 8K long source clips | research and education only; commercial use requires permission | research-only hard-set and human-aware experiments |
| LAVIB | large and diverse interpolation benchmark with OOD motion/appearance splits | variable web-video properties; 10-second segments cropped into 1-second tubelets | chunked in 20 GB archives; exact total should be read from downloaded manifest | CC BY-SA-NC 4.0 | non-commercial research teacher/robustness benchmark |
| BVI-VFI | VFI outputs with human differential opinion scores | multiple spatial resolutions and frame rates | 108 reference + 540 distorted sequences | academic use only; Bristol retains IP | metric validation and human-quality correlation only |
| SA-V | diverse natural videos with object masklets | repository evaluation uses 24 FPS videos; varying resolutions | 51K videos, 643K masklets | CC BY 4.0 | optional segmentation/visibility auxiliary data, not high-FPS GT |
| VFIbench synthetic set | synthetic test data designed around a linear-motion assumption | see benchmark release | test-focused; exact downloadable scale pending acquisition | **Unknown pending benchmark terms** | attribute-conditioned evaluation when obtainable |

## Primary source and download index

- Vimeo90K: [dataset page](https://data.csail.mit.edu/tofu/) and
  [official repository](https://github.com/anchen1011/toflow)
- X4K1000FPS: [official XVFI repository](https://github.com/JihyongOh/XVFI)
- SNU-FILM: [official CAIN repository](https://github.com/myungsub/CAIN)
- Middlebury: [official dataset page](https://vision.middlebury.edu/flow/data/)
- DAVIS: [CVPR 2016 paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)
- UCF101: [official UCF page](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- GoPro: [dataset owner page](https://seungjunnah.github.io/Datasets/gopro.html)
- Adobe240 mirror/instructions: [DeMFI repository](https://github.com/JihyongOh/DeMFI)
- SportsSloMo: [official repository](https://github.com/neu-vi/SportsSloMo)
- LAVIB: [official repository](https://github.com/alexandrosstergiou/LAVIB)
- BVI-VFI: [official repository](https://github.com/danier97/BVI-VFI-database)
- SA-V: [official repository](https://github.com/facebookresearch/sam2/tree/main/sav_dataset)
- VFIbench: [paper](https://arxiv.org/abs/2403.17128) and
  [benchmark site](https://sniklaus.com/vfibench)

## Data-role policy

The project accepts risk-aware research use requested by the project owner, but
must preserve the ability to understand and later replace risky sources.

Every downloaded source receives a manifest with:

- canonical source URL and retrieval date;
- archive checksum and extraction tool version;
- stated dataset, source-video, and annotation terms;
- allowed project role: `train`, `validation`, `benchmark`, `teacher-only`, or
  `human-evaluation-only`;
- risk: `green`, `yellow`, or `red` plus a plain-language reason;
- derived sample IDs mapping back to source clip and frame indices.

Research checkpoints trained on yellow/red sources are labelled accordingly and
cannot silently become release checkpoints. Teacher-generated targets retain the
teacher checkpoint and source-data provenance in their manifests.

## Split rules

1. Split by original source video or scene, never by independently cropped triplet.
2. De-duplicate exact and near-duplicate clips before splitting.
3. Maintain dedicated stress slices for large motion, occlusion, thin structures,
   text/UI, faces/hands, animation, compression, motion blur, and camera panning.
4. Report both midpoint and arbitrary-time/multi-frame performance.
5. Keep benchmark ground truth out of hyperparameter selection where a distinct
   validation split is available.

