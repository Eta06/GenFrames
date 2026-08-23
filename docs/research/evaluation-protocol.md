# Common evaluation protocol

Last updated: 2026-08-24

## Correctness before quality

Every candidate must pass identity, endpoint, shape, finite-gradient, deterministic
seed, save/load, and arbitrary-time tests. A model that cannot overfit a tiny
controlled dataset is treated as a pipeline failure, regardless of benchmark code.

## Full-reference quality

- PSNR on float RGB `[0, 1]`, with no unreported border crop.
- SSIM with implementation/version and color space recorded.
- LPIPS with network/version recorded.
- Optional Y-channel results are additional columns, never substitutes for RGB.
- FloLPIPS/VFIPS are added when their exact implementations and checkpoint terms
  have been pinned.
- Report midpoint and each arbitrary target time separately, not only their mean.

## Temporal and diagnostic quality

- temporal warping/flow consistency over generated sequences;
- endpoint symmetry: swap inputs and replace `t` with `1-t`;
- acceleration/non-linear-motion slices;
- boundary error around occlusion and thin structures;
- stress slices: faces, hands, text, animation, game capture, particles,
  transparency, motion blur, compression, camera pan, and scene cuts;
- visual grids and short lossless comparison clips for failures.

## Efficiency

For each device, resolution, precision, and batch size:

- parameters and serialized checkpoint bytes;
- multiply-accumulate/FLOP estimate with tool/version;
- median, p90, and p99 latency after warm-up;
- throughput for one and multiple `t` values;
- peak allocated/reserved device memory;
- host RAM where available;
- exact preprocessing and data-transfer inclusion policy.

CUDA timings use events or explicit synchronization. The default local profiles
are batch 1 at 256p crop, 480p, 720p, and 1080p in FP32 and FP16. The RTX 3070
8 GB environment is not used to claim cross-device performance.

## Human A/B evaluation

1. Render outputs with randomized opaque labels.
2. Preserve identical encoding and playback parameters.
3. Ask for A, B, tie, or both-bad plus optional failure tag.
4. Reveal identities only after the answer is recorded.
5. Store the prompt, randomized mapping, source sample ID, and response in the
   experiment record.

## Experiment decision

Each experiment ends with `continue`, `revise`, or `eliminate`, supported by the
measured trade-off and qualitative failures. A module stays only if its gain is
repeatable and worth its latency/memory/operator cost.

