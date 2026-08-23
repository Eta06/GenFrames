# Prism: hard-motion component failure analysis

Decision: **flow/correspondence is primary; fusion/occlusion is secondary; do
not proceed directly to multi-flow**.

## Scope and method

The six unanimous `both-bad` cases from `prism-human-ab-002` were re-evaluated
with GT-assisted component diagnostics for both Prism checkpoints. Full JSON and
diagnostic panels are stored externally under
`benchmarks/prism-human-ab-002/diagnostics`.

Two counterfactuals isolate the pipeline:

- unwarped oracle chooses, per pixel, the raw endpoint closer to GT;
- warp oracle chooses, per pixel, the model-warped endpoint closer to GT.

If warp oracle loses to unwarped oracle, learned flow damages alignment. The gap
from final prediction to warp oracle estimates remaining fusion opportunity.

## Aggregate result on the six hard cases

| Diagnostic | Frozen real baseline | Coarse current-best |
|---|---:|---:|
| final PSNR | 14.4143 dB | 14.5274 dB |
| unwarped oracle PSNR | 17.1247 dB | 17.1247 dB |
| warp oracle PSNR | 16.6219 dB | 16.6061 dB |
| flow alignment gain | **-0.5027 dB** | **-0.5186 dB** |
| fusion opportunity | +2.2077 dB | +2.0787 dB |
| high warp-disagreement pixels | 56.24% | 54.90% |
| residual PSNR contribution | +0.0205 dB | +0.0034 dB |
| prediction / GT gradient energy | 77.70% | 77.04% |

Flow alignment gain is negative in every individual case, ranging from -0.13 to
-1.09 dB for the coarse checkpoint. The failure is therefore not one outlier.

## Common visual patterns

1. Predicted velocity is high-frequency and texture-following rather than a
   coherent object/camera-motion field.
2. Misaligned endpoint warps create doubled edges and translucent copies across
   riders, vehicles, tree trunks, weapons, and background structure.
3. Blend weights stay indecisive: mean 0.534, standard deviation 0.145, and zero
   pixels below 0.1 or above 0.9. The fusion branch does not behave like a usable
   visibility selector.
4. The bounded RGB residual is effectively inactive and cannot reconstruct
   details or undo geometric ghosting.
5. Output gradient energy is about 23% below GT, explaining perceived blur, but
   blur is a consequence of misalignment/fusion rather than the first cause.

## Root-cause ranking

1. **Incorrect flow/correspondence:** primary and directly proven by negative
   warp-oracle alignment gain.
2. **Occlusion/fusion:** major secondary problem; even current poor warps have a
   two-decibel GT-oracle selection gap.
3. **Detail/refinement:** insufficient, but a pixel residual cannot repair large
   geometric errors and should not be enlarged first.
4. **Domain gap:** still relevant because exact synthetic flow transfers poorly,
   though real fine-tuning already recovered 0.78 dB over the Orbit checkpoint.

## Next controlled ablation

Before adding correlation or multiple flows, add a real-GT-compatible oracle-warp
loss: at each pixel, penalize the lower reconstruction error of the two endpoint
warps. This directly supervises “at least one warp aligns” without pretending
real DAVIS has flow labels or punishing genuinely occluded endpoints.

Keep architecture, initialization, data mix, schedule, and evaluation subset
fixed. If warp-oracle alignment does not become positive, the next justified
change is an explicit endpoint correspondence mechanism or audited teacher-flow
supervision. Multi-flow remains deferred until a single hypothesis can align.
