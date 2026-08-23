# Synthetic data plan

Last updated: 2026-08-24

Synthetic data is a first-class supervision source because ordinary RGB video
does not provide exact motion, depth, visibility, or shutter integration. It is
not expected to replace real video; it supplies controlled edge cases and dense
auxiliary targets.

## Renderer choice

Start with Blender in headless mode because its Python API, deterministic scene
construction, Cycles/Eevee render engines, depth pass, vector pass, object IDs,
and compositing are scriptable without a game project. Unreal remains an option
for photorealistic domain expansion after the supervision contract is stable.

## Output contract per sequence

```text
sequence_id/
  metadata.json
  rgb_linear/
  rgb_display/
  depth/
  motion_forward/
  motion_backward/
  visibility_forward/
  visibility_backward/
  instance_id/
  semantic_id/
  alpha/
  shutter_integrated/
```

Metadata records renderer/version, scene seed, frame times, exposure interval,
camera matrices, object transforms, material flags, units, color transform, and
all file hashes.

## Ground-truth derivation

- Render RGB at exact sub-frame times for arbitrary `t` targets.
- Export depth in a linear, documented camera-space convention.
- Use renderer motion/vector passes plus camera/object matrices; validate signs
  and pixel units with analytic translation tests.
- Derive visibility by forward projection with target-time depth testing. Keep
  disocclusion separate from out-of-frame motion.
- Generate motion blur by integrating multiple linear-light sub-frame renders
  over a sampled shutter interval. Do not average gamma-encoded RGB.
- Retain instance/semantic IDs to measure boundary, human-like articulated, and
  thin-object errors separately.

## Procedural scenario families

1. textured rigid primitives with independently sampled 3D motion;
2. articulated characters and cloth-like deformation;
3. camera pan, tilt, roll, dolly, orbit, zoom, and handheld trajectories;
4. foreground/background depth crossings and full/partial occlusion;
5. fences, wires, spokes, hair cards, foliage, particles, and small fast objects;
6. emissive surfaces, reflections, shadows, lighting changes, and exposure shifts;
7. transparent/translucent layers with an explicit “ambiguous supervision” flag;
8. text, UI-like planes, pixel art, and animation-style flat shading;
9. non-linear acceleration, direction reversal, rotation, and scale/depth change;
10. controlled compression and sensor-noise post-processing after clean GT render.

## Curriculum

- **S0 analytic:** translations/rotations whose expected flows are testable.
- **S1 rigid:** simple textured 3D scenes, sharp shutter, modest displacement.
- **S2 occlusion:** depth crossings, disocclusion, thin structures, larger gaps.
- **S3 non-linear:** acceleration, curved paths, camera/object interaction.
- **S4 appearance:** lighting, shadows, transparency, blur, noise, compression.
- **S5 domain mix:** combine synthetic auxiliary supervision with real RGB losses.

## Acceptance checks

- zero-motion frames reconstruct exactly within color-encoding tolerance;
- analytic 1/2/4/8-pixel translations match exported flow conventions;
- forward/backward visibility masks agree away from boundaries;
- frame timestamps and shutter sample weights sum correctly;
- source-scene seeds are disjoint across train/validation/test;
- rendering is resumable and deterministic at the manifest level.

