# GenFrames

GenFrames is a research project for developing an original video frame
interpolation model. The target is arbitrary-time interpolation, including
30 to 60 FPS and 60 to 120 FPS conversion, with first-class PyTorch/CUDA and
Apple Silicon MLX portability.

The previous RIFE wrapper implementation has been removed. GenFrames will
contain its own architecture, training pipeline, evaluation harness, and
checkpoints. Existing VFI systems are treated as research references,
baselines, or potential supervision sources, not as the product identity.

## Current status

Phase 5 (Prism): real-data and stronger-motion development. DAVIS 2017 ingestion,
real/synthetic mixing, regional evaluation, and immutable checkpoint tracking are
active. The first mixed checkpoint beats its frozen synthetic predecessor by
0.78 dB on the fixed real-video subset, but its 0.04 dB lead over linear blend is
not yet sufficient to complete the phase.

Research records, dataset provenance, architectural decisions, and experiment
results will be committed to this repository as the project progresses.

## License

The repository is currently distributed under the MIT License. Third-party
datasets, checkpoints, and research artifacts retain their own terms. Their
provenance and risk classification will be documented before use.
