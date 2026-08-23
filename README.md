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

Phase 0: repository reset and research refresh.

Research records, dataset provenance, architectural decisions, and experiment
results will be committed to this repository as the project progresses.

## License

The repository is currently distributed under the MIT License. Third-party
datasets, checkpoints, and research artifacts retain their own terms. Their
provenance and risk classification will be documented before use.
