"""Fine-tune the immutable Orbit baseline on real and exact-flow synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from genframes.data import (
    AnalyticMotionDataset,
    DatasetManifest,
    ManifestFrameDataset,
    MixedFrameDataset,
    Split,
)
from genframes.eval import evaluate_model
from genframes.models import BilateralFlowConfig, GenFramesBilateralFlow, LinearBlend
from genframes.storage import StorageLayout
from genframes.training import InterpolationLoss, LossConfig, TrainConfig, train_steps


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    layout.create()
    destination = layout.checkpoints / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {destination}")
    manifest_path = layout.root / "datasets" / "davis-2017" / "genframes-manifest.json"
    dataset_root = layout.root / "datasets" / "davis-2017" / "DAVIS"
    manifest = DatasetManifest.load(manifest_path)
    crop = (arguments.crop_size, arguments.crop_size)
    real_train = ManifestFrameDataset(
        manifest,
        dataset_root=dataset_root,
        split=Split.TRAIN,
        crop_size=crop,
        seed=arguments.seed,
        horizontal_flip=True,
    )
    synthetic_train = AnalyticMotionDataset(
        length=arguments.mixture_samples,
        height=arguments.crop_size,
        width=arguments.crop_size,
        seed=arguments.seed + 100_000,
    )
    mixture = MixedFrameDataset(
        {"real": real_train, "synthetic": synthetic_train},
        weights={"real": arguments.real_weight, "synthetic": 1.0 - arguments.real_weight},
        length=arguments.mixture_samples,
        seed=arguments.seed,
    )
    loader = DataLoader(
        mixture,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    model_config = BilateralFlowConfig(
        base_channels=24,
        max_flow=20.0,
        coarse_velocity=arguments.coarse_velocity,
        correlation_radius=arguments.correlation_radius,
        correspondence_limit=arguments.correspondence_limit,
    )
    model = GenFramesBilateralFlow(model_config)
    initial_checkpoint = (
        layout.checkpoints / arguments.initial_checkpoint_id / "model.safetensors"
    )
    incompatible = model.load_state_dict(load_file(initial_checkpoint), strict=False)
    allowed_keys: set[str] = set()
    if arguments.coarse_velocity:
        allowed_keys.update({"coarse_head.weight", "coarse_head.bias"})
    if arguments.correlation_radius > 0:
        allowed_keys.update(
            key
            for key in model.state_dict()
            if key.startswith(("match_encoder.", "correspondence_body.", "correspondence_head."))
        )
    if not set(incompatible.missing_keys).issubset(allowed_keys) or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible initial checkpoint: {incompatible}")
    train_config = TrainConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        amp=True,
        seed=arguments.seed,
        log_every=arguments.log_every,
    )
    loss_config = LossConfig(
        bilateral_flow_weight=arguments.flow_weight,
        warp_oracle_weight=arguments.warp_oracle_weight,
    )
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def report(step: int, losses: dict[str, float]) -> None:
        print(json.dumps({"step": step, **losses}), flush=True)

    started = time.perf_counter()
    training_result = train_steps(
        model,
        loader,
        device=device,
        config=train_config,
        criterion=InterpolationLoss(loss_config),
        callback=report,
    )
    wall_seconds = time.perf_counter() - started
    real_validation = ManifestFrameDataset(
        manifest,
        dataset_root=dataset_root,
        split=Split.VALIDATION,
        crop_size=crop,
        seed=arguments.seed + 1,
    )
    real_validation = Subset(real_validation, range(min(256, len(real_validation))))
    synthetic_validation = AnalyticMotionDataset(
        length=256,
        height=arguments.crop_size,
        width=arguments.crop_size,
        seed=arguments.seed + 1_000_000,
    )
    real_loader = DataLoader(real_validation, batch_size=arguments.batch_size)
    synthetic_loader = DataLoader(synthetic_validation, batch_size=arguments.batch_size)
    destination.mkdir(parents=True)
    checkpoint = destination / "model.safetensors"
    checkpoint_tensors = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    save_file(checkpoint_tensors, checkpoint)
    result = {
        "experiment_id": arguments.experiment_id,
        "model": asdict(model_config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": _sha256(initial_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training": asdict(train_config),
        "loss": asdict(loss_config),
        "data": {
            "real_weight": arguments.real_weight,
            "synthetic_weight": 1.0 - arguments.real_weight,
            "mixture_samples": arguments.mixture_samples,
            "crop_size": arguments.crop_size,
        },
        "train_result": asdict(training_result),
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "validation": {
            "real_linear": asdict(evaluate_model(LinearBlend(), real_loader, device=device)),
            "real_model": asdict(evaluate_model(model, real_loader, device=device)),
            "synthetic_linear": asdict(
                evaluate_model(LinearBlend(), synthetic_loader, device=device)
            ),
            "synthetic_model": asdict(evaluate_model(model, synthetic_loader, device=device)),
        },
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="prism-bilateral-flow-davis-mixed-001")
    parser.add_argument(
        "--initial-checkpoint-id", default="orbit-bilateral-flow-analytic-003"
    )
    parser.add_argument("--storage-root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mixture-samples", type=int, default=4096)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--real-weight", type=float, default=0.75)
    parser.add_argument("--flow-weight", type=float, default=0.01)
    parser.add_argument("--warp-oracle-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--coarse-velocity", action="store_true")
    parser.add_argument("--correlation-radius", type=int, default=0)
    parser.add_argument("--correspondence-limit", type=float, default=16.0)
    arguments = parser.parse_args()
    if not 0.0 < arguments.real_weight < 1.0:
        parser.error("--real-weight must lie inside (0, 1)")
    return arguments


if __name__ == "__main__":
    main()
