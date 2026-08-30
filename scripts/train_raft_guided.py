"""Train the GenFrames-owned fusion head over frozen RAFT global correspondence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from genframes.data import (
    AnalyticMotionDataset,
    DatasetManifest,
    ManifestFrameDataset,
    MixedFrameDataset,
    Split,
)
from genframes.eval import evaluate_model
from genframes.models import GenFramesRaftGuided, LinearBlend, RaftGuidedConfig
from genframes.storage import StorageLayout
from genframes.training import InterpolationLoss, LossConfig, TrainConfig, train_steps


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    layout.create()
    destination = layout.checkpoints / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {destination}")

    crop = (arguments.crop_size, arguments.crop_size)
    gopro_root = layout.datasets / "gopro-large-all"
    davis_root = layout.datasets / "davis-2017"
    gopro_manifest_path = gopro_root / "genframes-manifest.json"
    davis_manifest_path = davis_root / "genframes-manifest.json"
    gopro_manifest = DatasetManifest.load(gopro_manifest_path)
    davis_manifest = DatasetManifest.load(davis_manifest_path)
    gopro_train = ManifestFrameDataset(
        gopro_manifest,
        dataset_root=gopro_root,
        split=Split.TRAIN,
        crop_size=crop,
        seed=arguments.seed,
        horizontal_flip=True,
    )
    davis_train = ManifestFrameDataset(
        davis_manifest,
        dataset_root=davis_root / "DAVIS",
        split=Split.TRAIN,
        crop_size=crop,
        seed=arguments.seed + 1,
        horizontal_flip=True,
    )
    real_train = MixedFrameDataset(
        {"gopro": gopro_train, "davis": davis_train},
        weights={"gopro": arguments.gopro_weight, "davis": 1.0 - arguments.gopro_weight},
        length=arguments.mixture_samples,
        seed=arguments.seed + 2,
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
        num_workers=arguments.workers,
        pin_memory=True,
        persistent_workers=arguments.workers > 0,
    )

    model_config = RaftGuidedConfig(fusion_channels=arguments.fusion_channels)
    model = GenFramesRaftGuided(model_config)
    train_config = TrainConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        amp=True,
        seed=arguments.seed,
        log_every=arguments.log_every,
    )
    loss_config = LossConfig(
        bilateral_flow_weight=0.0,
        warp_oracle_weight=0.0,
        edge_weight=arguments.edge_weight,
        visibility_weight=arguments.visibility_weight,
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

    davis_validation = ManifestFrameDataset(
        davis_manifest,
        dataset_root=davis_root / "DAVIS",
        split=Split.VALIDATION,
        crop_size=crop,
        seed=arguments.seed + 3,
    )
    davis_validation = Subset(
        davis_validation, range(min(arguments.validation_samples, len(davis_validation)))
    )
    synthetic_validation = AnalyticMotionDataset(
        length=arguments.validation_samples,
        height=arguments.crop_size,
        width=arguments.crop_size,
        seed=arguments.seed + 1_000_000,
    )
    real_loader = DataLoader(davis_validation, batch_size=arguments.batch_size)
    synthetic_loader = DataLoader(synthetic_validation, batch_size=arguments.batch_size)

    destination.mkdir(parents=True)
    checkpoint = destination / "fusion.safetensors"
    save_file(
        {name: value.detach().cpu() for name, value in model.fusion_state_dict().items()},
        checkpoint,
    )
    result = {
        "experiment_id": arguments.experiment_id,
        "model": asdict(model_config),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_fusion": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "motion_dependency": {
            "name": "TorchVision RAFT-small C_T_V2",
            "weights_id": model.flow_weights_id,
            "weights_url": model.flow_weights_url,
            "bundled_in_checkpoint": False,
            "frozen": True,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training": asdict(train_config),
        "loss": asdict(loss_config),
        "data": {
            "real_weight": arguments.real_weight,
            "synthetic_weight": 1.0 - arguments.real_weight,
            "gopro_within_real_weight": arguments.gopro_weight,
            "davis_within_real_weight": 1.0 - arguments.gopro_weight,
            "gopro_manifest": str(gopro_manifest_path),
            "davis_manifest": str(davis_manifest_path),
            "mixture_samples": arguments.mixture_samples,
            "crop_size": arguments.crop_size,
        },
        "train_result": asdict(training_result),
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "validation": {
            "davis_linear": asdict(evaluate_model(LinearBlend(), real_loader, device=device)),
            "davis_model": asdict(evaluate_model(model, real_loader, device=device)),
            "synthetic_linear": asdict(
                evaluate_model(LinearBlend(), synthetic_loader, device=device)
            ),
            "synthetic_model": asdict(evaluate_model(model, synthetic_loader, device=device)),
        },
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="prism-raft-guided-fusion-gopro-mixed-001")
    parser.add_argument("--storage-root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixture-samples", type=int, default=8192)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--fusion-channels", type=int, default=24)
    parser.add_argument("--real-weight", type=float, default=0.85)
    parser.add_argument("--gopro-weight", type=float, default=0.8)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--visibility-weight", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--log-every", type=int, default=100)
    arguments = parser.parse_args()
    for name in ("real_weight", "gopro_weight"):
        if not 0.0 < getattr(arguments, name) < 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie inside (0, 1)")
    return arguments


if __name__ == "__main__":
    main()
