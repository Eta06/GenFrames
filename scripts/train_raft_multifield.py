"""Train only the Parallax selector over fixed RAFT multi-field candidates."""

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
from genframes.models import (
    GenFramesRaftEvidenceSelector,
    GenFramesRaftGuided,
    GenFramesRaftMultiField,
    GenFramesRaftRegionAssignment,
    LinearBlend,
    RaftEvidenceSelectorConfig,
    RaftMultiFieldConfig,
    RaftRegionAssignmentConfig,
)
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
    gopro_manifest = DatasetManifest.load(gopro_root / "genframes-manifest.json")
    davis_manifest = DatasetManifest.load(davis_root / "genframes-manifest.json")
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
    if arguments.region_assignment:
        model_config = RaftRegionAssignmentConfig(
            selector_channels=arguments.selector_channels,
            region_stride=arguments.region_stride,
        )
        model = GenFramesRaftRegionAssignment(model_config)
    elif arguments.evidence_selector:
        model_config = RaftEvidenceSelectorConfig(
            selector_channels=arguments.selector_channels
        )
        model = GenFramesRaftEvidenceSelector(model_config)
    else:
        model_config = RaftMultiFieldConfig(selector_channels=arguments.selector_channels)
        model = GenFramesRaftMultiField(model_config)
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
        candidate_selection_weight=arguments.selection_weight,
        candidate_static_weight=arguments.candidate_static_weight,
        candidate_soft_target_temperature=arguments.soft_target_temperature,
        selector_spatial_weight=arguments.spatial_weight,
        selector_spatial_edge_scale=arguments.spatial_edge_scale,
        region_assignment_weight=arguments.assignment_weight,
        unsupported_error_threshold=arguments.unsupported_error_threshold,
        ownership_weight=arguments.ownership_weight,
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
    destination.mkdir(parents=True)
    checkpoint = destination / "selector.safetensors"
    save_file(
        {name: value.detach().cpu() for name, value in model.selector_state_dict().items()},
        checkpoint,
    )
    validation = _validation(
        model,
        davis_manifest,
        davis_root,
        gopro_manifest,
        gopro_root,
        crop,
        arguments,
        device,
    )
    result = {
        "experiment_id": arguments.experiment_id,
        "git_keyword": "Parallax",
        "model": asdict(model_config),
        "parameters": {
            "total_research_oracle": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_selector": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "motion_dependency": {
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
            "mixture_samples": arguments.mixture_samples,
            "crop_size": arguments.crop_size,
        },
        "train_result": asdict(training_result),
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "validation": validation,
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def _validation(model, davis_manifest, davis_root, gopro_manifest, gopro_root, crop, args, device):
    davis = ManifestFrameDataset(
        davis_manifest,
        dataset_root=davis_root / "DAVIS",
        split=Split.VALIDATION,
        crop_size=crop,
        seed=args.seed + 3,
    )
    gopro = ManifestFrameDataset(
        gopro_manifest,
        dataset_root=gopro_root,
        split=Split.TEST,
        crop_size=crop,
        seed=args.seed + 4,
    )
    davis = Subset(davis, range(min(args.validation_samples, len(davis))))
    gopro = Subset(gopro, range(min(args.gopro_validation_samples, len(gopro))))
    analytic = AnalyticMotionDataset(
        length=args.validation_samples,
        height=args.crop_size,
        width=args.crop_size,
        seed=args.seed + 1_000_000,
    )
    loaders = {
        "davis": DataLoader(davis, batch_size=args.batch_size),
        "gopro": DataLoader(gopro, batch_size=args.batch_size),
        "analytic": DataLoader(analytic, batch_size=args.batch_size),
    }
    single = GenFramesRaftGuided()
    return {
        name: {
            "linear": asdict(evaluate_model(LinearBlend(), loader, device=device)),
            "single_field": asdict(evaluate_model(single, loader, device=device)),
            "multi_field_selector": asdict(evaluate_model(model, loader, device=device)),
        }
        for name, loader in loaders.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="parallax-raft-multifield-selector-gopro-001")
    parser.add_argument("--storage-root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixture-samples", type=int, default=16384)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--gopro-validation-samples", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--selector-channels", type=int, default=32)
    parser.add_argument("--evidence-selector", action="store_true")
    parser.add_argument("--region-assignment", action="store_true")
    parser.add_argument("--region-stride", type=int, default=8)
    parser.add_argument("--real-weight", type=float, default=0.85)
    parser.add_argument("--gopro-weight", type=float, default=0.8)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--selection-weight", type=float, default=0.05)
    parser.add_argument("--candidate-static-weight", type=float, default=0.1)
    parser.add_argument("--soft-target-temperature", type=float, default=0.02)
    parser.add_argument("--spatial-weight", type=float, default=0.02)
    parser.add_argument("--spatial-edge-scale", type=float, default=10.0)
    parser.add_argument("--assignment-weight", type=float, default=0.1)
    parser.add_argument("--unsupported-error-threshold", type=float, default=0.04)
    parser.add_argument("--ownership-weight", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
