"""Evaluate the immutable bilateral-flow baseline on a real-data manifest."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.eval import evaluate_model
from genframes.models import BilateralFlowConfig, GenFramesBilateralFlow, LinearBlend
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    manifest_path = Path(arguments.manifest) if arguments.manifest else (
        layout.datasets / "davis-2017" / "genframes-manifest.json"
    )
    dataset_root = Path(arguments.dataset_root) if arguments.dataset_root else (
        layout.datasets / "davis-2017" / "DAVIS"
    )
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else (
        layout.checkpoints / "orbit-bilateral-flow-analytic-003" / "model.safetensors"
    )
    dataset = ManifestFrameDataset(
        DatasetManifest.load(manifest_path), dataset_root=dataset_root, split=Split.VALIDATION
    )
    selected = _stable_subset(dataset, arguments.max_samples, arguments.seed)
    loader = DataLoader(selected, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(arguments.device)
    model = GenFramesBilateralFlow(BilateralFlowConfig(base_channels=24, max_flow=20.0))
    model.load_state_dict(load_file(checkpoint))
    result = {
        "dataset": {
            "manifest": str(manifest_path),
            "manifest_samples": len(dataset),
            "evaluated_samples": len(selected),
            "subset_seed": arguments.seed,
            "moving_region_definition": "mean endpoint RGB difference > 8/255",
            "object_region_definition": "DAVIS target-frame non-background annotation",
        },
        "linear_blend": asdict(evaluate_model(LinearBlend(), loader, device=device)),
        "bilateral_flow": asdict(evaluate_model(model, loader, device=device)),
        "runtime": _profile(model, selected, device, arguments.profile_samples),
        "checkpoint": str(checkpoint),
    }
    destination = Path(arguments.output) if arguments.output else (
        layout.benchmarks / "prism-davis-baseline-001" / "result.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _stable_subset(dataset: ManifestFrameDataset, count: int, seed: int) -> Subset:
    count = min(count, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), count)
    return Subset(dataset, sorted(indices))


def _profile(
    model: GenFramesBilateralFlow,
    dataset: Subset,
    device: torch.device,
    count: int,
) -> dict[str, float | int | str]:
    model = model.to(device).eval()
    latencies: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in range(min(count + 3, len(dataset))):
            sample = dataset[index]
            frame0 = sample["frame0"].unsqueeze(0).to(device)
            frame1 = sample["frame1"].unsqueeze(0).to(device)
            target_time = sample["time"].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(frame0, frame1, target_time)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= 3:
                latencies.append((time.perf_counter() - started) * 1000.0)
    return {
        "samples": len(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p90_ms": float(np.percentile(latencies, 90)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "timing_policy": "model-only batch-1; host-to-device and image decode excluded",
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--profile-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2405)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
