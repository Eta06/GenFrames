"""Create a reproducible blinded hard-motion comparison package."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.models import BilateralFlowConfig, GenFramesBilateralFlow
from genframes.storage import StorageLayout

BASELINE_ID = "prism-coarse-oracle-warp-davis-mixed-001"
CHALLENGER_ID = "prism-coarse-explicit-displacement-davis-mixed-001"


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    destination = layout.benchmarks / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite A/B package: {destination}")
    cases_root = destination / "cases"
    cases_root.mkdir(parents=True)
    manifest_path = layout.datasets / "davis-2017" / "genframes-manifest.json"
    dataset = ManifestFrameDataset(
        DatasetManifest.load(manifest_path),
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
    )
    candidate_indices = random.Random(arguments.subset_seed).sample(
        range(len(dataset)), min(300, len(dataset))
    )
    ranked = sorted(
        candidate_indices,
        key=lambda index: _stress_score(dataset[index]),
        reverse=True,
    )
    selected: list[int] = []
    selected_sequences: set[str] = set()
    for index in ranked:
        sequence_id = str(dataset[index]["sequence_id"])
        if sequence_id in selected_sequences:
            continue
        selected.append(index)
        selected_sequences.add(sequence_id)
        if len(selected) == arguments.cases:
            break
    device = torch.device(arguments.device)
    baseline = _load_model(layout, BASELINE_ID, coarse=True, device=device)
    challenger = _load_model(
        layout, CHALLENGER_ID, coarse=True, radius=4, moments=True, device=device
    )
    blind_random = random.Random(arguments.blinding_seed)
    public_cases: list[dict[str, object]] = []
    private_key: list[dict[str, str]] = []
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = sample["frame0"].unsqueeze(0).to(device)
            frame1 = sample["frame1"].unsqueeze(0).to(device)
            target_time = sample["time"].unsqueeze(0).to(device)
            baseline_frame = baseline(frame0, frame1, target_time).frame[0].clamp(0, 1).cpu()
            challenger_frame = challenger(frame0, frame1, target_time).frame[0].clamp(0, 1).cpu()
            challenger_is_a = blind_random.random() < 0.5
            candidate_a = challenger_frame if challenger_is_a else baseline_frame
            candidate_b = baseline_frame if challenger_is_a else challenger_frame
            case_id = f"case-{ordinal:02d}"
            output_path = cases_root / f"{case_id}.png"
            _save_panel(
                output_path,
                sample["frame0"],
                candidate_a,
                sample["target"],
                candidate_b,
                sample["frame1"],
            )
            public_cases.append(
                {
                    "case_id": case_id,
                    "image": str(output_path),
                    "sequence_id": sample["sequence_id"],
                    "stress_score": _stress_score(sample),
                }
            )
            private_key.append(
                {
                    "case_id": case_id,
                    "candidate_a": CHALLENGER_ID if challenger_is_a else BASELINE_ID,
                    "candidate_b": BASELINE_ID if challenger_is_a else CHALLENGER_ID,
                }
            )
    public_manifest = {
        "experiment_id": arguments.experiment_id,
        "instructions": "Choose A, B, tie, or both-bad without opening blinding-key.json.",
        "panel_order": ["input0", "candidate A", "ground truth", "candidate B", "input1"],
        "selection": "top hard-motion score within fixed DAVIS-300; unique sequences",
        "subset_seed": arguments.subset_seed,
        "blinding_seed_sha256_not_disclosed": True,
        "cases": public_cases,
    }
    (destination / "manifest.json").write_text(
        json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "blinding-key.json").write_text(
        json.dumps(private_key, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(public_manifest, indent=2))


def _load_model(
    layout: StorageLayout,
    checkpoint_id: str,
    *,
    coarse: bool,
    radius: int = 0,
    moments: bool = False,
    device: torch.device,
) -> GenFramesBilateralFlow:
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=24,
            max_flow=20.0,
            coarse_velocity=coarse,
            correlation_radius=radius,
            correlation_moments=moments,
        )
    )
    path = layout.checkpoints / checkpoint_id / "model.safetensors"
    model.load_state_dict(load_file(path))
    return model.to(device).eval()


def _stress_score(sample: dict[str, torch.Tensor | str]) -> float:
    moving = sample["moving_mask"]
    objects = sample["object_mask"]
    if not isinstance(moving, torch.Tensor) or not isinstance(objects, torch.Tensor):
        raise TypeError("regional masks must be tensors")
    return float(moving.float().mean() + 0.5 * objects.float().mean())


def _save_panel(path: Path, *frames: torch.Tensor | str) -> None:
    labels = ("Input 0", "Candidate A", "Ground truth", "Candidate B", "Input 1")
    images = [_to_pil(frame) for frame in frames]
    panel_width, panel_height, header = 300, 190, 28
    canvas = Image.new("RGB", (panel_width * len(images), panel_height + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        left = index * panel_width + (panel_width - image.width) // 2
        top = header + (panel_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((index * panel_width + 8, 7), label, fill="white")
    canvas.save(path)


def _to_pil(value: torch.Tensor | str) -> Image.Image:
    if not isinstance(value, torch.Tensor):
        raise TypeError("frame must be a tensor")
    array = (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="prism-human-ab-explicit-displacement-001")
    parser.add_argument("--storage-root")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--subset-seed", type=int, default=2405)
    parser.add_argument("--blinding-seed", type=int, default=9173)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
