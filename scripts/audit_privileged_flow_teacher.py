"""Audit whether a GT-privileged optical-flow teacher provides usable VFI warps."""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.ops import backward_warp
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    destination = layout.benchmarks / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite audit: {destination}")
    destination.mkdir(parents=True)
    dataset = ManifestFrameDataset(
        DatasetManifest.load(layout.datasets / "davis-2017" / "genframes-manifest.json"),
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
    )
    selected = _select_cases(dataset, subset_seed=2405, count=6)
    device = torch.device(arguments.device)
    model, weights = _load_teacher(arguments.variant, device)
    transforms = weights.transforms()
    cases = []
    started = time.perf_counter()
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
            frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
            target = _tensor(sample, "target").unsqueeze(0).to(device)
            flow_t0, flow_t1 = _estimate_target_flows(
                model,
                transforms,
                arguments.mode,
                frame0,
                frame1,
                target,
                _tensor(sample, "time").to(device),
            )
            warped0 = backward_warp(frame0, flow_t0)
            warped1 = backward_warp(frame1, flow_t1)
            oracle = _pixel_oracle(warped0, warped1, target)
            target_time = _tensor(sample, "time").to(device).reshape(-1, 1, 1, 1)
            blended = (
                (1.0 - target_time) * warped0 + target_time * warped1
            ).clamp(0, 1)
            unwarped_oracle = _pixel_oracle(frame0, frame1, target)
            unwarped_psnr = _psnr(unwarped_oracle, target)
            warped_psnr = _psnr(oracle, target)
            case_id = f"case-{ordinal:02d}"
            panel = destination / f"{case_id}.png"
            _save_panel(
                panel,
                frame0[0],
                target[0],
                frame1[0],
                warped0[0],
                warped1[0],
                oracle[0],
                blended[0],
                flow_t0[0],
            )
            cases.append(
                {
                    "case_id": case_id,
                    "sequence_id": sample["sequence_id"],
                    "dataset_index": index,
                    "unwarped_oracle_psnr_db": unwarped_psnr,
                    "teacher_warp_oracle_psnr_db": warped_psnr,
                    "flow_alignment_gain_db": warped_psnr - unwarped_psnr,
                    "teacher_warp_oracle_mae": float((oracle - target).abs().mean()),
                    "time_blend_psnr_db": _psnr(blended, target),
                    "time_blend_mae": float((blended - target).abs().mean()),
                    "flow_t0_p95_px": _flow_p95(flow_t0),
                    "flow_t1_p95_px": _flow_p95(flow_t1),
                    "panel": str(panel),
                }
            )
    gains = [case["flow_alignment_gain_db"] for case in cases]
    blend_psnr = [case["time_blend_psnr_db"] for case in cases]
    result = {
        "experiment_id": arguments.experiment_id,
        "purpose": (
            "training-only privileged teacher ceiling"
            if arguments.mode == "privileged"
            else "inference-available endpoint global-flow interpolation ceiling"
        ),
        "teacher": {
            "name": f"torchvision RAFT {arguments.variant}",
            "weights": str(weights),
            "weights_url": weights.url,
            "license": "BSD-3-Clause implementation; pretrained weight terms follow TorchVision",
            "mode": arguments.mode,
            "input_pair": (
                "ground-truth target frame -> each endpoint frame"
                if arguments.mode == "privileged"
                else "endpoint frame 0 <-> endpoint frame 1; no target access"
            ),
        },
        "cases": cases,
        "aggregate": {
            "flow_alignment_gain_db": sum(gains) / len(gains),
            "positive_cases": sum(gain > 0 for gain in gains),
            "minimum_gain_db": min(gains),
            "time_blend_psnr_db": sum(blend_psnr) / len(blend_psnr),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _load_teacher(variant: str, device: torch.device):
    if variant == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    return model.to(device).eval(), weights


def _estimate_target_flows(
    model,
    transforms,
    mode: str,
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    target: torch.Tensor,
    target_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "privileged":
        target_input, frame0_input, original_size = _prepare_pair(
            target, frame0, transforms
        )
        _, frame1_input, _ = _prepare_pair(target, frame1, transforms)
        crop = (..., slice(0, original_size[0]), slice(0, original_size[1]))
        return model(target_input, frame0_input)[-1][crop], model(
            target_input, frame1_input
        )[-1][crop]
    frame0_input, frame1_input, original_size = _prepare_pair(
        frame0, frame1, transforms
    )
    crop = (..., slice(0, original_size[0]), slice(0, original_size[1]))
    flow01 = model(frame0_input, frame1_input)[-1][crop]
    flow10 = model(frame1_input, frame0_input)[-1][crop]
    time = target_time.reshape(-1, 1, 1, 1)
    flow_t0 = -(1.0 - time) * time * flow01 + time.square() * flow10
    flow_t1 = (1.0 - time).square() * flow01 - time * (1.0 - time) * flow10
    return flow_t0, flow_t1


def _prepare_pair(first, second, transforms):
    height, width = first.shape[-2:]
    padded_height = math.ceil(height / 8) * 8
    padded_width = math.ceil(width / 8) * 8
    padding = (0, padded_width - width, 0, padded_height - height)
    first = functional.pad(first, padding, mode="replicate")
    second = functional.pad(second, padding, mode="replicate")
    first, second = transforms(first, second)
    return first, second, (height, width)


def _pixel_oracle(first, second, target):
    first_error = (first - target).square().mean(1, keepdim=True)
    second_error = (second - target).square().mean(1, keepdim=True)
    return torch.where(first_error <= second_error, first, second)


def _psnr(prediction, target) -> float:
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _flow_p95(flow) -> float:
    return float(torch.quantile(torch.linalg.vector_norm(flow, dim=1), 0.95))


def _select_cases(dataset, *, subset_seed: int, count: int) -> list[int]:
    candidates = random.Random(subset_seed).sample(range(len(dataset)), min(300, len(dataset)))
    ranked = sorted(candidates, key=lambda index: _stress_score(dataset[index]), reverse=True)
    selected = []
    sequences = set()
    for index in ranked:
        sequence = str(dataset[index]["sequence_id"])
        if sequence not in sequences:
            selected.append(index)
            sequences.add(sequence)
        if len(selected) == count:
            break
    return selected


def _stress_score(sample) -> float:
    return float(
        _tensor(sample, "moving_mask").float().mean()
        + 0.5 * _tensor(sample, "object_mask").float().mean()
    )


def _save_panel(
    path, frame0, target, frame1, warped0, warped1, oracle, blended, flow0
) -> None:
    error = (oracle - target).abs().mean(0)
    panels = (
        ("Input 0", _rgb(frame0)),
        ("Ground truth", _rgb(target)),
        ("Input 1", _rgb(frame1)),
        ("Teacher oracle", _rgb(oracle)),
        ("Time blend", _rgb(blended)),
        ("Teacher warp 0", _rgb(warped0)),
        ("Teacher warp 1", _rgb(warped1)),
        ("Oracle error", _heat(error, maximum=0.3)),
        ("Teacher flow t->0", _flow(flow0)),
    )
    tile_width, tile_height, header, columns = 300, 190, 26, 5
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + header)), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 7, row * (tile_height + header) + 6), label, fill="white")
    canvas.save(path)


def _rgb(tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
    array = array.round().astype(np.uint8)
    return Image.fromarray(array)


def _heat(tensor, *, maximum: float) -> Image.Image:
    value = (tensor.detach().cpu().numpy() / maximum).clip(0, 1)
    colors = np.stack(
        (value, np.clip(2 * value - 0.5, 0, 1), np.clip(4 * value - 3, 0, 1)),
        axis=-1,
    )
    return Image.fromarray((colors * 255).astype(np.uint8))


def _flow(flow) -> Image.Image:
    x, y = flow.detach().cpu().numpy()
    angle = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    magnitude = np.clip(np.sqrt(x * x + y * y) / 40.0, 0, 1)
    hsv = (np.stack((angle, np.ones_like(magnitude), magnitude), axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _tensor(sample, key: str) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"sample field {key!r} must be a tensor")
    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--experiment-id", default="prism-privileged-raft-small-audit-001")
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument(
        "--mode", choices=("privileged", "endpoint-quadratic"), default="privileged"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
