"""Audit forward-projected endpoint motion before training another fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from analyze_failures import _mae, _psnr, _rgb_image, _select_cases, _tensor
from PIL import Image, ImageDraw

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.models import GenFramesRaftGuided
from genframes.ops import backward_warp, forward_splat
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    destination = layout.benchmarks / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {destination}")
    destination.mkdir(parents=True)
    dataset = ManifestFrameDataset(
        DatasetManifest.load(layout.datasets / "davis-2017" / "genframes-manifest.json"),
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
    )
    selected = _select_cases(dataset, subset_seed=2405, count=6)
    device = torch.device(arguments.device)
    model = GenFramesRaftGuided().to(device).eval()
    cases = []
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
            frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
            target = _tensor(sample, "target").unsqueeze(0).to(device)
            target_time = _tensor(sample, "time").reshape(1, 1, 1, 1).to(device)
            quadratic = model(frame0, frame1, target_time.flatten()).frame.clamp(0, 1)
            flow01, flow10 = model._endpoint_flows(frame0, frame1)
            variants = {}
            images = {}
            for alpha in arguments.alphas:
                prediction, evidence = _project(
                    frame0, frame1, flow01, flow10, target_time, quadratic, alpha
                )
                name = f"forward-alpha-{alpha:g}"
                variants[name] = {
                    "prediction_psnr_db": _psnr(prediction, target),
                    "prediction_mae": _mae(prediction, target),
                    "delta_vs_quadratic_psnr_db": _psnr(prediction, target)
                    - _psnr(quadratic, target),
                    "hole_fraction": float(evidence["hole"].float().mean()),
                    "collision_fraction": float(evidence["collision"].float().mean()),
                    "disagreement_region_mae": _masked_mae(
                        prediction, target, evidence["disagreement"]
                    ),
                }
                images[name] = prediction[0]
            case_id = f"case-{ordinal:02d}"
            _save_panel(
                destination / f"{case_id}.png",
                frame0[0],
                target[0],
                frame1[0],
                quadratic[0],
                images,
            )
            cases.append(
                {
                    "case_id": case_id,
                    "sequence_id": sample["sequence_id"],
                    "quadratic": {
                        "prediction_psnr_db": _psnr(quadratic, target),
                        "prediction_mae": _mae(quadratic, target),
                    },
                    "variants": variants,
                }
            )
    names = tuple(cases[0]["variants"])
    result = {
        "experiment_id": arguments.experiment_id,
        "hypothesis": (
            "forward target-time projection with mass/hole evidence avoids quadratic "
            "backward-flow reversal artifacts"
        ),
        "cases": cases,
        "aggregate": {
            name: {
                key: sum(case["variants"][name][key] for case in cases) / len(cases)
                for key in cases[0]["variants"][name]
            }
            for name in names
        },
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _project(frame0, frame1, flow01, flow10, target_time, fallback, alpha):
    consistency0 = (frame0 - backward_warp(frame1, flow01)).abs().mean(1, keepdim=True)
    consistency1 = (frame1 - backward_warp(frame0, flow10)).abs().mean(1, keepdim=True)
    importance0 = torch.exp(-alpha * consistency0)
    importance1 = torch.exp(-alpha * consistency1)
    splat0, mass0 = forward_splat(frame0, target_time * flow01, importance=importance0)
    splat1, mass1 = forward_splat(
        frame1, (1.0 - target_time) * flow10, importance=importance1
    )
    weighted0 = (1.0 - target_time) * mass0
    weighted1 = target_time * mass1
    mass = weighted0 + weighted1
    prediction = (weighted0 * splat0 + weighted1 * splat1) / mass.clamp_min(1e-6)
    hole = mass < 1e-3
    prediction = torch.where(hole, fallback, prediction).clamp(0, 1)
    return prediction, {
        "hole": hole,
        "collision": mass > 1.5,
        "disagreement": (splat0 - splat1).abs().mean(1, keepdim=True) > 0.1,
    }


def _masked_mae(prediction, target, mask) -> float:
    if not mask.any():
        return 0.0
    return float((prediction - target).abs().mean(1, keepdim=True)[mask].mean())


def _save_panel(path: Path, frame0, target, frame1, quadratic, variants) -> None:
    panels = [
        ("Input 0", _rgb_image(frame0)),
        ("Ground truth", _rgb_image(target)),
        ("Input 1", _rgb_image(frame1)),
        ("Quadratic", _rgb_image(quadratic)),
    ]
    panels.extend((name, _rgb_image(image)) for name, image in variants.items())
    tile_width, tile_height, header = 400, 240, 26
    canvas = Image.new("RGB", (tile_width * 2, (tile_height + header) * 4), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % 2, index // 2
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 7, row * (tile_height + header) + 6), label, fill="white")
    canvas.save(path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--experiment-id", default="prism-forward-projection-audit-001")
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.0, 10.0, 20.0, 40.0))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
