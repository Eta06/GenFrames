"""Component-level diagnostics for the sealed Prism hard-motion A/B cases."""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.models import BilateralFlowConfig, GenFramesBilateralFlow
from genframes.storage import StorageLayout

BASELINE_ID = "prism-bilateral-flow-davis-mixed-001"
CHALLENGER_ID = "prism-bilateral-flow-coarse-davis-mixed-001"
ORACLE_WARP_ID = "prism-coarse-oracle-warp-davis-mixed-001"
LOCAL_CORRELATION_ID = "prism-coarse-local-correlation-davis-mixed-001"
MODEL_SPECS = {
    BASELINE_ID: {"coarse": False, "radius": 0},
    CHALLENGER_ID: {"coarse": True, "radius": 0},
    ORACLE_WARP_ID: {"coarse": True, "radius": 0},
    LOCAL_CORRELATION_ID: {"coarse": True, "radius": 4},
}


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    package = layout.benchmarks / arguments.ab_experiment
    responses = json.loads((package / "responses.json").read_text(encoding="utf-8"))
    if any(response["preference"] != "both-bad" for response in responses["responses"]):
        raise ValueError("this diagnostic is scoped to the sealed unanimous both-bad response")
    destination = package / arguments.output_name
    destination.mkdir(parents=True, exist_ok=True)
    dataset = ManifestFrameDataset(
        DatasetManifest.load(layout.datasets / "davis-2017" / "genframes-manifest.json"),
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
    )
    selected = _select_cases(dataset, subset_seed=2405, count=6)
    device = torch.device(arguments.device)
    models = {
        model_id: _load_model(layout, model_id, device=device, **spec)
        for model_id, spec in MODEL_SPECS.items()
    }
    cases: list[dict[str, object]] = []
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
            frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
            target = _tensor(sample, "target").unsqueeze(0).to(device)
            target_time = _tensor(sample, "time").unsqueeze(0).to(device)
            outputs = {
                name: model(frame0, frame1, target_time) for name, model in models.items()
            }
            metrics = {
                name: _component_metrics(frame0, frame1, target, output)
                for name, output in outputs.items()
            }
            case_id = f"case-{ordinal:02d}"
            diagnostic_path = destination / f"{case_id}.png"
            _save_diagnostic_panel(
                diagnostic_path,
                frame0[0],
                target[0],
                frame1[0],
                outputs[LOCAL_CORRELATION_ID],
            )
            cases.append(
                {
                    "case_id": case_id,
                    "sequence_id": sample["sequence_id"],
                    "diagnostic_image": str(diagnostic_path),
                    "models": metrics,
                }
            )
    analysis = {
        "experiment_id": f"{arguments.ab_experiment}/{arguments.output_name}",
        "definitions": {
            "unwarped_oracle": "per-pixel endpoint with lower RGB squared error to GT",
            "warp_oracle": "per-pixel predicted warp with lower RGB squared error to GT",
            "flow_alignment_gain_db": "warp-oracle PSNR minus unwarped-oracle PSNR",
            "fusion_gap_db": "warp-oracle PSNR minus final prediction PSNR",
            "residual_gain_db": "final prediction PSNR minus pre-residual fusion PSNR",
            "detail_ratio": "prediction gradient L1 energy divided by GT gradient energy",
            "disagreement": "mean RGB difference between endpoint warps above 0.1",
        },
        "cases": cases,
        "aggregate": _aggregate(cases),
    }
    (destination / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))


def _component_metrics(frame0, frame1, target, output) -> dict[str, float]:
    prediction = output.frame.clamp(0, 1)
    auxiliary = output.auxiliary
    warped0 = auxiliary["warped0"].clamp(0, 1)
    warped1 = auxiliary["warped1"].clamp(0, 1)
    weight1 = auxiliary["weight1"]
    residual = auxiliary["residual"]
    fused = ((1.0 - weight1) * warped0 + weight1 * warped1).clamp(0, 1)
    warp_oracle = _pixel_oracle(warped0, warped1, target)
    unwarped_oracle = _pixel_oracle(frame0, frame1, target)
    disagreement = (warped0 - warped1).abs().mean(1, keepdim=True) > 0.1
    flow = auxiliary["velocity"]
    flow_magnitude = torch.linalg.vector_norm(flow, dim=1)
    prediction_psnr = _psnr(prediction, target)
    warp_oracle_psnr = _psnr(warp_oracle, target)
    unwarped_oracle_psnr = _psnr(unwarped_oracle, target)
    return {
        "prediction_mae": _mae(prediction, target),
        "prediction_psnr_db": prediction_psnr,
        "unwarped_oracle_psnr_db": unwarped_oracle_psnr,
        "warp_oracle_psnr_db": warp_oracle_psnr,
        "flow_alignment_gain_db": warp_oracle_psnr - unwarped_oracle_psnr,
        "fusion_gap_db": warp_oracle_psnr - prediction_psnr,
        "pre_residual_psnr_db": _psnr(fused, target),
        "residual_gain_db": prediction_psnr - _psnr(fused, target),
        "warped0_mae": _mae(warped0, target),
        "warped1_mae": _mae(warped1, target),
        "warp_disagreement_fraction": float(disagreement.float().mean()),
        "disagreement_region_mae": _masked_mae(prediction, target, disagreement),
        "flow_magnitude_mean_px_per_interval": float(flow_magnitude.mean()),
        "flow_magnitude_p95_px_per_interval": float(torch.quantile(flow_magnitude, 0.95)),
        "blend_weight_mean": float(weight1.mean()),
        "blend_weight_std": float(weight1.std()),
        "blend_saturation_fraction": float(((weight1 < 0.1) | (weight1 > 0.9)).float().mean()),
        "residual_abs_mean": float(residual.abs().mean()),
        "detail_ratio": _gradient_energy(prediction) / max(_gradient_energy(target), 1e-12),
    }


def _pixel_oracle(first: torch.Tensor, second: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    first_error = (first - target).square().mean(1, keepdim=True)
    second_error = (second - target).square().mean(1, keepdim=True)
    return torch.where(first_error <= second_error, first, second)


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).abs().mean())


def _masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((prediction - target).abs().mean(1, keepdim=True)[mask].mean())


def _gradient_energy(frame: torch.Tensor) -> float:
    horizontal = (frame[:, :, :, 1:] - frame[:, :, :, :-1]).abs().mean()
    vertical = (frame[:, :, 1:, :] - frame[:, :, :-1, :]).abs().mean()
    return float(horizontal + vertical)


def _aggregate(cases: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model_id in MODEL_SPECS:
        model_metrics = [case["models"][model_id] for case in cases]
        keys = model_metrics[0].keys()
        result[model_id] = {
            key: sum(float(metrics[key]) for metrics in model_metrics) / len(model_metrics)
            for key in keys
        }
    return result


def _select_cases(dataset: ManifestFrameDataset, *, subset_seed: int, count: int) -> list[int]:
    candidates = random.Random(subset_seed).sample(range(len(dataset)), min(300, len(dataset)))
    ranked = sorted(candidates, key=lambda index: _stress_score(dataset[index]), reverse=True)
    selected: list[int] = []
    sequences: set[str] = set()
    for index in ranked:
        sequence = str(dataset[index]["sequence_id"])
        if sequence not in sequences:
            selected.append(index)
            sequences.add(sequence)
        if len(selected) == count:
            break
    return selected


def _stress_score(sample: dict[str, torch.Tensor | str]) -> float:
    moving = _tensor(sample, "moving_mask")
    objects = _tensor(sample, "object_mask")
    return float(moving.float().mean() + 0.5 * objects.float().mean())


def _load_model(layout, checkpoint_id, *, coarse, radius, device):
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=24,
            max_flow=20.0,
            coarse_velocity=coarse,
            correlation_radius=radius,
        )
    )
    model.load_state_dict(load_file(layout.checkpoints / checkpoint_id / "model.safetensors"))
    return model.to(device).eval()


def _save_diagnostic_panel(path, frame0, target, frame1, output) -> None:
    auxiliary = output.auxiliary
    prediction = output.frame[0].clamp(0, 1)
    warped0 = auxiliary["warped0"][0].clamp(0, 1)
    warped1 = auxiliary["warped1"][0].clamp(0, 1)
    oracle = _pixel_oracle(warped0[None], warped1[None], target[None])[0]
    error = (prediction - target).abs().mean(0)
    flow = auxiliary["velocity"][0]
    weight = auxiliary["weight1"][0, 0]
    residual = auxiliary["residual"][0]
    panels = (
        ("Input 0", _rgb_image(frame0)),
        ("Ground truth", _rgb_image(target)),
        ("Input 1", _rgb_image(frame1)),
        ("Prediction", _rgb_image(prediction)),
        ("Absolute error", _heat_image(error, maximum=0.3)),
        ("Warp 0", _rgb_image(warped0)),
        ("Warp 1", _rgb_image(warped1)),
        ("GT oracle warp", _rgb_image(oracle)),
        ("Velocity", _flow_image(flow)),
        ("Blend weight", _gray_image(weight)),
        ("Residual x5", _rgb_image((residual * 5.0 + 0.5).clamp(0, 1))),
        ("Warp disagreement", _heat_image((warped0 - warped1).abs().mean(0), maximum=0.3)),
    )
    tile_width, tile_height, header, columns = 300, 190, 26, 4
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 7, row * (tile_height + header) + 6), label, fill="white")
    canvas.save(path)


def _rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _gray_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def _heat_image(tensor: torch.Tensor, *, maximum: float) -> Image.Image:
    value = (tensor.detach().cpu().numpy() / maximum).clip(0, 1)
    red = value
    green = np.clip(2.0 * value - 0.5, 0, 1)
    blue = np.clip(4.0 * value - 3.0, 0, 1)
    return Image.fromarray((np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8))


def _flow_image(flow: torch.Tensor) -> Image.Image:
    x, y = flow.detach().cpu().numpy()
    angle = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    magnitude = np.sqrt(x * x + y * y)
    value = np.clip(magnitude / 20.0, 0, 1)
    hsv = np.stack((angle, np.ones_like(value), value), axis=-1)
    hsv = (hsv * 255).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _tensor(sample: dict[str, torch.Tensor | str], key: str) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"sample field {key!r} must be a tensor")
    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--ab-experiment", default="prism-human-ab-002")
    parser.add_argument("--output-name", default="diagnostics-local-correlation-001")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
