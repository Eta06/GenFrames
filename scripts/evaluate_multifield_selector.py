"""Evaluate a trained Parallax selector on the sealed DAVIS hard cases."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from analyze_failures import _heat_image, _rgb_image, _select_cases, _tensor
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from genframes.data import DatasetManifest, ManifestFrameDataset, Split
from genframes.models import GenFramesRaftGuided, GenFramesRaftMultiField
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
    single = GenFramesRaftGuided().to(device).eval()
    multi = GenFramesRaftMultiField()
    checkpoint = layout.checkpoints / arguments.checkpoint_id / "selector.safetensors"
    multi.load_selector_state_dict(load_file(checkpoint))
    multi = multi.to(device).eval()
    records = []
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
            frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
            target = _tensor(sample, "target").unsqueeze(0).to(device)
            target_time = _tensor(sample, "time").reshape(1).to(device)
            single_output = single(frame0, frame1, target_time)
            multi_output = multi(frame0, frame1, target_time)
            disagreement = (
                single_output.auxiliary["warped0"]
                - single_output.auxiliary["warped1"]
            ).abs().mean(1, keepdim=True) > 0.1
            candidates = multi_output.auxiliary["candidate_stack"]
            oracle = _oracle(candidates, target)
            record = {
                "case_id": f"case-{ordinal:02d}",
                "sequence_id": str(sample["sequence_id"]),
                "single_psnr_db": _psnr(single_output.frame, target),
                "multi_psnr_db": _psnr(multi_output.frame, target),
                "multi_minus_single_psnr_db": _psnr(multi_output.frame, target)
                - _psnr(single_output.frame, target),
                "oracle_psnr_db": _psnr(oracle, target),
                "single_mae": _mae(single_output.frame, target),
                "multi_mae": _mae(multi_output.frame, target),
                "high_disagreement_fraction": float(disagreement.float().mean()),
                "high_disagreement_single_mae": _masked_mae(
                    single_output.frame, target, disagreement
                ),
                "high_disagreement_multi_mae": _masked_mae(
                    multi_output.frame, target, disagreement
                ),
                "edge_ghosting_proxy_single": _masked_edge_error(
                    single_output.frame, target, disagreement
                ),
                "edge_ghosting_proxy_multi": _masked_edge_error(
                    multi_output.frame, target, disagreement
                ),
                "extra_field_weight_mean": float(
                    multi_output.auxiliary["candidate_weights"][:, 2:].sum(1).mean()
                ),
                "selector_entropy": float(
                    _entropy(multi_output.auxiliary["candidate_weights"]).mean()
                ),
            }
            records.append(record)
            _save_panel(
                destination / f"case-{ordinal:02d}.png",
                frame0[0],
                target[0],
                frame1[0],
                single_output.frame[0],
                multi_output.frame[0],
                oracle[0],
                multi_output.auxiliary["candidate_weights"][0],
            )
    aggregate = {
        key: statistics.fmean(float(record[key]) for record in records)
        for key in records[0]
        if key not in ("case_id", "sequence_id")
    }
    aggregate["cases_psnr_improved"] = sum(
        record["multi_minus_single_psnr_db"] > 0 for record in records
    )
    result = {
        "experiment_id": arguments.experiment_id,
        "checkpoint": str(checkpoint),
        "records": records,
        "aggregate": aggregate,
        "runtime": _profile(single, multi, dataset[selected[0]], device),
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _oracle(candidates, target):
    errors = (candidates - target[:, None]).abs().mean(2)
    index = errors.argmin(1)
    gather = index[:, None, None].expand(-1, 1, 3, -1, -1)
    return candidates.gather(1, gather).squeeze(1)


def _psnr(prediction, target) -> float:
    mse = (prediction.clamp(0, 1) - target).square().mean().clamp_min(1e-12)
    return float(10 * torch.log10(1 / mse))


def _mae(prediction, target) -> float:
    return float((prediction.clamp(0, 1) - target).abs().mean())


def _masked_mae(prediction, target, mask) -> float:
    error = (prediction.clamp(0, 1) - target).abs().mean(1, keepdim=True)
    return float(error[mask].mean()) if mask.any() else 0.0


def _masked_edge_error(prediction, target, mask) -> float:
    kernel = prediction.new_tensor(
        ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(3, 1, 3, 3)
    prediction_edge = functional.conv2d(prediction, kernel, padding=1, groups=3)
    target_edge = functional.conv2d(target, kernel, padding=1, groups=3)
    error = (prediction_edge - target_edge).abs().mean(1, keepdim=True)
    return float(error[mask].mean()) if mask.any() else 0.0


def _entropy(weights):
    return -(weights * weights.clamp_min(1e-8).log()).sum(1)


def _profile(single, multi, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1).to(device)
    timings = {"single": [], "multi": []}
    peak = {}
    with torch.inference_mode():
        for name, model in (("single", single), ("multi", multi)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for iteration in range(13):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(frame0, frame1, target_time)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if iteration >= 3:
                    timings[name].append((time.perf_counter() - started) * 1000)
            peak[name] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "resolution": list(frame0.shape[-2:]),
        "single_median_ms": statistics.median(timings["single"]),
        "multi_median_ms": statistics.median(timings["multi"]),
        "single_peak_allocated_bytes": peak["single"],
        "multi_peak_allocated_bytes": peak["multi"],
        "single_parameters": sum(parameter.numel() for parameter in single.parameters()),
        "multi_parameters": sum(parameter.numel() for parameter in multi.parameters()),
        "multi_trainable_selector_parameters": sum(
            parameter.numel() for parameter in multi.parameters() if parameter.requires_grad
        ),
    }


def _save_panel(path: Path, frame0, target, frame1, single, multi, oracle, weights) -> None:
    extra_weight = weights[2:].sum(0)
    selected = weights.argmax(0).float() / (weights.shape[0] - 1)
    panels = (
        ("Input 0", _rgb_image(frame0)),
        ("Ground truth", _rgb_image(target)),
        ("Input 1", _rgb_image(frame1)),
        ("Single-field", _rgb_image(single.clamp(0, 1))),
        ("Multi-field selector", _rgb_image(multi.clamp(0, 1))),
        ("Multi-field oracle", _rgb_image(oracle.clamp(0, 1))),
        ("Single error", _heat_image((single - target).abs().mean(0), maximum=0.3)),
        ("Multi error", _heat_image((multi - target).abs().mean(0), maximum=0.3)),
        ("Extra-field weight", _gray_image(extra_weight)),
        ("Selected field", _gray_image(selected)),
    )
    tile_width, tile_height, header, columns = 400, 240, 26, 2
    rows = 5
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


def _gray_image(tensor) -> Image.Image:
    array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--checkpoint-id", default="parallax-raft-multifield-selector-gopro-001")
    parser.add_argument("--experiment-id", default="parallax-multifield-selector-hard-001")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
