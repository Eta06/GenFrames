"""Parallax: test multi-field candidate quality before learning a selector/fusion."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from analyze_failures import _rgb_image, _select_cases, _tensor
from PIL import Image, ImageDraw

from genframes.data import (
    AnalyticMotionDataset,
    DatasetManifest,
    ManifestFrameDataset,
    Split,
)
from genframes.models import GenFramesRaftGuided
from genframes.ops import backward_warp, inverse_flow_hypotheses
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    destination = layout.benchmarks / arguments.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {destination}")
    destination.mkdir(parents=True)
    device = torch.device(arguments.device)
    model = GenFramesRaftGuided().to(device).eval()
    datasets, hard_indices = _datasets(layout, arguments.samples, arguments.seed)
    group_results = {}
    with torch.inference_mode():
        for group, dataset in datasets.items():
            records = []
            for ordinal in range(len(dataset)):
                sample = dataset[ordinal]
                result, images = _evaluate_sample(model, sample, device)
                records.append(result)
                if group == "davis_hard":
                    _save_panel(
                        destination / f"case-{ordinal + 1:02d}.png",
                        sample,
                        images,
                        hard_indices[ordinal],
                    )
            group_results[group] = {
                "samples": len(records),
                "aggregate": _mean_records(records),
                "records": records if group == "davis_hard" else None,
            }
    profile_sample = datasets["gopro_test"][0]
    result = {
        "experiment_id": arguments.experiment_id,
        "git_keyword": "Parallax",
        "hypothesis": (
            "fixed-point target-flow iterates retain useful competing appearance "
            "candidates where one quadratic field is ambiguous"
        ),
        "representation": {
            "single_fields": 2,
            "multi_fields": 10,
            "refinement_steps_per_endpoint": [0, 1, 2, 4],
            "damping": 0.75,
            "learned_parameters_added": 0,
            "fusion_or_selector": "none; GT oracle is diagnostic only",
        },
        "groups": group_results,
        "profile": _profile(model, profile_sample, device),
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _datasets(layout, count: int, seed: int):
    davis_manifest = DatasetManifest.load(
        layout.datasets / "davis-2017" / "genframes-manifest.json"
    )
    davis_native = ManifestFrameDataset(
        davis_manifest,
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
    )
    hard_indices = _select_cases(davis_native, subset_seed=2405, count=6)
    davis_crop = ManifestFrameDataset(
        davis_manifest,
        dataset_root=layout.datasets / "davis-2017" / "DAVIS",
        split=Split.VALIDATION,
        crop_size=(256, 256),
        seed=seed,
    )
    gopro_root = layout.datasets / "gopro-large-all"
    gopro = ManifestFrameDataset(
        DatasetManifest.load(gopro_root / "genframes-manifest.json"),
        dataset_root=gopro_root,
        split=Split.TEST,
        crop_size=(256, 256),
        seed=seed + 1,
    )
    generator = random.Random(seed)
    davis_indices = sorted(generator.sample(range(len(davis_crop)), min(count, len(davis_crop))))
    gopro_indices = sorted(generator.sample(range(len(gopro)), min(count, len(gopro))))
    return {
        "davis_hard": torch.utils.data.Subset(davis_native, hard_indices),
        "davis_validation": torch.utils.data.Subset(davis_crop, davis_indices),
        "gopro_test": torch.utils.data.Subset(gopro, gopro_indices),
        "analytic": AnalyticMotionDataset(
            length=count, height=256, width=256, seed=seed + 1_000_000
        ),
    }, hard_indices


def _evaluate_sample(model, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target = _tensor(sample, "target").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1, 1, 1, 1).to(device)
    output = model(frame0, frame1, target_time.flatten())
    flow01 = output.auxiliary["flow01"]
    flow10 = output.auxiliary["flow10"]
    single = (output.auxiliary["warped0"], output.auxiliary["warped1"])
    fields0 = inverse_flow_hypotheses(flow01, target_time.flatten())
    fields1 = inverse_flow_hypotheses(flow10, 1.0 - target_time.flatten())
    candidates = single + tuple(backward_warp(frame0, field) for field in fields0) + tuple(
        backward_warp(frame1, field) for field in fields1
    )
    single_oracle, single_error, _ = _oracle(single, target)
    multi_oracle, multi_error, best_index = _oracle(candidates, target)
    disagreement = (single[0] - single[1]).abs().mean(1, keepdim=True) > 0.1
    record = {
        "sequence_id": str(sample["sequence_id"]),
        "single_warp_oracle_psnr_db": _psnr(single_oracle, target),
        "multi_warp_oracle_psnr_db": _psnr(multi_oracle, target),
        "multi_minus_single_psnr_db": _psnr(multi_oracle, target)
        - _psnr(single_oracle, target),
        "single_warp_oracle_mae": float(single_error.mean()),
        "multi_warp_oracle_mae": float(multi_error.mean()),
        "high_disagreement_fraction": float(disagreement.float().mean()),
        "high_disagreement_single_mae": _masked_mean(single_error, disagreement),
        "high_disagreement_multi_mae": _masked_mean(multi_error, disagreement),
        "high_disagreement_improvement_fraction": _improvement_fraction(
            single_error, multi_error, disagreement
        ),
        "global_improvement_fraction": _improvement_fraction(
            single_error, multi_error, torch.ones_like(disagreement)
        ),
        "candidate_appearance_spread": float(
            torch.stack(candidates, dim=1).std(dim=1).mean()
        ),
    }
    if "visibility0" in sample and "visibility1" in sample:
        visibility0 = _tensor(sample, "visibility0").unsqueeze(0).to(device).bool()
        visibility1 = _tensor(sample, "visibility1").unsqueeze(0).to(device).bool()
        occlusion = ~(visibility0 & visibility1)
        record.update(
            {
                "occlusion_fraction": float(occlusion.float().mean()),
                "occlusion_single_mae": _masked_mean(single_error, occlusion),
                "occlusion_multi_mae": _masked_mean(multi_error, occlusion),
                "occlusion_improvement_fraction": _improvement_fraction(
                    single_error, multi_error, occlusion
                ),
            }
        )
    return record, {
        "frame0": frame0[0],
        "target": target[0],
        "frame1": frame1[0],
        "single_oracle": single_oracle[0],
        "multi_oracle": multi_oracle[0],
        "best_index": best_index[0].float() / (len(candidates) - 1),
        "single_blend": (0.5 * single[0] + 0.5 * single[1])[0],
    }


def _oracle(candidates, target):
    stack = torch.stack(candidates, dim=1).clamp(0, 1)
    error = (stack - target[:, None]).abs().mean(2)
    best_error, best_index = error.min(dim=1)
    gather_index = best_index[:, None, None].expand(-1, 1, 3, -1, -1)
    best = stack.gather(1, gather_index).squeeze(1)
    return best, best_error[:, None], best_index


def _masked_mean(values, mask) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def _improvement_fraction(single_error, multi_error, mask) -> float:
    improved = (single_error - multi_error) > (1.0 / 255.0)
    return float(improved[mask].float().mean()) if mask.any() else 0.0


def _psnr(prediction, target) -> float:
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _mean_records(records):
    values = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if key != "sequence_id":
                values[key].append(float(value))
    return {key: statistics.fmean(items) for key, items in values.items()}


def _profile(model, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1, 1, 1, 1).to(device)
    with torch.inference_mode():
        timings = {
            "single_ms": [],
            "multi_ms": [],
            "single_end_to_end_ms": [],
            "multi_end_to_end_ms": [],
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(13):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(frame0, frame1, target_time.flatten())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_ended = time.perf_counter()
            flow01, flow10 = output.auxiliary["flow01"], output.auxiliary["flow10"]
            single = (
                backward_warp(frame0, output.auxiliary["flow_t0"]),
                backward_warp(frame1, output.auxiliary["flow_t1"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            split = time.perf_counter()
            fields0 = inverse_flow_hypotheses(flow01, target_time.flatten())
            fields1 = inverse_flow_hypotheses(flow10, 1.0 - target_time.flatten())
            _ = single + tuple(backward_warp(frame0, field) for field in fields0) + tuple(
                backward_warp(frame1, field) for field in fields1
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ended = time.perf_counter()
            if iteration >= 3:
                timings["single_ms"].append((split - model_ended) * 1000)
                timings["multi_ms"].append((ended - model_ended) * 1000)
                timings["single_end_to_end_ms"].append((split - started) * 1000)
                timings["multi_end_to_end_ms"].append((ended - started) * 1000)
    return {
        "resolution": list(frame0.shape[-2:]),
        "single_representation_median_ms": statistics.median(timings["single_ms"]),
        "multi_representation_median_ms": statistics.median(timings["multi_ms"]),
        "multi_overhead_median_ms": statistics.median(timings["multi_ms"])
        - statistics.median(timings["single_ms"]),
        "single_end_to_end_median_ms": statistics.median(timings["single_end_to_end_ms"]),
        "multi_end_to_end_median_ms": statistics.median(timings["multi_end_to_end_ms"]),
        "peak_allocated_bytes_end_to_end": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "added_parameters": 0,
        "research_oracle_total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "common_raft_cost_included_in_end_to_end_timing": True,
    }


def _save_panel(path: Path, sample, images, source_index: int) -> None:
    panels = (
        ("Input 0", _rgb_image(images["frame0"])),
        ("Ground truth", _rgb_image(images["target"])),
        ("Input 1", _rgb_image(images["frame1"])),
        ("Single blend", _rgb_image(images["single_blend"])),
        ("Single-field oracle", _rgb_image(images["single_oracle"])),
        ("Multi-field oracle", _rgb_image(images["multi_oracle"])),
        ("Chosen field index", _gray_image(images["best_index"])),
    )
    tile_width, tile_height, header, columns = 400, 240, 26, 2
    rows = 4
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 7, row * (tile_height + header) + 6), label, fill="white")
    draw.text((410, 3 * (tile_height + header) + 10), f"dataset_index={source_index}", fill="white")
    canvas.save(path)


def _gray_image(tensor) -> Image.Image:
    array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--experiment-id", default="parallax-fixed-point-multifield-audit-001")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
