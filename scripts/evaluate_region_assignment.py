"""Audit hard region ownership against single-field and dense soft selection."""

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
import torch.nn.functional as functional
from analyze_failures import _heat_image, _rgb_image, _select_cases, _tensor
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from genframes.data import AnalyticMotionDataset, DatasetManifest, ManifestFrameDataset, Split
from genframes.models import (
    GenFramesRaftEvidenceSelector,
    GenFramesRaftGuided,
    GenFramesRaftRegionAssignment,
)
from genframes.storage import StorageLayout


def main() -> None:
    args = parse_arguments()
    layout = StorageLayout.resolve(args.storage_root)
    destination = layout.benchmarks / args.experiment_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {destination}")
    destination.mkdir(parents=True)
    device = torch.device(args.device)
    single = GenFramesRaftGuided().to(device).eval()
    dense = GenFramesRaftEvidenceSelector()
    dense.load_selector_state_dict(
        load_file(layout.checkpoints / args.dense_checkpoint_id / "selector.safetensors")
    )
    dense = dense.to(device).eval()
    region = GenFramesRaftRegionAssignment()
    checkpoint = layout.checkpoints / args.checkpoint_id / "selector.safetensors"
    region.load_selector_state_dict(load_file(checkpoint))
    region = region.to(device).eval()
    datasets, hard_indices = _datasets(layout, args.samples, args.seed)
    groups = {}
    with torch.inference_mode():
        for group_name, dataset in datasets.items():
            records = []
            for ordinal in range(len(dataset)):
                record, images = _evaluate_sample(single, dense, region, dataset[ordinal], device)
                records.append(record)
                if group_name == "davis_hard":
                    _save_panel(destination / f"case-{ordinal + 1:02d}.png", images)
            groups[group_name] = {
                "samples": len(records),
                "aggregate": _aggregate(records),
                "records": records if group_name == "davis_hard" else None,
            }
    result = {
        "experiment_id": args.experiment_id,
        "git_keyword": "Parallax",
        "checkpoint": str(checkpoint),
        "dense_checkpoint_id": args.dense_checkpoint_id,
        "representation": "immutable 10-field fixed-point candidate stack",
        "groups": groups,
        "hard_case_indices": hard_indices,
        "runtime": _profile(single, dense, region, datasets["gopro_test"][0], device),
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _datasets(layout, count: int, seed: int):
    davis_root = layout.datasets / "davis-2017"
    manifest = DatasetManifest.load(davis_root / "genframes-manifest.json")
    native = ManifestFrameDataset(
        manifest, dataset_root=davis_root / "DAVIS", split=Split.VALIDATION
    )
    hard_indices = _select_cases(native, subset_seed=2405, count=6)
    crop = ManifestFrameDataset(
        manifest,
        dataset_root=davis_root / "DAVIS",
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
    crop_indices = sorted(generator.sample(range(len(crop)), min(count, len(crop))))
    gopro_indices = sorted(generator.sample(range(len(gopro)), min(count, len(gopro))))
    return {
        "davis_hard": torch.utils.data.Subset(native, hard_indices),
        "davis_validation": torch.utils.data.Subset(crop, crop_indices),
        "gopro_test": torch.utils.data.Subset(gopro, gopro_indices),
        "analytic": AnalyticMotionDataset(
            length=count, height=256, width=256, seed=seed + 1_000_000
        ),
    }, hard_indices


def _evaluate_sample(single, dense, region, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target = _tensor(sample, "target").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1).to(device)
    single_frame = single(frame0, frame1, target_time).frame.clamp(0, 1)
    dense_output = dense(frame0, frame1, target_time)
    dense_frame = dense_output.frame.clamp(0, 1)
    output = region(frame0, frame1, target_time)
    region_frame = output.frame.clamp(0, 1)
    candidates = output.auxiliary["candidate_stack"].clamp(0, 1)
    errors = (candidates - target[:, None]).abs().mean(2)
    oracle_error, oracle_index = errors.min(1)
    oracle = _gather_candidate(candidates, oracle_index)
    logits = output.auxiliary["assignment_logits"]
    probabilities = output.auxiliary["assignment_probabilities"]
    assignment = output.auxiliary["assignment_index"]
    fallback_index = output.auxiliary["fallback_index"]
    abstain = output.auxiliary["abstain_mask"][:, None]
    forced_index = logits[:, : candidates.shape[1]].argmax(1)
    forced = _gather_candidate(candidates, forced_index)
    actual_index = torch.where(abstain[:, 0], fallback_index, assignment)
    actual_error = errors.gather(1, actual_index[:, None]).squeeze(1)
    soft_entropy = _entropy(probabilities)
    dense_weights = dense_output.auxiliary["candidate_weights"]
    dense_index = dense_weights.argmax(1)
    spread = candidates.std(1).mean(1, keepdim=True)
    masks = {
        "global": torch.ones_like(spread, dtype=torch.bool),
        "high_disagreement": spread > 0.05,
    }
    if "object_mask" in sample:
        masks["object"] = _tensor(sample, "object_mask").unsqueeze(0).to(device).bool()
    if "visibility0" in sample and "visibility1" in sample:
        visibility0 = _tensor(sample, "visibility0").unsqueeze(0).to(device).bool()
        visibility1 = _tensor(sample, "visibility1").unsqueeze(0).to(device).bool()
        masks["occlusion"] = ~(visibility0 & visibility1)
        masks["disocclusion"] = visibility0 ^ visibility1
    record = {
        "sequence_id": str(sample["sequence_id"]),
        "single_psnr_db": _psnr(single_frame, target),
        "dense_psnr_db": _psnr(dense_frame, target),
        "region_psnr_db": _psnr(region_frame, target),
        "oracle_psnr_db": _psnr(oracle, target),
        "single_mae": _mae(single_frame, target),
        "dense_mae": _mae(dense_frame, target),
        "region_mae": _mae(region_frame, target),
        "oracle_mae": float(oracle_error.mean()),
        "hard_output_entropy": 0.0,
        "soft_assignment_entropy": float(soft_entropy.mean()),
        "abstain_fraction": float(abstain.float().mean()),
        "region_consistency": _low_edge_consistency(actual_index, frame0, frame1),
        "dense_argmax_consistency": _low_edge_consistency(dense_index, frame0, frame1),
        "selector_oracle_tolerance_accuracy": _masked_tolerance_accuracy(
            actual_error, oracle_error, masks["global"]
        ),
    }
    record["mae_oracle_gap_closed"] = _gain_fraction(
        record["single_mae"], record["region_mae"], record["oracle_mae"]
    )
    record["psnr_oracle_gap_closed"] = _gain_fraction(
        record["single_psnr_db"], record["region_psnr_db"], record["oracle_psnr_db"]
    )
    if abstain.any():
        record["abstain_region_mae"] = _masked_mae(region_frame, target, abstain)
        record["abstain_forced_mae"] = _masked_mae(forced, target, abstain)
        record["abstain_fallback_gain"] = (
            record["abstain_forced_mae"] - record["abstain_region_mae"]
        )
    else:
        record.update(
            {"abstain_region_mae": 0.0, "abstain_forced_mae": 0.0, "abstain_fallback_gain": 0.0}
        )
    for name, mask in masks.items():
        if name == "global":
            continue
        _add_region_metrics(
            record,
            name,
            mask,
            single_frame,
            dense_frame,
            region_frame,
            oracle,
            target,
            actual_error,
            oracle_error,
            abstain,
            soft_entropy,
        )
    if "object" in masks:
        record["object_dominant_assignment_fraction"] = _dominant_fraction(
            actual_index, masks["object"]
        )
    if "disocclusion" in masks:
        source_target = visibility1[:, 0].to(torch.long)
        record["disocclusion_ownership_accuracy"] = _masked_accuracy(
            output.auxiliary["chosen_source"], source_target, masks["disocclusion"]
        )
    return record, {
        "frame0": frame0[0],
        "target": target[0],
        "frame1": frame1[0],
        "single": single_frame[0],
        "dense": dense_frame[0],
        "region": region_frame[0],
        "forced": forced[0],
        "oracle": oracle[0],
        "dense_error": (dense_frame - target).abs().mean(1)[0],
        "region_error": (region_frame - target).abs().mean(1)[0],
        "assignment": actual_index[0].float() / (candidates.shape[1] - 1),
        "abstain": abstain[0, 0].float(),
        "entropy": soft_entropy[0] / np.log(candidates.shape[1] + 1),
        "disagreement": spread[0, 0],
    }


def _add_region_metrics(record, name, mask, single, dense, region, oracle, target,
                        actual_error, oracle_error, abstain, entropy):
    record[f"{name}_fraction"] = float(mask.float().mean())
    for model_name, prediction in (
        ("single", single), ("dense", dense), ("region", region), ("oracle", oracle)
    ):
        record[f"{name}_{model_name}_mae"] = _masked_mae(prediction, target, mask)
        if model_name != "oracle":
            record[f"{name}_{model_name}_ghosting"] = _masked_edge_error(
                prediction, target, mask
            )
    record[f"{name}_abstain_fraction"] = _masked_mean(abstain.float(), mask)
    record[f"{name}_soft_entropy"] = _masked_mean(entropy[:, None], mask)
    record[f"{name}_oracle_tolerance_accuracy"] = _masked_tolerance_accuracy(
        actual_error, oracle_error, mask
    )


def _gather_candidate(candidates, index):
    return candidates.gather(
        1, index[:, None, None].expand(-1, 1, 3, -1, -1)
    ).squeeze(1)


def _low_edge_consistency(index, frame0, frame1):
    guidance = 0.5 * (frame0 + frame1)
    dx = (guidance[:, :, :, 1:] - guidance[:, :, :, :-1]).abs().mean(1) < 0.05
    dy = (guidance[:, :, 1:, :] - guidance[:, :, :-1, :]).abs().mean(1) < 0.05
    same_x = index[:, :, 1:] == index[:, :, :-1]
    same_y = index[:, 1:, :] == index[:, :-1, :]
    numerator = same_x[dx].float().sum() + same_y[dy].float().sum()
    denominator = dx.sum() + dy.sum()
    return float(numerator / denominator.clamp_min(1))


def _dominant_fraction(index, mask):
    values = index[mask[:, 0]]
    if values.numel() == 0:
        return 0.0
    return float(torch.bincount(values.flatten()).max() / values.numel())


def _masked_tolerance_accuracy(selected_error, oracle_error, mask):
    selected = selected_error[mask[:, 0]]
    oracle = oracle_error[mask[:, 0]]
    return float((selected <= oracle + (1 / 255)).float().mean()) if selected.numel() else 0.0


def _masked_accuracy(predicted, target, mask):
    selected = mask[:, 0]
    return float((predicted == target)[selected].float().mean()) if selected.any() else 0.0


def _masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else 0.0


def _masked_mae(prediction, target, mask):
    return _masked_mean((prediction - target).abs().mean(1, keepdim=True), mask)


def _masked_edge_error(prediction, target, mask):
    kernel = prediction.new_tensor(((0, -1, 0), (-1, 4, -1), (0, -1, 0))).view(1, 1, 3, 3)
    kernel = kernel.expand(3, 1, 3, 3)
    error = (functional.conv2d(prediction, kernel, padding=1, groups=3)
             - functional.conv2d(target, kernel, padding=1, groups=3)).abs().mean(1, keepdim=True)
    return _masked_mean(error, mask)


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)


def _psnr(prediction, target):
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(10 * torch.log10(1 / mse))


def _mae(prediction, target):
    return float((prediction - target).abs().mean())


def _gain_fraction(start, selected, oracle):
    denominator = start - oracle
    return (start - selected) / denominator if abs(denominator) > 1e-9 else 0.0


def _aggregate(records):
    values = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if key != "sequence_id":
                values[key].append(float(value))
    aggregate = {key: statistics.fmean(items) for key, items in values.items()}
    aggregate["cases_region_psnr_improved"] = sum(
        record["region_psnr_db"] > record["single_psnr_db"] for record in records
    )
    aggregate["aggregate_mae_oracle_gap_closed"] = _gain_fraction(
        aggregate["single_mae"], aggregate["region_mae"], aggregate["oracle_mae"]
    )
    aggregate["aggregate_psnr_oracle_gap_closed"] = _gain_fraction(
        aggregate["single_psnr_db"], aggregate["region_psnr_db"], aggregate["oracle_psnr_db"]
    )
    return aggregate


def _profile(single, dense, region, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1).to(device)
    result = {}
    for name, model in (("single", single), ("dense", dense), ("region", region)):
        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for iteration in range(13):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(frame0, frame1, target_time)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if iteration >= 3:
                    timings.append((time.perf_counter() - started) * 1000)
        result[f"{name}_median_ms"] = statistics.median(timings)
        result[f"{name}_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
    result["region_trainable_parameters"] = sum(
        parameter.numel() for parameter in region.parameters() if parameter.requires_grad
    )
    return result


def _save_panel(path: Path, images):
    panels = (
        ("Input 0", _rgb_image(images["frame0"])),
        ("Ground truth", _rgb_image(images["target"])),
        ("Input 1", _rgb_image(images["frame1"])),
        ("Single field", _rgb_image(images["single"])),
        ("Dense evidence", _rgb_image(images["dense"])),
        ("Hard region", _rgb_image(images["region"])),
        ("Forced no-abstain", _rgb_image(images["forced"])),
        ("Oracle", _rgb_image(images["oracle"])),
        ("Dense error", _heat_image(images["dense_error"], maximum=0.3)),
        ("Region error", _heat_image(images["region_error"], maximum=0.3)),
        ("Assignment", _gray_image(images["assignment"])),
        ("Abstain", _gray_image(images["abstain"])),
        ("Soft entropy", _heat_image(images["entropy"], maximum=1.0)),
        ("Candidate disagreement", _heat_image(images["disagreement"], maximum=0.2)),
    )
    tile_width, tile_height, header, columns = 360, 216, 26, 3
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 6, row * (tile_height + header) + 6), label, fill="white")
    canvas.save(path)


def _gray_image(tensor):
    array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--checkpoint-id", default="parallax-raft-region-assignment-gopro-001")
    parser.add_argument(
        "--dense-checkpoint-id", default="parallax-raft-evidence-selector-gopro-001"
    )
    parser.add_argument("--experiment-id", default="parallax-region-assignment-audit-001")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
