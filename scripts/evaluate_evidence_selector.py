"""Evaluate the Parallax evidence selector without changing its representation."""

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
from genframes.models import GenFramesRaftEvidenceSelector, GenFramesRaftGuided
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
    selector = GenFramesRaftEvidenceSelector()
    checkpoint = layout.checkpoints / args.checkpoint_id / "selector.safetensors"
    selector.load_selector_state_dict(load_file(checkpoint))
    selector = selector.to(device).eval()
    datasets, hard_indices = _datasets(layout, args.samples, args.seed)
    groups = {}
    with torch.inference_mode():
        for group_name, dataset in datasets.items():
            records = []
            for ordinal in range(len(dataset)):
                record, images = _evaluate_sample(single, selector, dataset[ordinal], device)
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
        "representation": "immutable 10-field fixed-point candidate stack",
        "groups": groups,
        "hard_case_indices": hard_indices,
        "runtime": _profile(single, selector, datasets["gopro_test"][0], device),
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _datasets(layout, count: int, seed: int):
    davis_root = layout.datasets / "davis-2017"
    davis_manifest = DatasetManifest.load(davis_root / "genframes-manifest.json")
    davis_native = ManifestFrameDataset(
        davis_manifest, dataset_root=davis_root / "DAVIS", split=Split.VALIDATION
    )
    hard_indices = _select_cases(davis_native, subset_seed=2405, count=6)
    davis_crop = ManifestFrameDataset(
        davis_manifest,
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


def _evaluate_sample(single, selector, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target = _tensor(sample, "target").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1).to(device)
    single_frame = single(frame0, frame1, target_time).frame.clamp(0, 1)
    output = selector(frame0, frame1, target_time)
    selected = output.frame.clamp(0, 1)
    candidates = output.auxiliary["candidate_stack"].clamp(0, 1)
    errors = (candidates - target[:, None]).abs().mean(2)
    oracle_error, oracle_index = errors.min(1)
    oracle = candidates.gather(
        1, oracle_index[:, None, None].expand(-1, 1, 3, -1, -1)
    ).squeeze(1)
    weights = output.auxiliary["candidate_weights"]
    predicted_index = weights.argmax(1)
    confidence_index = output.auxiliary["confidence_prior"].argmax(1)
    candidate_spread = candidates.std(1).mean(1, keepdim=True)
    high_disagreement = candidate_spread > 0.05
    masks = {"global": torch.ones_like(high_disagreement), "high_disagreement": high_disagreement}
    if "visibility0" in sample and "visibility1" in sample:
        visibility0 = _tensor(sample, "visibility0").unsqueeze(0).to(device).bool()
        visibility1 = _tensor(sample, "visibility1").unsqueeze(0).to(device).bool()
        masks["occlusion"] = ~(visibility0 & visibility1)
    record = {
        "sequence_id": str(sample["sequence_id"]),
        "single_psnr_db": _psnr(single_frame, target),
        "selected_psnr_db": _psnr(selected, target),
        "oracle_psnr_db": _psnr(oracle, target),
        "single_mae": _mae(single_frame, target),
        "selected_mae": _mae(selected, target),
        "oracle_mae": float(oracle_error.mean()),
        "selector_accuracy": _masked_accuracy(predicted_index, oracle_index, masks["global"]),
        "confidence_accuracy": _masked_accuracy(confidence_index, oracle_index, masks["global"]),
        "selector_entropy": float(_entropy(weights).mean()),
        "high_disagreement_fraction": float(high_disagreement.float().mean()),
        "extra_field_weight_mean": float(weights[:, 2:].sum(1).mean()),
    }
    _add_region_metrics(record, "high_disagreement", masks["high_disagreement"], single_frame,
                        selected, oracle, target, predicted_index, confidence_index, oracle_index)
    if "occlusion" in masks:
        record["occlusion_fraction"] = float(masks["occlusion"].float().mean())
        _add_region_metrics(record, "occlusion", masks["occlusion"], single_frame, selected,
                            oracle, target, predicted_index, confidence_index, oracle_index)
    record["mae_oracle_gap_closed"] = _gap_closed(
        record["single_mae"], record["selected_mae"], record["oracle_mae"]
    )
    record["psnr_oracle_gap_closed"] = _gap_closed(
        record["oracle_psnr_db"], record["selected_psnr_db"], record["single_psnr_db"]
    )
    evidence = output.auxiliary["candidate_evidence"]
    chosen = predicted_index[:, None, None].expand(-1, 1, evidence.shape[2], -1, -1)
    chosen_evidence = evidence.gather(1, chosen).squeeze(1)
    confidence = output.auxiliary["confidence_prior"]
    confidence_margin = confidence.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
    return record, {
        "frame0": frame0[0], "target": target[0], "frame1": frame1[0],
        "single": single_frame[0], "selected": selected[0], "oracle": oracle[0],
        "single_error": (single_frame - target).abs().mean(1)[0],
        "selected_error": (selected - target).abs().mean(1)[0],
        "selected_index": predicted_index[0].float() / (candidates.shape[1] - 1),
        "oracle_index": oracle_index[0].float() / (candidates.shape[1] - 1),
        "confidence_margin": confidence_margin[0], "chosen_evidence": chosen_evidence[0],
    }


def _add_region_metrics(record, name, mask, single, selected, oracle, target,
                        predicted_index, confidence_index, oracle_index):
    record[f"{name}_single_mae"] = _masked_mae(single, target, mask)
    record[f"{name}_selected_mae"] = _masked_mae(selected, target, mask)
    record[f"{name}_oracle_mae"] = _masked_mae(oracle, target, mask)
    record[f"{name}_selector_accuracy"] = _masked_accuracy(predicted_index, oracle_index, mask)
    record[f"{name}_confidence_accuracy"] = _masked_accuracy(confidence_index, oracle_index, mask)
    record[f"{name}_single_ghosting"] = _masked_edge_error(single, target, mask)
    record[f"{name}_selected_ghosting"] = _masked_edge_error(selected, target, mask)


def _gap_closed(start, selected, oracle):
    denominator = start - oracle
    return (start - selected) / denominator if abs(denominator) > 1e-9 else 0.0


def _psnr(prediction, target):
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(10 * torch.log10(1 / mse))


def _mae(prediction, target):
    return float((prediction - target).abs().mean())


def _masked_mae(prediction, target, mask):
    error = (prediction - target).abs().mean(1, keepdim=True)
    return float(error[mask].mean()) if mask.any() else 0.0


def _masked_accuracy(predicted, target, mask):
    flat_mask = mask[:, 0]
    return float((predicted == target)[flat_mask].float().mean()) if flat_mask.any() else 0.0


def _masked_edge_error(prediction, target, mask):
    kernel = prediction.new_tensor(((0, -1, 0), (-1, 4, -1), (0, -1, 0))).view(1, 1, 3, 3)
    kernel = kernel.expand(3, 1, 3, 3)
    error = (functional.conv2d(prediction, kernel, padding=1, groups=3)
             - functional.conv2d(target, kernel, padding=1, groups=3)).abs().mean(1, keepdim=True)
    return float(error[mask].mean()) if mask.any() else 0.0


def _entropy(weights):
    return -(weights * weights.clamp_min(1e-8).log()).sum(1)


def _aggregate(records):
    values = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if key != "sequence_id":
                values[key].append(float(value))
    aggregate = {key: statistics.fmean(items) for key, items in values.items()}
    aggregate["cases_psnr_improved"] = sum(
        r["selected_psnr_db"] > r["single_psnr_db"] for r in records
    )
    # Aggregate errors before taking the ratio is less sensitive to tiny per-case oracle gaps.
    aggregate["aggregate_mae_oracle_gap_closed"] = _gap_closed(
        aggregate["single_mae"], aggregate["selected_mae"], aggregate["oracle_mae"]
    )
    aggregate["aggregate_psnr_oracle_gap_closed"] = _gap_closed(
        aggregate["oracle_psnr_db"], aggregate["selected_psnr_db"], aggregate["single_psnr_db"]
    )
    return aggregate


def _profile(single, selector, sample, device):
    frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
    frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
    target_time = _tensor(sample, "time").reshape(1).to(device)
    result = {}
    for name, model in (("single", single), ("selector", selector)):
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
    result["selector_trainable_parameters"] = sum(
        parameter.numel() for parameter in selector.parameters() if parameter.requires_grad
    )
    result["selector_total_research_parameters"] = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    return result


def _save_panel(path: Path, images):
    evidence = images["chosen_evidence"]
    panels = [
        ("Input 0", _rgb_image(images["frame0"])), ("Ground truth", _rgb_image(images["target"])),
        ("Input 1", _rgb_image(images["frame1"])), ("Single field", _rgb_image(images["single"])),
        ("Evidence selector", _rgb_image(images["selected"])),
        ("Oracle", _rgb_image(images["oracle"])),
        ("Single error", _heat_image(images["single_error"], maximum=0.3)),
        ("Selector error", _heat_image(images["selected_error"], maximum=0.3)),
        ("Selected field", _gray_image(images["selected_index"])),
        ("Oracle field", _gray_image(images["oracle_index"])),
        ("Confidence margin", _heat_image(images["confidence_margin"], maximum=1.0)),
    ]
    names = GenFramesRaftEvidenceSelector.evidence_names
    panels.extend((f"Chosen {name}", _heat_image(evidence[index], maximum=1.0))
                  for index, name in enumerate(names))
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
    parser.add_argument("--checkpoint-id", default="parallax-raft-evidence-selector-gopro-001")
    parser.add_argument("--experiment-id", default="parallax-evidence-selector-audit-001")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
