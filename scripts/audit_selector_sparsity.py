"""Audit whether soft candidate averaging is the remaining Parallax bottleneck."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from analyze_failures import _rgb_image, _select_cases, _tensor
from evaluate_multifield_selector import _masked_edge_error, _masked_mae, _psnr
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
    multi.load_selector_state_dict(
        load_file(layout.checkpoints / arguments.checkpoint_id / "selector.safetensors")
    )
    multi = multi.to(device).eval()
    modes = ("soft-1", "soft-0.5", "soft-0.25", "hard")
    records = []
    with torch.inference_mode():
        for ordinal, index in enumerate(selected, start=1):
            sample = dataset[index]
            frame0 = _tensor(sample, "frame0").unsqueeze(0).to(device)
            frame1 = _tensor(sample, "frame1").unsqueeze(0).to(device)
            target = _tensor(sample, "target").unsqueeze(0).to(device)
            target_time = _tensor(sample, "time").reshape(1).to(device)
            single_output = single(frame0, frame1, target_time)
            output = multi(frame0, frame1, target_time)
            candidates = output.auxiliary["candidate_stack"]
            logits = output.auxiliary["candidate_logits"]
            disagreement = (
                single_output.auxiliary["warped0"]
                - single_output.auxiliary["warped1"]
            ).abs().mean(1, keepdim=True) > 0.1
            predictions = {
                "soft-1": _select(candidates, logits, 1.0),
                "soft-0.5": _select(candidates, logits, 0.5),
                "soft-0.25": _select(candidates, logits, 0.25),
                "hard": _hard_select(candidates, logits),
            }
            metrics = {
                name: {
                    "psnr_db": _psnr(prediction, target),
                    "high_disagreement_mae": _masked_mae(prediction, target, disagreement),
                    "edge_ghosting_proxy": _masked_edge_error(
                        prediction, target, disagreement
                    ),
                }
                for name, prediction in predictions.items()
            }
            records.append(
                {
                    "case_id": f"case-{ordinal:02d}",
                    "sequence_id": str(sample["sequence_id"]),
                    "modes": metrics,
                }
            )
            _save_panel(
                destination / f"case-{ordinal:02d}.png",
                frame0[0],
                target[0],
                frame1[0],
                predictions,
            )
    aggregate = {
        name: {
            key: statistics.fmean(record["modes"][name][key] for record in records)
            for key in records[0]["modes"][name]
        }
        for name in modes
    }
    result = {
        "experiment_id": arguments.experiment_id,
        "checkpoint_id": arguments.checkpoint_id,
        "records": records,
        "aggregate": aggregate,
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _select(candidates, logits, temperature):
    weights = (logits / temperature).softmax(1)
    return (weights[:, :, None] * candidates).sum(1)


def _hard_select(candidates, logits):
    index = logits.argmax(1)
    gather = index[:, None, None].expand(-1, 1, 3, -1, -1)
    return candidates.gather(1, gather).squeeze(1)


def _save_panel(path, frame0, target, frame1, predictions):
    panels = [
        ("Input 0", _rgb_image(frame0)),
        ("Ground truth", _rgb_image(target)),
        ("Input 1", _rgb_image(frame1)),
    ]
    panels.extend(
        (name, _rgb_image(prediction[0].clamp(0, 1)))
        for name, prediction in predictions.items()
    )
    tile_width, tile_height, header, columns = 400, 240, 26, 2
    canvas = Image.new("RGB", (columns * tile_width, 4 * (tile_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        column, row = index % columns, index // columns
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * (tile_height + header) + header + (tile_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((column * tile_width + 7, row * (tile_height + header) + 6), label, fill="white")
    canvas.save(path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--checkpoint-id", default="parallax-raft-multifield-selector-gopro-001")
    parser.add_argument("--experiment-id", default="parallax-selector-sparsity-audit-001")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
