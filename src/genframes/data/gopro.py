"""GoPro_Large_all high-frame-rate sharp-sequence manifest construction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .manifest import (
    DatasetManifest,
    DatasetRole,
    RiskLevel,
    SampleRecord,
    SourceRecord,
    Split,
)

GOPRO_LARGE_ALL_URL = (
    "https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/"
    "GOPRO_Large_all.zip"
)
GOPRO_REPORTED_FPS = 240.0


def build_gopro_large_all_manifest(
    dataset_root: str | Path,
    *,
    archive_sha256: str,
    temporal_gaps: tuple[int, ...] = (1, 2, 4, 8),
    retrieved_at: str | None = None,
) -> DatasetManifest:
    """Index midpoint triplets while retaining official train/test sequences."""
    root = Path(dataset_root)
    if not (root / "train").is_dir() or not (root / "test").is_dir():
        raise FileNotFoundError(f"not a GOPRO_Large_all root: {root}")
    if any(gap <= 0 for gap in temporal_gaps):
        raise ValueError("temporal gaps must be positive")
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    samples = []
    for source_split, split in (("train", Split.TRAIN), ("test", Split.TEST)):
        sequence_roots = sorted(
            path for path in (root / source_split).iterdir() if path.is_dir()
        )
        for sequence_root in sequence_roots:
            frames = sorted(sequence_root.glob("*.png"), key=lambda path: int(path.stem))
            if len(frames) < 3:
                continue
            width, height = _image_size(frames[0])
            frame_numbers = [int(path.stem) for path in frames]
            for gap in temporal_gaps:
                for left in range(len(frames) - 2 * gap):
                    positions = (left, left + gap, left + 2 * gap)
                    numbers = tuple(frame_numbers[position] for position in positions)
                    if numbers != (numbers[0], numbers[0] + gap, numbers[0] + 2 * gap):
                        continue
                    selected = tuple(frames[position] for position in positions)
                    samples.append(
                        SampleRecord(
                            sample_id=(
                                f"gopro-{source_split}-{sequence_root.name}-"
                                f"{numbers[0]:06d}-g{gap}"
                            ),
                            source_id="gopro-large-all-sharp",
                            sequence_id=f"gopro/{source_split}/{sequence_root.name}",
                            split=split,
                            frame_paths=tuple(
                                path.relative_to(root).as_posix() for path in selected
                            ),
                            frame_times=tuple(
                                number / GOPRO_REPORTED_FPS for number in numbers
                            ),
                            input_indices=(0, 2),
                            target_index=1,
                            source_fps=GOPRO_REPORTED_FPS,
                            width=width,
                            height=height,
                            attributes={
                                "source_split": source_split,
                                "source_frame_indices": list(numbers),
                                "input_temporal_gap_frames": 2 * gap,
                                "target_offset_frames": gap,
                                "target_time": 0.5,
                                "capture_fps_status": "dataset-owner reported 240 FPS",
                                "license": "CC BY 4.0",
                            },
                        )
                    )
    manifest = DatasetManifest(
        name="gopro-large-all-sharp-midpoint",
        created_at=timestamp,
        sources=(
            SourceRecord(
                source_id="gopro-large-all-sharp",
                name="GOPRO_Large_all sharp frames",
                canonical_url=GOPRO_LARGE_ALL_URL,
                retrieved_at=timestamp,
                archive_sha256=archive_sha256,
                dataset_license="Creative Commons Attribution 4.0 International",
                content_license="Creative Commons Attribution 4.0 International",
                roles=(DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.BENCHMARK),
                risk=RiskLevel.GREEN,
                risk_reason=(
                    "Dataset owner explicitly releases GOPRO dataset under CC BY 4.0"
                ),
                notes=(
                    "Official train/test directories retained; only sharp 240 FPS frames "
                    "are indexed and no derived frames are copied."
                ),
            ),
        ),
        samples=tuple(samples),
    )
    manifest.validate()
    return manifest


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
