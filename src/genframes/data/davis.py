"""DAVIS 2017 ingestion with source-level split and temporal metadata preservation."""

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

DAVIS_2017_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)
DAVIS_REPORTED_FPS = 24.0


def build_davis_2017_manifest(
    dataset_root: str | Path,
    *,
    archive_sha256: str,
    temporal_gaps: tuple[int, ...] = (1, 2, 3),
    retrieved_at: str | None = None,
) -> DatasetManifest:
    """Index midpoint triplets without copying or renaming source frames."""
    root = Path(dataset_root)
    jpeg_root = root / "JPEGImages" / "480p"
    annotation_root = root / "Annotations" / "480p"
    split_root = root / "ImageSets" / "2017"
    if not jpeg_root.is_dir() or not split_root.is_dir():
        raise FileNotFoundError(f"not a DAVIS 2017 root: {root}")
    if any(gap <= 0 for gap in temporal_gaps):
        raise ValueError("temporal gaps must be positive")

    samples: list[SampleRecord] = []
    for split_name, split in (("train", Split.TRAIN), ("val", Split.VALIDATION)):
        sequences = _read_sequence_list(split_root / f"{split_name}.txt")
        for sequence in sequences:
            frames = sorted((jpeg_root / sequence).glob("*.jpg"))
            for gap in temporal_gaps:
                for left in range(0, len(frames) - 2 * gap):
                    indices = (left, left + gap, left + 2 * gap)
                    selected = tuple(frames[index] for index in indices)
                    width, height = _image_size(selected[0])
                    relative_frames = tuple(path.relative_to(root).as_posix() for path in selected)
                    masks = tuple(
                        (annotation_root / sequence / f"{path.stem}.png")
                        .relative_to(root)
                        .as_posix()
                        for path in selected
                    )
                    times = tuple(index / DAVIS_REPORTED_FPS for index in indices)
                    samples.append(
                        SampleRecord(
                            sample_id=f"davis17-{split_name}-{sequence}-{left:05d}-g{gap}",
                            source_id="davis-2017-trainval-480p",
                            sequence_id=f"davis17/{sequence}",
                            split=split,
                            frame_paths=relative_frames,
                            frame_times=times,
                            input_indices=(0, 2),
                            target_index=1,
                            source_fps=DAVIS_REPORTED_FPS,
                            width=width,
                            height=height,
                            attributes={
                                "source_frame_indices": list(indices),
                                "input_temporal_gap_frames": 2 * gap,
                                "target_offset_frames": gap,
                                "target_time": 0.5,
                                "annotation_paths": list(masks),
                                "resolution_variant": "480p",
                                "fps_status": "dataset-level reported value",
                            },
                        )
                    )

    manifest = DatasetManifest(
        name="davis-2017-trainval-480p-midpoint",
        created_at=retrieved_at or datetime.now(UTC).isoformat(),
        sources=(
            SourceRecord(
                source_id="davis-2017-trainval-480p",
                name="DAVIS 2017 TrainVal 480p",
                canonical_url=DAVIS_2017_URL,
                retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
                archive_sha256=archive_sha256,
                dataset_license="Citation/research dataset; archive-wide terms not explicit",
                content_license="Source-video-specific terms not enumerated in archive",
                roles=(
                    DatasetRole.TRAIN,
                    DatasetRole.VALIDATION,
                    DatasetRole.HUMAN_EVALUATION_ONLY,
                ),
                risk=RiskLevel.YELLOW,
                risk_reason=(
                    "Official download and split are available, but a single explicit "
                    "redistribution/trained-weight license was not located"
                ),
                notes="Official sequence split retained; masks indexed for regional metrics.",
            ),
        ),
        samples=tuple(samples),
    )
    manifest.validate()
    return manifest


def _read_sequence_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
