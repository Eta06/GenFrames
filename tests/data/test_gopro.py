from pathlib import Path

from PIL import Image

from genframes.data.gopro import build_gopro_large_all_manifest
from genframes.data.manifest import RiskLevel, Split


def test_gopro_manifest_preserves_official_split_fps_and_gaps(tmp_path: Path) -> None:
    for source_split in ("train", "test"):
        sequence = tmp_path / source_split / f"{source_split}-sequence"
        sequence.mkdir(parents=True)
        for index in range(1, 6):
            Image.new("RGB", (48, 32), (index, 0, 0)).save(sequence / f"{index:06d}.png")
    manifest = build_gopro_large_all_manifest(
        tmp_path,
        archive_sha256="b" * 64,
        temporal_gaps=(1, 2),
        retrieved_at="2026-08-30T00:00:00Z",
    )
    assert len(manifest.samples) == 8
    assert {sample.split for sample in manifest.samples} == {Split.TRAIN, Split.TEST}
    assert {sample.source_fps for sample in manifest.samples} == {240.0}
    assert {sample.attributes["input_temporal_gap_frames"] for sample in manifest.samples} == {
        2,
        4,
    }
    assert manifest.sources[0].risk is RiskLevel.GREEN
    assert manifest.samples[0].width == 48
    assert manifest.samples[0].height == 32


def test_gopro_manifest_skips_non_contiguous_triplets(tmp_path: Path) -> None:
    for source_split in ("train", "test"):
        sequence = tmp_path / source_split / "sequence"
        sequence.mkdir(parents=True)
        for index in (1, 2, 4, 5, 6):
            Image.new("RGB", (16, 16)).save(sequence / f"{index:06d}.png")
    manifest = build_gopro_large_all_manifest(
        tmp_path,
        archive_sha256="c" * 64,
        temporal_gaps=(1,),
    )
    assert {tuple(sample.attributes["source_frame_indices"]) for sample in manifest.samples} == {
        (4, 5, 6)
    }
