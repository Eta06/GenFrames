from pathlib import Path

from PIL import Image

from genframes.data.davis import build_davis_2017_manifest
from genframes.data.manifest import Split


def test_davis_manifest_preserves_sequence_split_and_gap(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "train.txt").write_text("train-seq\n")
    (root / "ImageSets" / "2017" / "val.txt").write_text("val-seq\n")
    for sequence in ("train-seq", "val-seq"):
        frames = root / "JPEGImages" / "480p" / sequence
        masks = root / "Annotations" / "480p" / sequence
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (40, 24), (index, 0, 0)).save(frames / f"{index:05d}.jpg")
            Image.new("L", (40, 24), index).save(masks / f"{index:05d}.png")

    manifest = build_davis_2017_manifest(
        root,
        archive_sha256="a" * 64,
        temporal_gaps=(1, 2),
        retrieved_at="2026-08-24T00:00:00Z",
    )
    assert len(manifest.samples) == 8
    train_samples = [sample for sample in manifest.samples if sample.split is Split.TRAIN]
    assert {sample.sequence_id for sample in train_samples} == {"davis17/train-seq"}
    assert {sample.attributes["input_temporal_gap_frames"] for sample in train_samples} == {2, 4}
    assert train_samples[0].source_fps == 24.0
    assert train_samples[0].width == 40
    assert train_samples[0].height == 24
