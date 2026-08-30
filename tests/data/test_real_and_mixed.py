from pathlib import Path

from PIL import Image

from genframes.data import (
    DatasetManifest,
    DatasetRole,
    ManifestFrameDataset,
    MixedFrameDataset,
    RiskLevel,
    SampleRecord,
    SourceRecord,
    Split,
)


def test_manifest_frames_and_mixture_share_training_contract(tmp_path: Path) -> None:
    for index in range(3):
        Image.new("RGB", (32, 24), (index * 50, 20, 10)).save(tmp_path / f"{index}.png")
        Image.new("L", (32, 24), 1 if index == 1 else 0).save(tmp_path / f"m{index}.png")
    manifest = _manifest()
    real = ManifestFrameDataset(
        manifest,
        dataset_root=tmp_path,
        split=Split.TRAIN,
        crop_size=(16, 20),
        seed=4,
    )
    sample = real[0]
    assert sample["target"].shape == (3, 16, 20)
    assert sample["object_mask"].shape == (1, 16, 20)
    assert sample["moving_mask"].shape == (1, 16, 20)
    assert not sample["flow_valid"]

    mixed = MixedFrameDataset({"real": real}, weights={"real": 1.0}, length=3, seed=8)
    mixed_sample = mixed[1]
    assert mixed_sample["flow_t0"].shape == (2, 16, 20)
    assert mixed_sample["source_kind"] == "real"
    assert mixed_sample["visibility0"].shape == (1, 16, 20)
    assert not mixed_sample["visibility_valid"]


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        name="real-test",
        created_at="2026-08-24T00:00:00Z",
        sources=(
            SourceRecord(
                source_id="real",
                name="real",
                canonical_url="https://example.test/data",
                retrieved_at="2026-08-24T00:00:00Z",
                archive_sha256="b" * 64,
                dataset_license="test",
                content_license="test",
                roles=(DatasetRole.TRAIN,),
                risk=RiskLevel.GREEN,
                risk_reason="test fixture",
            ),
        ),
        samples=(
            SampleRecord(
                sample_id="one",
                source_id="real",
                sequence_id="seq",
                split=Split.TRAIN,
                frame_paths=("0.png", "1.png", "2.png"),
                frame_times=(0.0, 1.0, 2.0),
                input_indices=(0, 2),
                target_index=1,
                source_fps=1.0,
                width=32,
                height=24,
                attributes={"annotation_paths": ["m0.png", "m1.png", "m2.png"]},
            ),
        ),
    )
