"""Create an auditable GenFrames manifest for an extracted GOPRO_Large_all archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from genframes.data import build_gopro_large_all_manifest
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    dataset_root = (
        Path(arguments.dataset_root)
        if arguments.dataset_root
        else layout.datasets / "gopro-large-all"
    )
    manifest = build_gopro_large_all_manifest(
        dataset_root,
        archive_sha256=arguments.archive_sha256,
        temporal_gaps=tuple(arguments.temporal_gaps),
        retrieved_at=arguments.retrieved_at,
    )
    destination = dataset_root / "genframes-manifest.json"
    manifest.save(destination)
    split_counts = {
        split: sum(sample.split.value == split for sample in manifest.samples)
        for split in ("train", "validation", "test")
    }
    print(f"dataset_root={dataset_root}")
    print(f"manifest={destination}")
    print(f"samples={len(manifest.samples)}")
    print(f"split_counts={split_counts}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--retrieved-at")
    parser.add_argument("--temporal-gaps", nargs="+", type=int, default=(1, 2, 4, 8))
    return parser.parse_args()


if __name__ == "__main__":
    main()
