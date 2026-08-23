"""Download, safely extract, and index the official DAVIS 2017 TrainVal archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from genframes.data.davis import DAVIS_2017_URL, build_davis_2017_manifest
from genframes.storage import StorageLayout


def main() -> None:
    arguments = parse_arguments()
    layout = StorageLayout.resolve(arguments.storage_root)
    layout.create()
    archive = layout.archives / "DAVIS-2017-trainval-480p.zip"
    if not archive.is_file():
        _download(arguments.url, archive)
    digest = _sha256(archive)
    extraction_root = layout.datasets / "davis-2017"
    dataset_root = extraction_root / "DAVIS"
    if not dataset_root.is_dir():
        _safe_extract(archive, extraction_root)
    manifest = build_davis_2017_manifest(
        dataset_root,
        archive_sha256=digest,
        temporal_gaps=tuple(arguments.temporal_gaps),
    )
    manifest_path = extraction_root / "genframes-manifest.json"
    manifest.save(manifest_path)
    print(f"archive={archive}")
    print(f"archive_sha256={digest}")
    print(f"dataset_root={dataset_root}")
    print(f"manifest={manifest_path}")
    print(f"samples={len(manifest.samples)}")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root")
    parser.add_argument("--url", default=DAVIS_2017_URL)
    parser.add_argument("--temporal-gaps", nargs="+", type=int, default=(1, 2, 3))
    return parser.parse_args()


if __name__ == "__main__":
    main()
