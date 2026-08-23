"""External storage layout for large, non-repository research artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

STORAGE_ENVIRONMENT_VARIABLE = "GENFRAMES_STORAGE_ROOT"


@dataclass(frozen=True)
class StorageLayout:
    root: Path

    @classmethod
    def resolve(cls, explicit_root: str | Path | None = None) -> StorageLayout:
        configured = explicit_root or os.environ.get(STORAGE_ENVIRONMENT_VARIABLE)
        if configured is None:
            raise RuntimeError(
                f"set {STORAGE_ENVIRONMENT_VARIABLE} or pass an explicit storage root; "
                "large files must not silently fall back to the repository disk"
            )
        root = Path(configured).expanduser().resolve()
        return cls(root=root)

    @property
    def archives(self) -> Path:
        return self.root / "archives"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def benchmarks(self) -> Path:
        return self.root / "benchmarks"

    @property
    def extracted_frames(self) -> Path:
        return self.root / "extracted-frames"

    def create(self) -> None:
        for directory in (
            self.archives,
            self.datasets,
            self.cache,
            self.checkpoints,
            self.benchmarks,
            self.extracted_frames,
        ):
            directory.mkdir(parents=True, exist_ok=True)
