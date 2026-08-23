"""Versioned, auditable dataset provenance records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    TEACHER_ONLY = "teacher-only"
    HUMAN_EVALUATION_ONLY = "human-evaluation-only"


class RiskLevel(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    name: str
    canonical_url: str
    retrieved_at: str
    archive_sha256: str | None
    dataset_license: str
    content_license: str
    roles: tuple[DatasetRole, ...]
    risk: RiskLevel
    risk_reason: str
    notes: str = ""

    def validate(self) -> None:
        if not self.source_id or any(char.isspace() for char in self.source_id):
            raise ValueError("source_id must be non-empty and contain no whitespace")
        if not self.canonical_url.startswith(("https://", "http://", "generated://")):
            raise ValueError("canonical_url must be HTTP(S) or generated://")
        if not self.roles:
            raise ValueError("a source must have at least one allowed role")
        if self.archive_sha256 is not None and (
            len(self.archive_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.archive_sha256.lower())
        ):
            raise ValueError("archive_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source_id: str
    sequence_id: str
    split: Split
    frame_paths: tuple[str, ...]
    frame_times: tuple[float, ...]
    input_indices: tuple[int, int]
    target_index: int
    source_fps: float | None
    width: int
    height: int
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def target_time(self) -> float:
        left, right = self.input_indices
        denominator = self.frame_times[right] - self.frame_times[left]
        if denominator <= 0:
            raise ValueError("input frame times must be strictly increasing")
        return (self.frame_times[self.target_index] - self.frame_times[left]) / denominator

    def validate(self) -> None:
        count = len(self.frame_paths)
        if count < 3 or len(self.frame_times) != count:
            raise ValueError("samples require matching paths/times for at least three frames")
        adjacent_times = zip(self.frame_times, self.frame_times[1:], strict=False)
        if any(current >= following for current, following in adjacent_times):
            raise ValueError("frame_times must be strictly increasing")
        left, right = self.input_indices
        if not (0 <= left < self.target_index < right < count):
            raise ValueError("target_index must lie strictly between input indices")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sample dimensions must be positive")
        if not 0.0 < self.target_time < 1.0:
            raise ValueError("derived target time must lie inside (0, 1)")


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    created_at: str
    sources: tuple[SourceRecord, ...]
    samples: tuple[SampleRecord, ...]
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source IDs must be unique")
        for source in self.sources:
            source.validate()
        known_sources = set(source_ids)
        sample_ids: set[str] = set()
        for sample in self.samples:
            sample.validate()
            if sample.sample_id in sample_ids:
                raise ValueError(f"duplicate sample ID: {sample.sample_id}")
            if sample.source_id not in known_sources:
                raise ValueError(f"unknown source ID: {sample.source_id}")
            sample_ids.add(sample.sample_id)

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            name=payload["name"],
            created_at=payload["created_at"],
            schema_version=payload["schema_version"],
            sources=tuple(
                SourceRecord(
                    **{
                        **source,
                        "roles": tuple(DatasetRole(role) for role in source["roles"]),
                        "risk": RiskLevel(source["risk"]),
                    }
                )
                for source in payload["sources"]
            ),
            samples=tuple(
                SampleRecord(
                    **{
                        **sample,
                        "split": Split(sample["split"]),
                        "frame_paths": tuple(sample["frame_paths"]),
                        "frame_times": tuple(sample["frame_times"]),
                        "input_indices": tuple(sample["input_indices"]),
                    }
                )
                for sample in payload["samples"]
            ),
        )
        manifest.validate()
        return manifest


def assign_split(
    source_sequence_id: str,
    *,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    salt: str = "genframes-v1",
) -> Split:
    """Assign all samples from one source sequence to a stable split."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    digest = hashlib.sha256(f"{salt}:{source_sequence_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_fraction:
        return Split.TEST
    if value < test_fraction + validation_fraction:
        return Split.VALIDATION
    return Split.TRAIN
