"""Manifest-backed real-frame triplets with deterministic spatial sampling."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import DatasetManifest, Split


class ManifestFrameDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest: DatasetManifest | str | Path,
        *,
        dataset_root: str | Path,
        split: Split,
        crop_size: tuple[int, int] | None = None,
        seed: int = 0,
        horizontal_flip: bool = False,
    ) -> None:
        self.manifest = (
            DatasetManifest.load(manifest) if isinstance(manifest, (str, Path)) else manifest
        )
        self.root = Path(dataset_root)
        self.samples = tuple(sample for sample in self.manifest.samples if sample.split is split)
        self.crop_size = crop_size
        self.seed = seed
        self.horizontal_flip = horizontal_flip
        if not self.samples:
            raise ValueError(f"manifest contains no samples for split {split.value}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        frames = [_load_rgb(self.root / path) for path in sample.frame_paths]
        annotation_paths = sample.attributes.get("annotation_paths")
        target_mask = (
            _load_mask(self.root / annotation_paths[sample.target_index])
            if isinstance(annotation_paths, list)
            else torch.zeros((1, sample.height, sample.width), dtype=torch.bool)
        )
        frames, target_mask = self._spatial_transform(frames, target_mask, index)
        height, width = frames[0].shape[-2:]
        zero_flow = torch.zeros((2, height, width), dtype=torch.float32)
        return {
            "frame0": frames[sample.input_indices[0]],
            "frame1": frames[sample.input_indices[1]],
            "target": frames[sample.target_index],
            "time": torch.tensor(sample.target_time, dtype=torch.float32),
            "flow_t0": zero_flow,
            "flow_t1": zero_flow.clone(),
            "flow_valid": torch.tensor(False),
            "object_mask": target_mask,
            "sequence_id": sample.sequence_id,
            "source_kind": "real",
        }

    def _spatial_transform(
        self, frames: list[Tensor], mask: Tensor, index: int
    ) -> tuple[list[Tensor], Tensor]:
        generator = random.Random(self.seed + index * 104_729)
        height, width = frames[0].shape[-2:]
        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            if crop_height > height or crop_width > width:
                raise ValueError(
                    f"crop {self.crop_size} exceeds sample dimensions {(height, width)}"
                )
            top = generator.randint(0, height - crop_height)
            left = generator.randint(0, width - crop_width)
            frames = [
                frame[:, top : top + crop_height, left : left + crop_width] for frame in frames
            ]
            mask = mask[:, top : top + crop_height, left : left + crop_width]
        if self.horizontal_flip and generator.random() < 0.5:
            frames = [frame.flip(-1) for frame in frames]
            mask = mask.flip(-1)
        return frames, mask


def _load_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _load_mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array > 0).unsqueeze(0)
