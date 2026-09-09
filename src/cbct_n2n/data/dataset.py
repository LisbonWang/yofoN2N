from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Padding:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


def pad_to_multiple(image: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, Padding]:
    if image.ndim != 2 or multiple <= 0:
        raise ValueError("image must be 2D and multiple must be positive")
    h, w = image.shape
    padded_shape = (h + (-h % multiple), w + (-w % multiple))
    if padded_shape == (h, w):
        # Already aligned: skip the zero-alloc + full copy.  Callers either
        # copy again (torch.from_numpy(...).clone()) or stack the slice into a
        # fresh array, so returning the input array is safe.
        return image, Padding((h, w), padded_shape)
    padded = np.zeros(padded_shape, dtype=image.dtype)
    padded[:h, :w] = image
    return padded, Padding((h, w), padded_shape)


def unpad(image: np.ndarray, padding: Padding) -> np.ndarray:
    if tuple(image.shape[-2:]) != padding.padded_shape:
        raise ValueError(f"Expected padded shape {padding.padded_shape}, got {image.shape[-2:]}")
    h, w = padding.original_shape
    return image[..., :h, :w]


class PairedSliceDataset(Dataset):
    """Full axial slices with full-view or cross-subset supervision."""

    def __init__(self, manifest: dict[str, Any], split: str, augment: bool = False,
                 supervision_mode: str = "full_view"):
        if supervision_mode not in {"full_view", "half_view_n2n"}:
            raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
        self.augment = augment
        self.supervision_mode = supervision_mode
        records = {str(item["case_id"]): item for item in manifest["records"]}
        case_ids = manifest["splits"][split]
        self.volumes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self.slices: list[tuple[int, int, str]] = []
        for case_id in case_ids:
            item = records[case_id]
            even = np.load(Path(item["even_path"]), mmap_mode="r")
            odd = np.load(Path(item["odd_path"]), mmap_mode="r")
            full = np.load(Path(item["full_path"]), mmap_mode="r")
            if even.ndim != 3 or even.shape != odd.shape or even.shape != full.shape:
                raise ValueError(f"Invalid paired volumes for {case_id}: even={even.shape}, odd={odd.shape}, full={full.shape}")
            index = len(self.volumes)
            self.volumes.append((case_id, even, odd, full))
            slice_count = even.shape[0]
            self.slices.extend((index, z, source_kind) for z in range(slice_count) for source_kind in ("even", "odd"))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume_index, z, source_kind = self.slices[index]
        case_id, even_volume, odd_volume, full_volume = self.volumes[volume_index]
        source_volume = even_volume if source_kind == "even" else odd_volume
        source = np.asarray(source_volume[z], dtype=np.float32)
        if self.supervision_mode == "full_view":
            target_volume = full_volume
        else:
            target_volume = odd_volume if source_kind == "even" else even_volume
        target = np.asarray(target_volume[z], dtype=np.float32)
        source, padding = pad_to_multiple(source)
        target, target_padding = pad_to_multiple(target)
        if padding != target_padding:
            raise ValueError("Source and target slice padding mismatch")
        source_tensor = torch.from_numpy(source.copy()).unsqueeze(0)
        target_tensor = torch.from_numpy(target.copy()).unsqueeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                source_tensor, target_tensor = torch.flip(source_tensor, (-1,)), torch.flip(target_tensor, (-1,))
            if torch.rand(()) < 0.5:
                source_tensor, target_tensor = torch.flip(source_tensor, (-2,)), torch.flip(target_tensor, (-2,))
            if source_tensor.shape[-2] == source_tensor.shape[-1]:
                turns = int(torch.randint(0, 4, ()).item())
                source_tensor, target_tensor = torch.rot90(source_tensor, turns, (-2, -1)), torch.rot90(target_tensor, turns, (-2, -1))
        return {"source": source_tensor, "target": target_tensor, "case_id": case_id, "slice_index": z, "source_kind": source_kind}
