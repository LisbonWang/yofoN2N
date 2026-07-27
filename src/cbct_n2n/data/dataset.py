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
    padded = np.zeros(padded_shape, dtype=image.dtype)
    padded[:h, :w] = image
    return padded, Padding((h, w), padded_shape)


def unpad(image: np.ndarray, padding: Padding) -> np.ndarray:
    if tuple(image.shape[-2:]) != padding.padded_shape:
        raise ValueError(f"Expected padded shape {padding.padded_shape}, got {image.shape[-2:]}")
    h, w = padding.original_shape
    return image[..., :h, :w]


class PairedSliceDataset(Dataset):
    """Full axial slices; deliberately has no random crop path."""

    def __init__(self, manifest: dict[str, Any], split: str, window: tuple[float, float], augment: bool = False):
        low, high = map(float, window)
        if high <= low:
            raise ValueError("window upper bound must exceed lower bound")
        self.window, self.augment = (low, high), augment
        records = {str(item["case_id"]): item for item in manifest["records"]}
        case_ids = manifest["splits"][split]
        self.volumes: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.slices: list[tuple[int, int]] = []
        for case_id in case_ids:
            item = records[case_id]
            even = np.load(Path(item["even_path"]), mmap_mode="r")
            odd = np.load(Path(item["odd_path"]), mmap_mode="r")
            if even.shape != odd.shape or even.ndim != 3:
                raise ValueError(f"Invalid paired volumes for {case_id}: {even.shape}, {odd.shape}")
            index = len(self.volumes)
            self.volumes.append((case_id, even, odd))
            self.slices.extend((index, z) for z in range(even.shape[0]))

    def __len__(self) -> int:
        return len(self.slices)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        low, high = self.window
        return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume_index, z = self.slices[index]
        case_id, even_volume, odd_volume = self.volumes[volume_index]
        even, odd = self._normalize(even_volume[z]), self._normalize(odd_volume[z])
        even, padding = pad_to_multiple(even)
        odd, odd_padding = pad_to_multiple(odd)
        if padding != odd_padding:
            raise ValueError("Paired slice padding mismatch")
        even_tensor, odd_tensor = torch.from_numpy(even.copy()).unsqueeze(0), torch.from_numpy(odd.copy()).unsqueeze(0)
        source_tensor, target_tensor = even_tensor, odd_tensor
        if self.augment and index % 2 == 1:
            source_tensor, target_tensor = odd_tensor, even_tensor
        if self.augment:
            if torch.rand(()) < 0.5:
                even_tensor, odd_tensor = torch.flip(even_tensor, (-1,)), torch.flip(odd_tensor, (-1,))
            if torch.rand(()) < 0.5:
                even_tensor, odd_tensor = torch.flip(even_tensor, (-2,)), torch.flip(odd_tensor, (-2,))
            if even_tensor.shape[-2] == even_tensor.shape[-1]:
                turns = int(torch.randint(0, 4, ()).item())
                even_tensor, odd_tensor = torch.rot90(even_tensor, turns, (-2, -1)), torch.rot90(odd_tensor, turns, (-2, -1))
            source_tensor, target_tensor = (even_tensor, odd_tensor) if index % 2 == 0 else (odd_tensor, even_tensor)
        return {"even": even_tensor, "odd": odd_tensor, "source": source_tensor, "target": target_tensor, "case_id": case_id, "slice_index": z}
