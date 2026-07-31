from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import pad_to_multiple


class RawPairedSliceDataset(Dataset):
    """Full axial slices from relative-path float32 RAW reconstruction records."""

    def __init__(self, manifest: dict[str, Any], manifest_path: str | Path, split: str, *, augment: bool = False,
                 supervision_mode: str = "full_view", max_slices_per_case: int | None = None):
        if supervision_mode not in {"full_view", "half_view_n2n"}:
            raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
        if max_slices_per_case is not None and max_slices_per_case < 1:
            raise ValueError("max_slices_per_case must be positive when provided")
        self.augment = augment
        self.supervision_mode = supervision_mode
        base = Path(manifest_path).resolve().parent
        records = {str(item["case_id"]): item for item in manifest["records"]}
        self.volumes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self.slices: list[tuple[int, int, str]] = []
        for case_id in manifest["splits"][split]:
            item = records[str(case_id)]
            if item.get("reconstruction_status") not in {None, "success"}:
                continue
            shape = tuple(int(value) for value in item["shape"])
            dtype = np.dtype(item.get("dtype", "<f4"))
            outputs = item["outputs"]
            even = np.memmap(base / outputs["even"], dtype=dtype, mode="r", shape=shape)
            odd = np.memmap(base / outputs["odd"], dtype=dtype, mode="r", shape=shape)
            full = np.memmap(base / outputs["full"], dtype=dtype, mode="r", shape=shape)
            index = len(self.volumes)
            self.volumes.append((str(case_id), even, odd, full))
            count = min(shape[0], max_slices_per_case or shape[0])
            self.slices.extend((index, z, source_kind) for z in range(count) for source_kind in ("even", "odd"))
        if not self.volumes:
            raise ValueError(f"No successful reconstruction records in split {split!r}")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume_index, z, source_kind = self.slices[index]
        case_id, even_volume, odd_volume, full_volume = self.volumes[volume_index]
        source_volume = even_volume if source_kind == "even" else odd_volume
        target_volume = full_volume if self.supervision_mode == "full_view" else (odd_volume if source_kind == "even" else even_volume)
        source, source_padding = pad_to_multiple(np.asarray(source_volume[z], dtype=np.float32))
        target, target_padding = pad_to_multiple(np.asarray(target_volume[z], dtype=np.float32))
        if source_padding != target_padding:
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
