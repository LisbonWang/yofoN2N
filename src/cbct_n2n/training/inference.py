from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cbct_n2n.data.dataset import pad_to_multiple, unpad
from .metrics import compute_metrics


def resolve_input_kinds(input_kind: str) -> tuple[str, ...]:
    if input_kind == "both":
        return ("even", "odd")
    if input_kind in {"even", "odd"}:
        return (input_kind,)
    raise ValueError(f"Unsupported input kind: {input_kind}")


@torch.no_grad()
def infer_volume(model, volume: np.ndarray, device: torch.device, batch_size: int = 4) -> np.ndarray:
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    padded_slices, paddings = [], []
    for image in source:
        padded, padding = pad_to_multiple(image.astype(np.float32))
        padded_slices.append(padded)
        paddings.append(padding)
    model.eval()
    outputs = []
    for start in range(0, len(padded_slices), batch_size):
        batch = torch.from_numpy(np.stack(padded_slices[start:start + batch_size])).unsqueeze(1).to(device)
        outputs.append(model(batch).detach().float().cpu().numpy()[:, 0])
    restored = []
    for image, padding in zip(np.concatenate(outputs, axis=0), paddings):
        restored.append(unpad(image, padding))
    return np.stack(restored).astype(np.float32)


def evaluate_volume(prediction: np.ndarray, target: np.ndarray, metric_data_range: float) -> dict[str, float]:
    return compute_metrics(prediction, target, data_range=metric_data_range)


def write_raw_volume(path: str | Path, volume: np.ndarray) -> None:
    """Write a (z, y, x) volume as headerless little-endian float32 RAW."""
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        for image in source:
            np.asarray(image, dtype="<f4").tofile(stream)
