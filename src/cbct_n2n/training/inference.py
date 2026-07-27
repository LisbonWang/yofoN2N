from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cbct_n2n.data.dataset import pad_to_multiple, unpad
from .metrics import compute_metrics


@torch.no_grad()
def infer_volume(model, volume: np.ndarray, window: tuple[float, float], device: torch.device, batch_size: int = 4) -> np.ndarray:
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    low, high = map(float, window)
    if high <= low or batch_size < 1:
        raise ValueError("window and batch_size are invalid")
    padded_slices, paddings = [], []
    for image in source:
        normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        padded, padding = pad_to_multiple(normalized)
        padded_slices.append(padded)
        paddings.append(padding)
    model.eval()
    outputs = []
    for start in range(0, len(padded_slices), batch_size):
        batch = torch.from_numpy(np.stack(padded_slices[start:start + batch_size])).unsqueeze(1).to(device)
        outputs.append(model(batch).detach().float().cpu().numpy()[:, 0])
    restored = []
    for image, padding in zip(np.concatenate(outputs, axis=0), paddings):
        restored.append(unpad(image, padding) * (high - low) + low)
    return np.stack(restored).astype(np.float32)


def evaluate_volume(prediction: np.ndarray, target: np.ndarray, window: tuple[float, float]) -> dict[str, float]:
    low, high = map(float, window)
    return compute_metrics(prediction, target, data_range=high - low)
