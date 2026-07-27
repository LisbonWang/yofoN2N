from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def compute_metrics(prediction: np.ndarray, target: np.ndarray, data_range: float) -> dict[str, float]:
    pred, truth = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or not pred.size:
        raise ValueError("prediction and target must be non-empty arrays with matching shapes")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    difference = pred - truth
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(data_range**2 / mse)
    pairs = zip(pred, truth) if pred.ndim >= 3 else [(pred, truth)]
    ssim_values = []
    for left, right in pairs:
        minimum = min(left.shape[-2:])
        window = min(7, minimum if minimum % 2 else minimum - 1)
        if window < 3:
            raise ValueError("SSIM requires image dimensions of at least 3")
        ssim_values.append(float(structural_similarity(left, right, data_range=data_range, win_size=window)))
    return {"mae": mae, "rmse": math.sqrt(mse), "psnr": psnr, "ssim": float(np.mean(ssim_values))}
