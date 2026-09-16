from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import convolve2d
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


# ---------------------------------------------------------------------------
# No-reference noise-level estimation.
#
# PSNR/SSIM are meaningless for denoising quality here because the supervision
# target (``full``) is itself noisy, not clean truth: half -> full training
# inherently carries a Noise2Noise component, so the model's output can be
# cleaner than ``full`` *without* improving (and often while slightly lowering)
# PSNR against ``full``.  The recursive scheme's real objective is a
# lower-noise image, so we measure the residual noise level directly.
# ---------------------------------------------------------------------------


def _subsample_slices(v: np.ndarray, max_slices: int) -> np.ndarray:
    """Uniformly subsample along the slice axis to bound runtime.

    Noise is statistically homogeneous across slices, so a few dozen evenly
    spaced slices are ample for a robust std estimate; this keeps the 3D
    median/uniform filters fast on full 360-slice volumes.
    """
    n = v.shape[0]
    if max_slices < 1:
        raise ValueError("max_slices must be positive")
    if n <= max_slices:
        return v
    idx = np.linspace(0, n - 1, max_slices).round().astype(int)
    return v[idx]


def estimate_noise_std(
    volume: np.ndarray,
    *,
    local_size: int = 5,
    tissue_floor: float = 0.01,
    max_slices: int = 32,
) -> float:
    """Robust no-reference noise std (same units as ``volume``).

    Detrend with a local median (median filter) to remove signal -- including
    edges, to which a mean filter over-responds -- keep only *tissue* voxels
    (local median above ``tissue_floor``; excludes air at mu ~ 0), then
    ``sigma = median(|residual|) / 0.6745`` -- the unbiased robust estimator
    for zero-mean Gaussian noise.  Structure (edges) is sparsely populated, so
    the median over residual magnitudes is dominated by flat-region noise.
    Only ``max_slices`` evenly-spaced slices are used (see
    :func:`_subsample_slices`) to keep the 3D median filter fast.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError("volume must be 3D (z, y, x)")
    if local_size < 1 or local_size % 2 == 0:
        raise ValueError("local_size must be a positive odd integer")
    v = _subsample_slices(v, max_slices)
    local_median = median_filter(v, size=local_size, mode="nearest")
    residual = v - local_median
    mask = local_median > tissue_floor
    if not np.any(mask):
        return float("nan")
    r = residual[mask]
    return float(np.median(np.abs(r)) / 0.6744897501960817)


def estimate_noise_std_immerkaer(
    volume: np.ndarray,
    *,
    tissue_floor: float = 0.01,
    max_slices: int = 32,
) -> float:
    """Immerkaer (1996) Laplacian-kernel noise std, tissue-masked.

    ``sigma = sqrt( sum_{tissue}(I * L)^2 / (36 * N_tissue) )`` computed
    per-slice with the 3x3 Laplacian ``L = [[1,-2,1],[-2,4,-2],[1,-2,1]]``,
    then the median over slices.  A second, independent estimator used to
    cross-check :func:`estimate_noise_std`.  Only ``max_slices`` evenly-spaced
    slices are used to bound runtime.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError("volume must be 3D (z, y, x)")
    v = _subsample_slices(v, max_slices)
    lap = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    local_mean = uniform_filter(v, size=5, mode="nearest")
    mask = local_mean > tissue_floor
    sigmas: list[float] = []
    for sl, m in zip(v, mask):
        conv = convolve2d(sl, lap, mode="valid")
        m_valid = m[1:-1, 1:-1]
        if not np.any(m_valid):
            continue
        conv_tissue = conv[m_valid]
        sigmas.append(float(np.sqrt(np.sum(conv_tissue**2) / (36.0 * conv_tissue.size))))
    if not sigmas:
        return float("nan")
    return float(np.median(sigmas))
