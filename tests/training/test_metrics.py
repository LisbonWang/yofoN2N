import numpy as np
import pytest
import torch

from cbct_n2n.training.metrics import (
    compute_metrics,
    estimate_noise_std,
    estimate_noise_std_immerkaer,
)


def test_metrics_are_zero_for_identical_images():
    image = np.zeros((16, 16), dtype=np.float32)
    metrics = compute_metrics(image, image, data_range=1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == 1.0


# ---------------------------------------------------------------------------
# No-reference noise-level estimators (recursive-scheme gate).
# ---------------------------------------------------------------------------


def _phantom(z=8, y=96, x=96, sigma=0.01):
    """A soft-tissue block (mu ~ 0.1) surrounded by air (mu ~ 0) plus Gaussian noise."""
    rng = np.random.default_rng(0)
    signal = np.zeros((z, y, x), dtype=np.float32)
    signal[:, 30:70, 30:70] = 0.10
    noisy = (signal + rng.normal(0, sigma, signal.shape)).astype(np.float32)
    return noisy, sigma


def test_noise_estimators_recover_sigma():
    noisy, sigma = _phantom()
    for estimator in (estimate_noise_std, estimate_noise_std_immerkaer):
        est = estimator(noisy)
        assert np.isfinite(est)
        # within 30% of the injected sigma (robust estimator, tiny phantom)
        assert 0.7 * sigma <= est <= 1.3 * sigma


def test_cleaner_volume_ranks_below_noisier():
    rng = np.random.default_rng(1)
    signal = np.zeros((8, 96, 96), dtype=np.float32)
    signal[:, 30:70, 30:70] = 0.10
    noisy = (signal + rng.normal(0, 0.010, signal.shape)).astype(np.float32)
    cleaner = (signal + rng.normal(0, 0.005, signal.shape)).astype(np.float32)
    for estimator in (estimate_noise_std, estimate_noise_std_immerkaer):
        assert estimator(cleaner) < estimator(noisy)


def test_noise_estimators_reject_non_3d_input():
    with pytest.raises(ValueError):
        estimate_noise_std(np.zeros((96, 96), dtype=np.float32))
    with pytest.raises(ValueError):
        estimate_noise_std_immerkaer(np.zeros((96, 96), dtype=np.float32))


def test_median_estimator_rejects_even_local_size():
    noisy, _ = _phantom()
    with pytest.raises(ValueError):
        estimate_noise_std(noisy, local_size=4)

