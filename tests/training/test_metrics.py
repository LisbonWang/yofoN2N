import numpy as np
import torch

from cbct_n2n.training.metrics import compute_metrics


def test_metrics_are_zero_for_identical_images():
    image = np.zeros((16, 16), dtype=np.float32)
    metrics = compute_metrics(image, image, data_range=1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == 1.0

