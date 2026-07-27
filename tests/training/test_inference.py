import numpy as np
import torch

from cbct_n2n.training.inference import infer_volume


def test_identity_inference_restores_shape_and_intensity():
    volume = np.linspace(0, 100, 2 * 33 * 35, dtype=np.float32).reshape(2, 33, 35)
    output = infer_volume(torch.nn.Identity(), volume, (0, 100), torch.device("cpu"), batch_size=2)
    assert output.shape == volume.shape
    np.testing.assert_allclose(output, volume, atol=1e-5)
