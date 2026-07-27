import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from cbct_n2n.data.dataset import PairedSliceDataset, pad_to_multiple, unpad


def test_padding_and_unpadding_round_trip():
    value = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    padded, padding = pad_to_multiple(value, 4)
    assert padded.shape == (4, 8)
    np.testing.assert_array_equal(unpad(padded, padding), value)


def test_dataset_reads_full_axial_slice_and_shared_transform(tmp_path):
    even = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    odd = even + 2
    np.save(tmp_path / "even.npy", even)
    np.save(tmp_path / "odd.npy", odd)
    record = {"case_id": "c", "even_path": str(tmp_path / "even.npy"), "odd_path": str(tmp_path / "odd.npy"), "shape": [2, 3, 5]}
    (tmp_path / "pair.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = {"records": [record], "splits": {"train": ["c"]}}
    ds = PairedSliceDataset(manifest, "train", window=(0, 20), augment=False)
    sample = ds[0]
    assert sample["even"].shape == (1, 32, 32)
    assert sample["target"].shape == sample["even"].shape
    valid_difference = sample["target"][..., :3, :5] - sample["even"][..., :3, :5]
    assert torch.allclose(valid_difference, torch.full_like(valid_difference, 0.1))
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["even"].shape == (2, 1, 32, 32)


def test_augmented_training_alternates_noise_direction_and_batches_non_square(tmp_path):
    even = np.zeros((2, 5, 37), dtype=np.float32)
    odd = np.ones((2, 5, 37), dtype=np.float32)
    np.save(tmp_path / "even.npy", even)
    np.save(tmp_path / "odd.npy", odd)
    record = {"case_id": "c", "even_path": str(tmp_path / "even.npy"), "odd_path": str(tmp_path / "odd.npy"), "shape": [2, 5, 37]}
    manifest = {"records": [record], "splits": {"train": ["c"]}}
    ds = PairedSliceDataset(manifest, "train", window=(0, 1), augment=True)
    assert ds[0]["source"].sum().item() == 0
    assert ds[1]["source"].sum().item() == 5 * 37
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["source"].shape == (2, 1, 32, 64)
