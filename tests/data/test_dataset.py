import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cbct_n2n.data.dataset import PairedSliceDataset, pad_to_multiple, unpad


def test_padding_and_unpadding_round_trip():
    value = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    padded, padding = pad_to_multiple(value, 4)
    assert padded.shape == (4, 8)
    np.testing.assert_array_equal(unpad(padded, padding), value)


def test_dataset_expands_each_axial_slice_to_even_and_odd_full_targets(tmp_path):
    even = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5) - 2
    odd = even + 2
    full = even + 1
    np.save(tmp_path / "even.npy", even)
    np.save(tmp_path / "odd.npy", odd)
    np.save(tmp_path / "full.npy", full)
    record = {"case_id": "c", "even_path": str(tmp_path / "even.npy"), "odd_path": str(tmp_path / "odd.npy"), "full_path": str(tmp_path / "full.npy"), "shape": [2, 3, 5]}
    (tmp_path / "pair.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = {"records": [record], "splits": {"train": ["c"]}}
    ds = PairedSliceDataset(manifest, "train", augment=False)
    assert len(ds) == 4
    even_sample, odd_sample = ds[0], ds[1]
    assert even_sample["source"].shape == (1, 32, 32)
    assert even_sample["target"].shape == even_sample["source"].shape
    assert even_sample["source_kind"] == "even"
    assert odd_sample["source_kind"] == "odd"
    assert even_sample["slice_index"] == odd_sample["slice_index"] == 0
    np.testing.assert_allclose(even_sample["source"][..., :3, :5], even[0][None])
    np.testing.assert_allclose(odd_sample["source"][..., :3, :5], odd[0][None])
    np.testing.assert_allclose(even_sample["target"][..., :3, :5], full[0][None])
    np.testing.assert_allclose(odd_sample["target"][..., :3, :5], full[0][None])
    assert ds[2]["slice_index"] == ds[3]["slice_index"] == 1
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["source"].shape == (2, 1, 32, 32)
    assert batch["target"].shape == (2, 1, 32, 32)
    assert batch["source_kind"] == ["even", "odd"]


def test_dataset_cross_pairs_half_view_n2n_targets(tmp_path):
    even = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    odd = even + 10
    full = even + 5
    for kind, volume in (("even", even), ("odd", odd), ("full", full)):
        np.save(tmp_path / f"{kind}.npy", volume)
    record = {"case_id": "c", **{f"{kind}_path": str(tmp_path / f"{kind}.npy") for kind in ("even", "odd", "full")}}
    manifest = {"records": [record], "splits": {"train": ["c"]}}

    dataset = PairedSliceDataset(manifest, "train", supervision_mode="half_view_n2n")

    np.testing.assert_allclose(dataset[0]["source"][..., :3, :5], even[0][None])
    np.testing.assert_allclose(dataset[0]["target"][..., :3, :5], odd[0][None])
    np.testing.assert_allclose(dataset[1]["source"][..., :3, :5], odd[0][None])
    np.testing.assert_allclose(dataset[1]["target"][..., :3, :5], even[0][None])


def test_dataset_rejects_unknown_supervision_mode(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)
    for kind in ("even", "odd", "full"):
        np.save(tmp_path / f"{kind}.npy", volume)
    record = {"case_id": "c", **{f"{kind}_path": str(tmp_path / f"{kind}.npy") for kind in ("even", "odd", "full")}}
    manifest = {"records": [record], "splits": {"train": ["c"]}}

    with pytest.raises(ValueError, match="Unsupported supervision mode"):
        PairedSliceDataset(manifest, "train", supervision_mode="unsupported")


def test_augmented_training_preserves_source_kind_and_batches_non_square(tmp_path):
    even = np.zeros((2, 5, 37), dtype=np.float32)
    odd = np.ones((2, 5, 37), dtype=np.float32)
    full = np.full((2, 5, 37), 0.5, dtype=np.float32)
    np.save(tmp_path / "even.npy", even)
    np.save(tmp_path / "odd.npy", odd)
    np.save(tmp_path / "full.npy", full)
    record = {"case_id": "c", "even_path": str(tmp_path / "even.npy"), "odd_path": str(tmp_path / "odd.npy"), "full_path": str(tmp_path / "full.npy"), "shape": [2, 5, 37]}
    manifest = {"records": [record], "splits": {"train": ["c"]}}
    ds = PairedSliceDataset(manifest, "train", augment=True)
    assert ds[0]["source"].sum().item() == 0
    assert ds[1]["source_kind"] == "odd"
    assert ds[1]["source"].sum().item() == 5 * 37
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["source"].shape == (2, 1, 32, 64)


def test_dataset_limits_each_case_to_requested_axial_slices(tmp_path):
    volume = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2) - 2
    for kind in ("even", "odd", "full"):
        np.save(tmp_path / f"{kind}.npy", volume)
    record = {"case_id": "c", **{f"{kind}_path": str(tmp_path / f"{kind}.npy") for kind in ("even", "odd", "full")}}
    manifest = {"records": [record], "splits": {"train": ["c"]}}
    ds = PairedSliceDataset(manifest, "train", augment=False, max_slices_per_case=1)
    assert len(ds) == 2
    np.testing.assert_allclose(ds[0]["source"][..., :2, :2], volume[0][None])
