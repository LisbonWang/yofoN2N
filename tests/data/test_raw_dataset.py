import json

import numpy as np

from cbct_n2n.data.raw_dataset import RawPairedSliceDataset


def test_raw_dataset_reads_relative_float32_paths_and_full_target(tmp_path):
    even = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    odd = even + 10
    full = even + 5
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    for name, volume in (("even", even), ("odd", odd), ("full", full)):
        volume.astype("<f4").tofile(case_dir / f"{name}.raw")
    record = {
        "case_id": "case",
        "reconstruction_status": "success",
        "shape": [2, 3, 5],
        "dtype": "<f4",
        "outputs": {name: f"case/{name}.raw" for name in ("even", "odd", "full")},
    }
    manifest = {"records": [record], "splits": {"train": ["case"], "val": ["case"], "test": ["case"]}}
    manifest_path = tmp_path / "reconstruction-index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = RawPairedSliceDataset(manifest, manifest_path, "train")

    np.testing.assert_allclose(dataset[0]["source"][..., :3, :5], even[0][None])
    np.testing.assert_allclose(dataset[0]["target"][..., :3, :5], full[0][None])
