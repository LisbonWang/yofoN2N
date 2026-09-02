import json

import numpy as np
from torch.utils.data import DataLoader

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

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["source"].shape == (2, 1, 32, 32)


def test_raw_dataset_keys_keep_multiple_studies_of_same_case(tmp_path):
    # 同一 case 两次扫描（同 case_id、不同 key）：旧逻辑按 case_id 索引会覆盖，
    # 新逻辑必须用 manifest key 区分（study_uid::group 复合键场景）。
    studies = []
    for n, volume in enumerate((
        np.full((2, 3, 5), 1.0, dtype=np.float32),
        np.full((2, 3, 5), 2.0, dtype=np.float32),
    )):
        case_dir = tmp_path / f"study{n}"
        case_dir.mkdir()
        for name, value in (("even", volume), ("odd", volume + 10), ("full", volume + 5)):
            value.astype("<f4").tofile(case_dir / f"{name}.raw")
        studies.append({
            "key": f"uid::{n}::Mode2_1609",  # 模拟 study_uid::group 复合键
            "case_id": "YPDL-26A-0009",      # 同一 case
            "reconstruction_status": "success",
            "shape": [2, 3, 5],
            "dtype": "<f4",
            "outputs": {name: f"study{n}/{name}.raw" for name in ("even", "odd", "full")},
        })
    manifest = {"records": studies, "splits": {"train": ["uid::0::Mode2_1609", "uid::1::Mode2_1609"], "val": [], "test": []}}
    manifest_path = tmp_path / "reconstruction-index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = RawPairedSliceDataset(manifest, manifest_path, "train")

    assert len(dataset.volumes) == 2  # 不再互相覆盖
    sources = {float(dataset.volumes[i][1][0, 0, 0]) for i in range(2)}  # even 卷角点值
    assert sources == {1.0, 2.0}
    assert {s["source_kind"] for s in (dataset[0], dataset[1])} <= {"even", "odd"}
