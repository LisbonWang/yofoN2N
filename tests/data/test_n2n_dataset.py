"""Regression tests for the configurable :class:`N2NDataset`.

Covers: explicit multi-direction mappings, legacy ``supervision_mode``
translation, per-direction sample expansion, MARFI/BHC residual augmentation
(including the automatic ``marfibhc`` combination), base-vs-residual sample
semantics, missing-volume skip behaviour (whole study vs. single tag),
``.raw``/non-``.raw`` outputs key compatibility and validation errors.
"""

import json

import numpy as np
import pytest
from torch.utils.data import DataLoader

from cbct_n2n.data.n2n_dataset import (
    N2NDataset,
    RESIDUAL_VOLUMES,
    SUPERVISION_MAPPINGS,
)

SHAPE = (2, 3, 5)


def _base_volumes():
    even = np.arange(2 * 3 * 5, dtype=np.float32).reshape(SHAPE)
    return {"even": even, "odd": even + 10.0, "full": even + 5.0}


def _write_volumes(tmp_path, subdir, volumes):
    case_dir = tmp_path / subdir
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, volume in volumes.items():
        volume.astype("<f4").tofile(case_dir / f"{name}.raw")
    return case_dir


def _record(key, subdir, *, case_id=None, volumes=None, output_key_suffix=".raw", status="success"):
    """Build one reconstruction-index-style record.

    ``output_key_suffix`` controls the outputs key style: ``".raw"`` mimics the
    reconstruction index (``even.raw``), ``""`` mimics hand-written manifests
    (``even``).  Relative paths always point at ``<subdir>/<name>.raw``.
    """
    volumes = volumes if volumes is not None else _base_volumes()
    if case_id is None:
        case_id = key if "::" not in key else key.split("::")[1]
    return {
        "key": key,
        "case_id": case_id,
        "reconstruction_status": status,
        "shape": list(SHAPE),
        "dtype": "<f4",
        "outputs": {f"{name}{output_key_suffix}": f"{subdir}/{name}.raw" for name in volumes},
    }


def _write_manifest(tmp_path, records, split_keys, split="train"):
    manifest = {"records": records, "splits": {split: split_keys, "val": [], "test": []}}
    manifest_path = tmp_path / "reconstruction-index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_explicit_mapping_even_to_odd(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("uid::0::Mode2_1609", "case", case_id="YPDL-26A-0009")
    manifest_path = _write_manifest(tmp_path, [record], ["uid::0::Mode2_1609"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["uid::0::Mode2_1609"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
    )

    assert len(dataset) == SHAPE[0]  # one direction, no residual -> 1 item per slice
    item = dataset[0]
    np.testing.assert_allclose(item["source"].numpy()[0, :3, :5], volumes["even"][0])
    np.testing.assert_allclose(item["target"].numpy()[0, :3, :5], volumes["odd"][0])
    assert item["mapping"] == "even->odd"
    assert item["source_kind"] == "even"
    assert item["target_kind"] == "odd"
    assert item["residual"] == ""
    assert item["case_id"] == "YPDL-26A-0009"
    assert item["source"].shape == (1, 32, 32)


def test_multiple_mappings_expand_each_slice_and_mix(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case")
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd", "odd->even"],
    )

    assert len(dataset) == 2 * SHAPE[0]  # each slice expanded across both directions
    items = [dataset[i] for i in range(len(dataset))]
    assert {item["mapping"] for item in items} == {"even->odd", "odd->even"}
    # the same z-slice appears under both directions
    z0 = [item for item in items if item["slice_index"] == 0]
    assert {item["mapping"] for item in z0} == {"even->odd", "odd->even"}
    # numerical check per direction
    even_target = [item for item in items if item["mapping"] == "odd->even"][0]
    np.testing.assert_allclose(even_target["source"].numpy()[0, :3, :5], volumes["odd"][0])
    np.testing.assert_allclose(even_target["target"].numpy()[0, :3, :5], volumes["even"][0])


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("half_view_n2n", {("even", "odd"), ("odd", "even")}),
        ("full_view", {("even", "full"), ("odd", "full")}),
    ],
)
def test_legacy_supervision_mode_translation(tmp_path, mode, expected):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case")
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        supervision_mode=mode,
    )

    assert dataset.mappings == tuple(sorted(expected))
    items = [dataset[i] for i in range(len(dataset))]
    assert {(item["source_kind"], item["target_kind"]) for item in items} == expected


def test_explicit_mappings_take_precedence_over_supervision_mode(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case")
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->full"],
        supervision_mode="half_view_n2n",
    )

    assert dataset.mappings == (("even", "full"),)
    assert len(dataset) == SHAPE[0]


def test_plain_outputs_keys_legacy_style(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case", output_key_suffix="")  # plain keys like hand-written manifests
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["odd->full"],
    )

    item = dataset[0]
    np.testing.assert_allclose(item["source"].numpy()[0, :3, :5], volumes["odd"][0])
    np.testing.assert_allclose(item["target"].numpy()[0, :3, :5], volumes["full"][0])


def test_bhc_residual_appends_pair_and_shifts_source_and_target(tmp_path):
    volumes = _base_volumes()
    volumes["full.bhc"] = volumes["full"] + 0.5  # residual R = 0.5 everywhere
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case", volumes=volumes)
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
        residuals=["bhc"],
    )

    # base sample is preserved; one residual-augmented sample is appended
    assert len(dataset) == 2 * SHAPE[0]
    items = [dataset[i] for i in range(len(dataset))]
    base_items = [item for item in items if item["residual"] == ""]
    residual_items = [item for item in items if item["residual"] == "bhc"]
    assert len(base_items) == SHAPE[0]
    assert len(residual_items) == SHAPE[0]
    residual = volumes["full.bhc"] - volumes["full"]
    for item in residual_items:
        z = item["slice_index"]
        np.testing.assert_allclose(item["source"].numpy()[0, :3, :5], volumes["even"][z] + residual[z])
        np.testing.assert_allclose(item["target"].numpy()[0, :3, :5], volumes["odd"][z] + residual[z])
        assert item["residual"] == "bhc"


def test_bhc_plus_marfi_auto_enables_marfibhc(tmp_path):
    volumes = _base_volumes()
    volumes["full.bhc"] = volumes["full"] + 0.5
    volumes["full.marfi"] = volumes["full"] + 1.0
    volumes["full.marfi.bhc"] = volumes["full"] + 1.5
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case", volumes=volumes)
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
        residuals=["bhc", "marfi"],
    )

    assert dataset.residual_tags == ("bhc", "marfi", "marfibhc")
    assert len(dataset) == SHAPE[0] * (1 + 3)  # base + 3 residuals per slice
    items = [dataset[i] for i in range(len(dataset))]
    tags = {item["residual"] for item in items}
    assert tags == {"", "bhc", "marfi", "marfibhc"}
    # numeric check of the combined residual
    residual = volumes["full.marfi.bhc"] - volumes["full"]
    combined = [item for item in items if item["residual"] == "marfibhc"][0]
    z = combined["slice_index"]
    np.testing.assert_allclose(combined["source"].numpy()[0, :3, :5], volumes["even"][z] + residual[z])
    np.testing.assert_allclose(combined["target"].numpy()[0, :3, :5], volumes["odd"][z] + residual[z])


def test_missing_variant_file_disables_only_that_tag(tmp_path):
    volumes = _base_volumes()
    volumes["full.bhc"] = volumes["full"] + 0.5
    volumes["full.marfi"] = volumes["full"] + 1.0
    # full.marfi.bhc deliberately *not* written, though bhc+marfi are enabled ->
    # marfibhc is auto-selected at normalize time but must be disabled per record
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case", volumes=volumes)
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
        residuals=["bhc", "marfi"],
    )

    # both marfibhc and marfi disabled? No: marfi volume exists, marfibhc does not.
    # -> available tags are bhc and marfi only.
    items = [dataset[i] for i in range(len(dataset))]
    tags = {item["residual"] for item in items}
    assert tags == {"", "bhc", "marfi"}
    assert dataset.residual_tags == ("bhc", "marfi", "marfibhc")  # selection keeps marfibhc
    assert len(dataset) == SHAPE[0] * (1 + 2)


def test_variant_missing_from_outputs_disables_tag_keeps_base(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)  # no full.bhc file at all
    record = _record("s1", "case")
    record["outputs"]["full.bhc.raw"] = "case/missing.raw"  # key present, file absent
    record["outputs"]["full.marfi.raw"] = "case/also_missing.raw"
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->full"],
        residuals=["bhc", "marfi"],
    )

    # all residual files missing -> only base samples survive
    items = [dataset[i] for i in range(len(dataset))]
    assert all(item["residual"] == "" for item in items)
    assert len(dataset) == SHAPE[0]
    assert len(dataset.volumes) == 1  # study itself kept


def test_missing_base_volume_skips_whole_study(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "complete", volumes)
    good = _record("complete-study", "complete", case_id="complete")
    # a second study that lacks the "even" base volume required by the mapping
    bad_volumes = {"odd": volumes["odd"], "full": volumes["full"]}
    _write_volumes(tmp_path, "noeven", bad_volumes)
    bad = _record("noeven-study", "noeven", case_id="noeven", volumes=bad_volumes)
    manifest_path = _write_manifest(tmp_path, [good, bad], ["complete-study", "noeven-study"])

    dataset = N2NDataset(
        {"records": [good, bad], "splits": {"train": ["complete-study", "noeven-study"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
    )

    assert len(dataset.volumes) == 1
    assert {v["case_id"] for v in dataset.volumes} == {"complete"}
    assert len(dataset) == SHAPE[0]


def test_failed_reconstruction_is_skipped(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "good", volumes)
    _write_volumes(tmp_path, "bad", volumes)
    good = _record("good", "good", status="success")
    bad = _record("bad", "bad", status="failed")
    manifest_path = _write_manifest(tmp_path, [good, bad], ["good", "bad"])

    dataset = N2NDataset(
        {"records": [good, bad], "splits": {"train": ["good", "bad"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
    )

    assert len(dataset.volumes) == 1
    assert dataset.volumes[0]["case_id"] == "good"


def test_multiple_studies_of_same_case_do_not_collide(tmp_path):
    studies = []
    for n, value in enumerate((np.full(SHAPE, 1.0, dtype=np.float32), np.full(SHAPE, 2.0, dtype=np.float32))):
        volumes = {"even": value, "odd": value + 10.0, "full": value + 5.0}
        _write_volumes(tmp_path, f"study{n}", volumes)
        # same patient, two studies -> distinct composite keys, same case_id
        studies.append(_record(f"uid::{n}::Mode2_1609", f"study{n}", case_id="YPDL-26A-0009", volumes=volumes))
    manifest_path = _write_manifest(tmp_path, studies, [s["key"] for s in studies])

    dataset = N2NDataset(
        {"records": studies, "splits": {"train": [s["key"] for s in studies], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd"],
    )

    assert len(dataset.volumes) == 2
    sources = {float(v["memmaps"]["even"][0, 0, 0]) for v in dataset.volumes}
    assert sources == {1.0, 2.0}


def test_no_usable_records_raises(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "bad", volumes)
    bad = _record("bad", "bad", status="failed")
    manifest_path = _write_manifest(tmp_path, [bad], ["bad"])

    with pytest.raises(ValueError, match="No usable reconstruction records"):
        N2NDataset(
            {"records": [bad], "splits": {"train": ["bad"], "val": [], "test": []}},
            manifest_path,
            "train",
            mappings=["even->odd"],
        )


def test_max_slices_per_case_limits_count(tmp_path):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case")
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd", "odd->even"],
        max_slices_per_case=1,
    )

    assert len(dataset) == 2  # 1 slice x 2 directions
    assert {item["slice_index"] for item in (dataset[0], dataset[1])} == {0}


def test_dataloader_batches_base_and_residual_samples(tmp_path):
    volumes = _base_volumes()
    volumes["full.bhc"] = volumes["full"] + 0.5
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case", volumes=volumes)
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])

    dataset = N2NDataset(
        {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}},
        manifest_path,
        "train",
        mappings=["even->odd", "odd->even"],
        residuals=["bhc"],
    )

    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), num_workers=0)))
    assert batch["source"].shape == (len(dataset), 1, 32, 32)
    assert batch["target"].shape == batch["source"].shape
    assert len(batch["mapping"]) == len(dataset)
    assert set(batch["residual"]).issubset({"", "bhc"})


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mappings": [], "supervision_mode": "half_view_n2n"}, "mappings is empty"),
        ({"mappings": ["full->even"]}, "source must be one of"),
        ({"mappings": ["even->even"]}, "must differ"),
        ({"mappings": ["even->odd"], "residuals": ["xyz"]}, "unknown residual tags"),
        ({"supervision_mode": "bogus"}, "unknown supervision_mode"),
        ({}, "either mappings or supervision_mode"),
    ],
)
def test_validation_errors(tmp_path, kwargs, match):
    volumes = _base_volumes()
    _write_volumes(tmp_path, "case", volumes)
    record = _record("s1", "case")
    manifest_path = _write_manifest(tmp_path, [record], ["s1"])
    manifest = {"records": [record], "splits": {"train": ["s1"], "val": [], "test": []}}

    with pytest.raises(ValueError, match=match):
        N2NDataset(manifest, manifest_path, "train", **kwargs)


def test_mapping_table_consistency():
    """The legacy translation table must only produce allowed directions."""
    allowed_sources = {"even", "odd"}
    allowed_targets = {"even", "odd", "full"}
    for mode, specs in SUPERVISION_MAPPINGS.items():
        for spec in specs:
            source, target = spec.split("->")
            assert source in allowed_sources
            assert target in allowed_targets
            assert source != target
    assert set(SUPERVISION_MAPPINGS) == {"full_view", "half_view_n2n"}
    assert set(RESIDUAL_VOLUMES) == {"bhc", "marfi", "marfibhc"}
