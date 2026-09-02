"""Unit tests for scripts/create_recon_splits.py manifest generation."""

from __future__ import annotations

import json

import pytest

from scripts.create_recon_splits import build_manifest, make_key, validate_location


def _record(uid: str, group: str, case_id: str = "DEV-A") -> dict:
    return {
        "study_uid": uid,
        "group": group,
        "case_id": case_id,
        "status": "success",
        "shape": [4, 8, 8],
        "dtype": "<f4",
        "kv": 100.0,
        "ma": 8.0,
        "mode": group.split("_")[0],
        "nviews": 320,
        "outputs": {f"{name}.raw": f"PDL/{case_id}/{uid}/{name}.raw" for name in ("even", "odd", "full")},
    }


def _index(records: list[dict]) -> dict:
    return {"records": records}


def _all_keys(manifest: dict) -> set:
    return {record["key"] for record in manifest["records"]}


def test_composite_keys_and_split_coverage(tmp_path):
    records = [_record(f"uid-{n}", "Mode2_1609") for n in range(10)]
    manifest, stats = build_manifest(_index(records), groups=["Mode2_1609"], seed=42)
    assert all(record["key"] == make_key(record["study_uid"], record["group"]) for record in manifest["records"])
    assert len(manifest["records"]) == 10
    assigned = set(manifest["splits"]["train"]) | set(manifest["splits"]["val"]) | set(manifest["splits"]["test"])
    assert assigned == _all_keys(manifest)
    assert len(assigned) == sum(len(v) for v in manifest["splits"].values())  # no double assignment
    assert stats["total"] == 10


def test_same_study_uid_across_groups_stays_two_records():
    records = [_record("shared-uid", "Mode2_1609"), _record("shared-uid", "Mode2_0808")]
    manifest, _ = build_manifest(_index(records), groups=["Mode2_1609", "Mode2_0808"],
                                 ratios=(0.5, 0.5, 0.0), seed=1)
    keys = {record["key"] for record in manifest["records"]}
    assert keys == {"shared-uid::Mode2_1609", "shared-uid::Mode2_0808"}


def test_test_groups_force_entire_group_into_test():
    records = [_record(f"m2-{n}", "Mode2_1609") for n in range(10)] + \
              [_record(f"m3-{n}", "Mode3_1609") for n in range(6)]
    manifest, _ = build_manifest(_index(records), groups=["Mode2_1609", "Mode3_1609"],
                                 test_groups=["Mode3_1609"], seed=42)
    test_keys = set(manifest["splits"]["test"])
    assert all("Mode3_1609" in key for key in test_keys if key.startswith("m3-"))
    forced = {record["key"] for record in manifest["records"] if record["group"] == "Mode3_1609"}
    assert forced <= test_keys


def test_deterministic_with_seed():
    records = [_record(f"uid-{n}", "Mode2_1609") for n in range(30)]
    a, _ = build_manifest(_index(records), groups=["Mode2_1609"], seed=7)
    b, _ = build_manifest(_index(records), groups=["Mode2_1609"], seed=7)
    assert a["splits"] == b["splits"]


def test_status_filter_and_missing_base_stats():
    records = [
        _record("ok", "Mode2_1609"),
        {**_record("bad", "Mode2_1609"), "status": "failed"},
        {**_record("noeven", "Mode2_1609"),
         "outputs": {"odd.raw": "x/odd.raw", "full.raw": "x/full.raw"}},
    ]
    manifest, stats = build_manifest(_index(records), groups=["Mode2_1609"],
                                     ratios=(1.0, 0.0, 0.0), seed=42)
    assert {record["key"] for record in manifest["records"]} == {"ok::Mode2_1609", "noeven::Mode2_1609"}
    assert stats["missing_base"]["even"] == 1


def test_unknown_group_rejected():
    with pytest.raises(ValueError, match="not present"):
        build_manifest(_index([_record("a", "Mode2_1609")]), groups=["Mode9_9999"])


def test_validate_location_requires_same_directory(tmp_path):
    output_root = tmp_path / "root"
    output_root.mkdir()
    validate_location(output_root, output_root / "splits.json")  # ok
    elsewhere = tmp_path / "elsewhere" / "splits.json"
    with pytest.raises(ValueError, match="--out must be inside"):
        validate_location(output_root, elsewhere)
