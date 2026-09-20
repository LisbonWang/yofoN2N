"""Unit tests for scripts/create_recon_splits.py manifest generation."""

from __future__ import annotations

import json
from collections import Counter

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


def test_excluded_1615_groups_never_enter_training_manifest():
    records = [
        _record("m0", "Mode0_1615"),
        _record("m1", "Mode1_1615"),
        _record("m2", "Mode2_1609"),
        _record("m3", "Mode3_1609"),
    ]
    manifest, stats = build_manifest(
        _index(records),
        groups=["Mode0_1615", "Mode1_1615", "Mode2_1609", "Mode3_1609"],
        exclude_groups=[],  # even an accidental override must not re-enable blocked 1615 groups
        ratios=(1.0, 0.0, 0.0),
    )

    assert {record["group"] for record in manifest["records"]} == {"Mode2_1609", "Mode3_1609"}
    assert manifest["exclude_groups"] == ["Mode0_1615", "Mode1_1615"]
    assert stats["excluded_by_group"] == {"Mode0_1615": 1, "Mode1_1615": 1}
    assert all("1615" not in key for keys in manifest["splits"].values() for key in keys)


def test_stratified_case_sampling_balances_target_across_all_cases():
    counts = {"YPDL-A": 100, "YPDL-B": 20, "YPDL-C": 3}
    records = [
        _record(f"{case_id}-{n:02d}", "Mode2_1609", case_id)
        for case_id, count in counts.items()
        for n in range(count)
    ] + [_record("not-ypdl", "Mode2_1609", "OTHER")]

    manifest, stats = build_manifest(
        _index(records),
        groups=["Mode2_1609"],
        case_prefix="YPDL",
        per_case=10,
        target_total=25,
        ratios=(1.0, 0.0, 0.0),
        seed=42,
    )

    selected = Counter(record["case_id"] for record in manifest["records"])
    assert selected == {"YPDL-A": 11, "YPDL-B": 11, "YPDL-C": 3}
    assert stats["available_by_case"] == counts
    assert stats["selected_by_case"] == dict(selected)
    assert len({record["study_uid"] for record in manifest["records"]}) == 25


def test_stratified_case_sampling_rejects_impossible_target():
    records = [_record(f"A-{n}", "Mode2_1609", "YPDL-A") for n in range(4)]
    with pytest.raises(ValueError, match="target_total=5 exceeds 4 available"):
        build_manifest(
            _index(records), groups=["Mode2_1609"], case_prefix="YPDL",
            per_case=20, target_total=5, ratios=(1.0, 0.0, 0.0),
        )


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
