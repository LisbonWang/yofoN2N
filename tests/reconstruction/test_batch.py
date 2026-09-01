from pathlib import Path

import numpy as np

from cbct_n2n.reconstruction import (
    load_index,
    merge_index_records,
    verify_volumes,
)


def _record(uid: str, status: str = "success", shape=None, dtype="<f4", group: str = "Mode2_1609"):
    names = ["even.raw", "odd.raw", "full.raw", "full.marfi.raw", "full.bhc.raw", "full.marfi.bhc.raw"]
    return {
        "case_id": "CASE-1",
        "study_uid": uid,
        "scan_id": uid,
        "mode": "Mode2",
        "group": group,
        "top_dir": "PDL",
        "status": status,
        "shape": shape or [8, 6, 4],
        "dtype": dtype,
        "outputs": {name: f"PDL/{group}/CASE-1/{uid}/{name}" for name in names},
    }


def test_load_index_missing(tmp_path):
    assert load_index(tmp_path / "nope.json") is None


def test_load_index_invalid_or_wrong_schema(tmp_path):
    p = tmp_path / "index.json"
    p.write_text("not json", encoding="utf-8")
    assert load_index(p) is None
    p.write_text('{"schema_version": 99, "records": []}', encoding="utf-8")
    assert load_index(p) is None


def test_load_index_roundtrip(tmp_path):
    p = tmp_path / "index.json"
    p.write_text('{"schema_version": 1, "records": [{"study_uid": "A"}]}', encoding="utf-8")
    data = load_index(p)
    assert data is not None and data["records"] == [{"study_uid": "A"}]


def test_merge_index_records_fresh_wins_and_order_preserved():
    existing = [_record("A"), _record("B")]
    fresh = [_record("B", status="error", shape=None)]
    merged = merge_index_records(existing, fresh)
    assert [r["study_uid"] for r in merged] == ["A", "B"]
    assert merged[1]["status"] == "error"
    # new uid appended at the end
    merged2 = merge_index_records(merged, [_record("C")])
    assert [r["study_uid"] for r in merged2] == ["A", "B", "C"]
    # non-dict / no-uid entries are dropped
    assert merge_index_records([{"study_uid": "A"}, "junk", {"no_uid": 1}], []) == [{"study_uid": "A"}]


def test_merge_index_records_keys_by_study_plus_group():
    """One study rebuilt into several groups yields one record per (study, group)."""
    existing = [_record("S", group="Mode2_1609"), _record("S", group="Mode2_0808")]
    fresh = [_record("S", group="Mode2_0808", status="error", shape=None)]
    merged = merge_index_records(existing, fresh)
    assert len(merged) == 2
    by_group = {r["group"]: r["status"] for r in merged}
    assert by_group == {"Mode2_1609": "success", "Mode2_0808": "error"}
    # fresh record for another group appends
    merged2 = merge_index_records(merged, [_record("T", group="Mode2_1609")])
    assert len(merged2) == 3
    # legacy records without a group merge under the uid-only key
    legacy = [{"study_uid": "S", "status": "old"}]
    merged3 = merge_index_records([], legacy + [_record("S", group="Mode2_1609")])
    assert len(merged3) == 2


def test_verify_volumes_ok(tmp_path):
    record = _record("STUDY-1")
    root = tmp_path
    for name, rel in record["outputs"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.zeros((8, 6, 4), dtype="<f4").tobytes())
    assert verify_volumes(record, root) == []


def test_verify_volumes_reports_missing_and_wrong_size(tmp_path):
    record = _record("STUDY-1")
    root = tmp_path
    items = list(record["outputs"].items())
    for i, (name, rel) in enumerate(items):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if i == 0:
            continue  # leave missing
        size = 96 if i == 1 else 768  # wrong size for the second one
        path.write_bytes(b"\x00" * size)
    issues = verify_volumes(record, root)
    assert len(issues) == 2
    assert any("missing" in i for i in issues)
    assert any("size" in i for i in issues)
