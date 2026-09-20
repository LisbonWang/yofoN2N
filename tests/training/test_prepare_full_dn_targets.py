"""Tests for the overview merge in scripts/prepare_full_dn_targets.py.

The merge exists so a resumed/partial target generation still reports full
coverage to the round gate; without it a rerun that only creates the missing
files would shrink `n_new + n_existing` and fail the gate for no real reason.
"""
from __future__ import annotations

from scripts.prepare_full_dn_targets import merge_overview_rows


def _row(key: str, red: float) -> dict:
    return {"key": key, "noise": {"full_immerkaer": 0.016, "dn_immerkaer": 0.016 * (1 - red)}}


def test_merge_keeps_previous_rows_when_this_run_adds_none():
    previous = {"rows": [_row(f"k{i}", 0.4) for i in range(200)]}
    assert len(merge_overview_rows(previous, [])) == 200


def test_merge_is_keyed_and_new_rows_win():
    previous = {"rows": [_row("a", 0.4), _row("b", 0.4)]}
    merged = {row["key"]: row for row in merge_overview_rows(previous, [_row("b", 0.9)])}
    assert set(merged) == {"a", "b"}
    assert merged["b"]["noise"]["dn_immerkaer"] < previous["rows"][1]["noise"]["dn_immerkaer"]


def test_merge_handles_missing_or_legacy_overview():
    new = [_row("a", 0.4), _row("b", 0.4)]
    assert len(merge_overview_rows(None, new)) == 2
    assert len(merge_overview_rows({}, new)) == 2
    # legacy overview without a 'rows' key must not crash
    assert len(merge_overview_rows({"n_new": 200, "noise_immerkaer": {}}, new)) == 2


def test_merge_sorts_by_key_for_stable_diffs():
    merged = merge_overview_rows(None, [_row("b", 0.4), _row("a", 0.4)])
    assert [row["key"] for row in merged] == ["a", "b"]
