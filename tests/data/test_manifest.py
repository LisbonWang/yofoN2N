import pytest

from cbct_n2n.data.manifest import create_case_splits


def test_case_splits_are_deterministic_and_disjoint():
    records = [{"case_id": f"c{i}"} for i in range(10)]
    first = create_case_splits(records, (0.6, 0.2, 0.2), seed=7)
    second = create_case_splits(records, (0.6, 0.2, 0.2), seed=7)
    assert first == second
    splits = first["splits"]
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert len(splits["train"]) == 6


def test_case_splits_reject_bad_ratios():
    with pytest.raises(ValueError):
        create_case_splits([{"case_id": "c"}], (1, 1, 1), 1)


def test_case_splits_reject_empty_positive_ratio_partition():
    with pytest.raises(ValueError, match="at least one case"):
        create_case_splits([{"case_id": "c"}], (0.8, 0.1, 0.1), 1)
