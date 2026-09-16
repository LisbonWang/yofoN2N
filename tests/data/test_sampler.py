"""Tests for :class:`GroupedShuffleSampler` (two-level grouped shuffle)."""

import random

import pytest

from cbct_n2n.data.sampler import GroupedShuffleSampler


def test_covers_all_indices_exactly_once():
    sampler = GroupedShuffleSampler([0, 0, 1, 1, 1, 2])
    random.seed(0)
    out = list(sampler)
    assert sorted(out) == list(range(6))
    assert len(out) == 6


def test_each_group_is_one_contiguous_block():
    group_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    sampler = GroupedShuffleSampler(group_ids)
    random.seed(42)
    out = list(sampler)
    # The group of each yielded sample, compressed to consecutive runs: every
    # group must appear as exactly one contiguous block (no interleaving).
    group_seq = [group_ids[i] for i in out]
    runs = [g for i, g in enumerate(group_seq) if i == 0 or g != group_seq[i - 1]]
    assert sorted(runs) == sorted(set(group_ids))
    assert len(runs) == len(set(group_ids))


def test_len_matches_group_ids():
    sampler = GroupedShuffleSampler([0, 0, 1, 2, 2])
    assert len(sampler) == 5


def test_iteration_is_fresh_each_call():
    # With a fixed global seed the two iterations must differ (the global RNG
    # advances), proving each epoch reshuffles rather than repeating.
    group_ids = [0] * 5 + [1] * 5 + [2] * 5
    sampler = GroupedShuffleSampler(group_ids)
    random.seed(7)
    first = list(sampler)
    second = list(sampler)
    assert sorted(first) == sorted(second) == list(range(15))
    assert first != second


def test_reproducible_given_same_global_seed():
    group_ids = [0, 0, 1, 1, 1, 2, 2]
    a = GroupedShuffleSampler(group_ids)
    b = GroupedShuffleSampler(group_ids)
    random.seed(123)
    out_a = list(a)
    random.seed(123)
    out_b = list(b)
    assert out_a == out_b
