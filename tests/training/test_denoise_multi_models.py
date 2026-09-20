"""Unit tests for the multi-model judging driver (pure helpers, no GPU needed)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.denoise_multi_models import (
    DTYPE,
    TISSUE_FLOOR,
    band_masks,
    expected_bytes,
    output_path_for,
    parse_model_spec,
    severity_stats,
)


def test_parse_model_spec_returns_name_and_path():
    name, path = parse_model_spec("r1b=outputs/judge/ckpt/r1b.best.pt")
    assert name == "r1b"
    assert path == Path("outputs/judge/ckpt/r1b.best.pt")


def test_parse_model_spec_strips_surrounding_space():
    name, path = parse_model_spec("  r2b = runs/x/best.pt  ")
    assert name == "r2b"
    assert path == Path("runs/x/best.pt")


def test_parse_model_spec_splits_on_first_equals_only():
    # Windows drive letters and any '=' inside the path must survive.
    name, path = parse_model_spec("r0=C:/a=b/best.pt")
    assert name == "r0"
    assert path == Path("C:/a=b/best.pt")


@pytest.mark.parametrize("spec", ["r1b", "", "=runs/x/best.pt", "r1b=", "a b=runs/x.pt", "a/b=runs/x.pt"])
def test_parse_model_spec_rejects_bad_input(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_spec(spec)


def test_output_path_appends_model_tag_after_full_domain_name():
    # 'full.bhc.raw' must become 'full.bhc.r1b.raw' -- the domain keeps its dots.
    got = output_path_for(Path("outputs/judge/full.bhc.raw"), "r1b")
    assert got == Path("outputs/judge/full.bhc.r1b.raw")


def test_output_path_handles_single_dot_name():
    got = output_path_for(Path("out/full.raw"), "r0")
    assert got == Path("out/full.r0.raw")


def test_output_path_honours_outdir():
    got = output_path_for(Path("inputs/full.bhc.raw"), "r2b", Path("results"))
    assert got == Path("results/full.bhc.r2b.raw")


def test_expected_bytes_matches_float32_volume():
    assert expected_bytes((360, 640, 640)) == 589_824_000
    assert expected_bytes((2, 3, 4)) == 2 * 3 * 4 * DTYPE.itemsize


def test_band_masks_partition_tissue_exactly():
    vol = np.array([[[TISSUE_FLOOR, 0.05, 0.2, 0.3, 0.4, 0.9]]], dtype=np.float32)
    masks = band_masks(vol, lo=0.2, hi=0.4)
    # air (<= floor) is in no band; the low edge is closed (<=lo) and the high edge is
    # open (>lo) so 0.2 -> low, 0.4 -> high, and 0.3 is the only mid voxel.
    assert [int(m.sum()) for m in (masks["low"], masks["mid"], masks["high"])] == [2, 1, 2]
    assert bool(masks["low"][0, 0, 1]) and bool(masks["low"][0, 0, 2])
    assert bool(masks["mid"][0, 0, 3])
    assert not bool(masks["mid"][0, 0, 2]) and not bool(masks["mid"][0, 0, 4])
    assert bool(masks["high"][0, 0, 4]) and bool(masks["high"][0, 0, 5])
    assert not bool(masks["low"][0, 0, 0])


def test_band_masks_are_disjoint():
    rng = np.random.default_rng(0)
    vol = np.abs(rng.normal(0.2, 0.2, size=(4, 8, 8))).astype(np.float32)
    masks = band_masks(vol, lo=0.2, hi=0.4)
    total = masks["low"].astype(int) + masks["mid"].astype(int) + masks["high"].astype(int)
    assert total.max() == 1
    assert np.array_equal(total.astype(bool), vol > TISSUE_FLOOR)


def test_severity_stats_reports_extremes():
    vol = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    stats = severity_stats(vol)
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] > 0.0
