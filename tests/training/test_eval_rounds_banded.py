"""Unit tests for the density-selectivity metric in ``eval_rounds_banded.py``.

The metric is what decides the plan-B delivery round, so its arithmetic needs
to be pinned: ``high/low`` sigma ratio per checkpoint, ~1.0 meaning "no density
adaptation" (bone and soft tissue smoothed equally hard).
"""
from __future__ import annotations

import pytest

from scripts.eval_rounds_banded import band_differentiation


def _bands(**vals: tuple[float, float]) -> dict[str, dict[str, float]]:
    """``{"M0": (low, high)}`` -> the ``bands`` structure the script builds."""
    return {
        "low": {k: v[0] for k, v in vals.items()},
        "mid": {(k): (v[0] + v[1]) / 2 for k, v in vals.items()},
        "high": {k: v[1] for k, v in vals.items()},
    }


def test_ratio_is_one_when_denoising_is_band_agnostic():
    bands = _bands(full=(0.02, 0.02), M0=(0.01, 0.01), M1b=(0.01, 0.01))
    diff = band_differentiation(bands, ["M0", "M1b"])
    assert diff == pytest.approx({"M0": 1.0, "M1b": 1.0, "full": 1.0})


def test_ratio_below_one_means_soft_tissue_protected():
    # bone carried an extra recursion step: 0.015 -> 0.007 while soft stayed put
    bands = _bands(full=(0.03, 0.03), M0=(0.015, 0.015), M1b=(0.015, 0.007))
    diff = band_differentiation(bands, ["M0", "M1b"])
    assert diff["M1b"] == pytest.approx(0.007 / 0.015)
    assert diff["M1b"] < 1.0
    assert diff["M0"] == pytest.approx(1.0)


def test_full_is_reported_too_so_the_baseline_is_visible():
    bands = _bands(full=(0.02, 0.024), M1b=(0.01, 0.007))
    diff = band_differentiation(bands, ["M1b"])
    assert diff["full"] == pytest.approx(1.2)


def test_missing_or_zero_low_band_yields_nan_not_crash():
    bands = {"low": {"M0": 0.0, "M1b": 0.01}, "mid": {}, "high": {"M0": 0.01, "M1b": 0.005}}
    diff = band_differentiation(bands, ["M0", "M1b"])
    assert diff["M0"] != diff["M0"]  # NaN
    assert diff["M1b"] == pytest.approx(0.5)
