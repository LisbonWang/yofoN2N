"""Unit tests for the conventional-vs-AI comparison renderer (pure helpers, no GPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.compare_conv_denoise import (
    _best_box,
    _cut,
    _load,
    _metrics,
    _window,
    band_masks,
    select_rois,
)
from scripts.denoise_multi_models import band_masks as reference_band_masks

DTYPE = np.dtype("<f4")


# --------------------------------------------------------------------------- band_masks


def test_band_masks_match_the_multi_model_definition():
    key = np.linspace(0.0, 0.6, 1000, dtype=np.float32).reshape(10, 10, 10)
    mine, theirs = band_masks(key, 0.2, 0.4), reference_band_masks(key, 0.2, 0.4)
    for band in ("low", "mid", "high"):
        assert np.array_equal(mine[band], theirs[band]), band


def test_band_masks_exclude_air_and_partition_tissue():
    key = np.array([[[0.0, 0.005, 0.01, 0.05, 0.2, 0.3, 0.4, 0.9]]], dtype=np.float32)
    flat = {band: mask.reshape(-1) for band, mask in band_masks(key, 0.2, 0.4).items()}
    assert not flat["low"][:3].any()  # 0.0/0.005 air, 0.01 is *at* the floor (strict >)
    assert flat["low"][3]             # 0.05 -> soft tissue
    assert flat["low"][4]             # the soft band's upper edge is inclusive (<= lo)
    assert not flat["mid"][4]
    assert flat["mid"][5]             # 0.3 -> transition band
    assert flat["high"][6]            # the bone band's lower edge is inclusive (>= hi)
    assert not flat["mid"][6]
    assert flat["high"][7]


# --------------------------------------------------------------------------- _best_box


def test_best_box_finds_the_dense_cluster():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:40, 40:50] = True  # a 10x10 cluster -> an 8x8 window holds at most 64
    y0, y1, x0, x1 = _best_box(mask, 8)
    assert (y1 - y0, x1 - x0) == (8, 8)
    assert int(mask[y0:y1, x0:x1].sum()) == 64


def test_best_box_is_not_the_centroid():
    # A U-shaped ring: the centroid of the mask sits in the empty middle, so a
    # centroid-centred crop would miss the structure entirely.
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:14] = True
    mask[10:50, 46:50] = True
    mask[46:50, 10:50] = True
    y0, y1, x0, x1 = _best_box(mask, 16)
    assert int(mask[y0:y1, x0:x1].sum()) > 0


def test_best_box_clamps_size_to_the_plane():
    # The window stays square, so it is clamped to the shorter side.
    mask = np.ones((5, 7), dtype=bool)
    y0, y1, x0, x1 = _best_box(mask, 32)
    assert (y1 - y0, x1 - x0) == (5, 5)
    assert 0 <= y0 < y1 <= 5 and 0 <= x0 < x1 <= 7


def test_best_box_tie_break_is_deterministic():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:4, 0:4] = True
    mask[16:20, 16:20] = True
    assert _best_box(mask, 4) == _best_box(mask, 4)


# --------------------------------------------------------------------------- select_rois


def _synthetic(shape=(20, 40, 40)):
    key = np.zeros(shape, dtype=np.float32)
    key[10, 10:20, 10:20] = 0.5   # a dense block -> the high band
    key[5, 5:15, 25:35] = 0.1     # a soft block  -> the low band
    return key.copy(), key


def test_select_rois_returns_three_crops_with_expected_axes():
    inp, key = _synthetic()
    rois = select_rois(inp, key, 0.2, 0.4, roi_size=8)
    assert [roi["key"] for roi in rois] == ["axial-bone", "axial-soft", "coronal-bone"]
    assert [roi["axis"] for roi in rois] == [0, 0, 1]


def test_select_rois_centres_the_bone_crop_on_the_bone():
    inp, key = _synthetic()
    rois = select_rois(inp, key, 0.2, 0.4, roi_size=8)
    bone = rois[0]
    assert bone["index"] == 10
    y0, y1, x0, x1 = bone["box"]
    assert int((key[10] >= 0.4)[y0:y1, x0:x1].sum()) == 64  # the full crop is bone


def test_select_rois_bone_slice_ignores_peripheral_bone():
    # Bone hugging the FOV rim (skull base) must lose to the central block: the
    # whole-plane argmax would otherwise pick the rim slice.
    key = np.zeros((20, 40, 40), dtype=np.float32)
    key[2, 0:2, :] = 0.5          # 80 rim voxels on a peripheral row
    key[10, 10:20, 10:20] = 0.5   # 100 voxels, inside the central 70 % box
    rois = select_rois(key.copy(), key, 0.2, 0.4, roi_size=8)
    assert rois[0]["index"] == 10


def test_select_rois_honours_explicit_slice_indices():
    inp, key = _synthetic()
    rois = select_rois(inp, key, 0.2, 0.4, roi_size=8, bone_z=3, soft_z=4)
    assert rois[0]["index"] == 3
    assert rois[1]["index"] == 4


def test_select_rois_labels_carry_the_slice_index():
    inp, key = _synthetic()
    rois = select_rois(inp, key, 0.2, 0.4, roi_size=8)
    assert f"z={rois[0]['index']}" in rois[0]["label"]
    assert f"y={rois[2]['index']}" in rois[2]["label"]


# --------------------------------------------------------------------------- metrics


def _tissue_input(shape=(8, 16, 16)):
    """One slice per band plus an air slice, so every band mask is populated."""
    inp = np.full(shape, 0.3, dtype=np.float32)
    inp[0] = 0.0  # air
    inp[1] = 0.1  # soft-tissue band
    inp[2] = 0.5  # bone band
    return inp


def _run_metrics(volume, lo=0.2, hi=0.4, max_slices=8):
    """Score *volume* on bands keyed on the input, exactly like the CLI does."""
    inp = _tissue_input()
    masks = band_masks(inp, lo, hi)
    tissue = inp > 0.01
    tissue_mean = float(inp[tissue].mean())
    row = _metrics(volume, masks, inp, 1.0, tissue, tissue_mean, max_slices)
    row["_input_bands"] = masks
    return row


def test_metrics_of_a_constant_volume_are_zero():
    row = _run_metrics(np.full((8, 16, 16), 0.3, dtype=np.float32))
    for band in ("low", "mid", "high"):
        assert row["bands"][band] == pytest.approx(0.0), band
    assert row["mean_offset_on_tissue"] == pytest.approx(0.0, abs=1e-6)
    # A zero soft-tissue sigma makes the ratio undefined; the guard returns NaN rather
    # than dividing by zero, matching denoise_multi_models.py.
    assert np.isnan(row["band_differentiation_high_over_low"])


def test_metrics_detect_noise_added_to_a_single_band():
    volume = np.full((8, 16, 16), 0.3, dtype=np.float32)
    volume[2, ::2, ::2] += 0.2  # checkerboard noise only inside the bone band
    row = _run_metrics(volume)
    assert row["bands"]["high"] > 0.0
    assert row["bands"]["low"] == pytest.approx(0.0)


def test_metrics_mean_offset_follows_a_global_brightness_shift():
    volume = np.full((8, 16, 16), 0.3, dtype=np.float32) + 0.02
    row = _run_metrics(volume)
    assert row["mean_offset_on_tissue"] == pytest.approx(0.02, abs=1e-6)


def test_band_differentiation_is_one_when_bands_are_equally_noisy():
    # A 3D checkerboard has the same Laplacian energy everywhere, so both bands
    # must score identically and the ratio must be exactly 1.
    volume = np.full((8, 16, 16), 0.3, dtype=np.float32)
    volume[:, ::2, ::2] = 0.5
    volume[:, 1::2, 1::2] = 0.5
    row = _run_metrics(volume)
    assert row["bands"]["low"] > 0.0
    assert row["band_differentiation_high_over_low"] == pytest.approx(1.0, rel=1e-9)


def test_cut_reports_band_reduction_against_the_input_bands():
    row = {
        "sigma_reduction_vs_input_pct": 40.0,
        "bands": {"low": 0.005, "mid": 0.0, "high": 0.010},
        "_input_bands": {"low": 0.010, "mid": 0.0, "high": 0.010},
    }
    assert _cut(row) == pytest.approx(40.0)
    assert _cut(row, "low") == pytest.approx(50.0)
    assert _cut(row, "high") == pytest.approx(0.0)


# --------------------------------------------------------------------------- io helpers


def test_load_rejects_a_short_file(tmp_path):
    path = tmp_path / "v.raw"
    path.write_bytes(b"\0" * 8)  # 2 float32, but (2, 2, 2) needs 8
    with pytest.raises(SystemExit, match="for shape"):
        _load(path, (2, 2, 2))


def test_load_reads_the_expected_shape(tmp_path):
    path = tmp_path / "v.raw"
    np.arange(8, dtype=DTYPE).tofile(path)
    assert _load(path, (2, 2, 2)).reshape(-1).tolist() == list(range(8))


def test_window_uses_percentiles():
    data = np.arange(1000, dtype=np.float32)
    low, high = _window(data, 5, 95)
    assert low == pytest.approx(np.percentile(data, 5))
    assert high == pytest.approx(np.percentile(data, 95))


def test_window_ignores_nan():
    data = np.array([np.nan, 0.0, 1.0, 2.0, np.nan], dtype=np.float32)
    low, high = _window(data, 0, 100)
    assert np.isfinite(low) and np.isfinite(high)


# --------------------------------------------------------------------------- markdown


def _fake_metrics():
    def row(low, mid, high, glob, cut, diff, offset):
        return {
            "global_sigma": glob, "sigma_reduction_vs_input_pct": cut,
            "band_differentiation_high_over_low": diff, "mean_offset_on_tissue": offset,
            "bands": {"low": low, "mid": mid, "high": high},
        }

    return {
        "input": row(0.01730, 0.01683, 0.01678, 0.01710, 0.0, 0.9700, 0.0),
        "conv": row(0.00551, 0.00534, 0.01059, 0.00621, 63.7, 1.9226, -1.5e-3),
        "r1b": row(0.00973, 0.00852, 0.00621, 0.00905, 47.1, 0.6386, -1.4e-3),
        "r0": row(0.00983, 0.00907, 0.00922, 0.00961, 43.8, 0.9386, -7e-4),
    }


def test_markdown_reports_measured_numbers_not_assumptions(tmp_path):
    from scripts.compare_conv_denoise import render_markdown

    metrics = _fake_metrics()
    out = tmp_path / "summary.md"
    render_markdown(metrics, ["input", "conv", "r1b", "r0"],
                    [{"key": "axial-bone", "label": "轴位 ROI · z=1", "box": (0, 8, 0, 8)}],
                    out, {"note": "note", "input": "full.bhc.raw", "input_md5": "abc",
                          "conv": "full.bhc.conv.raw", "conv_label": "conv",
                          "post_procs": "BHC ReduceNoise Anisotropic",
                          "pwls_scale": 0.01, "band_key": "r2b", "lo": 0.2, "hi": 0.4})
    text = out.read_text(encoding="utf-8")
    # The conventional result is inverted (soft tissue cut harder than bone), so the
    # generated prose must carry the measured diff instead of claiming "diff ~ 1".
    assert "1.923" in text
    assert "r1b" in text and "0.639" in text
    assert "+68.2%" in text or "68.2" in text
    assert "md5" in text
