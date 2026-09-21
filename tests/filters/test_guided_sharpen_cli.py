"""CLI-level tests for the guided-sharpening script (no GPU, no SDK).

The script itself is the deliverable, not just the filter library, so its two
easy-to-get-wrong non-numeric behaviours are pinned here:

* path resolution -- every path flag is relative to ``--dir`` unless absolute.
  Before this was fixed, ``--input``/``--guide`` used ``--dir`` while
  ``--out``/``--json``/``--markdown``/``--figures`` used the process CWD, which
  silently scattered a 1.1 GB output volume and the sidecars into the repo root
  when the flags were given as bare names.
* the tissue mask used for the overshoot QC must come from the *input* volume.
  Deriving it per-volume instead is self-selecting: a voxel pushed below the
  floor stops counting as tissue exactly when it should be counted, which hid
  the whole undershoot tail.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.guided_sharpen_volume import (
    _parse_named,
    _under,
    main,
    overshoot_stats,
)

SHAPE = (12, 64, 64)


def test_relative_paths_are_resolved_under_dir():
    assert _under(Path("out"), Path("a/b.raw")) == Path("out/a/b.raw")
    assert _under(Path("out"), Path("figs")) == Path("out/figs")


def test_absolute_paths_are_left_alone():
    assert _under(Path("out"), Path("C:/tmp/x.raw")) == Path("C:/tmp/x.raw")


def test_missing_flag_stays_none_so_defaults_apply():
    assert _under(Path("out"), None) is None


def test_named_reference_splits_on_the_first_equals():
    assert _parse_named("noisy=a/b.raw") == ("noisy", Path("a/b.raw"))


def test_overshoot_counts_a_voxel_that_falls_below_the_floor():
    """The mask must be the *input's*; a voxel driven negative must still count.

    Two tissue voxels in the input; sharpening drives one of them to -0.5.
    Re-deriving the mask from the sharpened volume drops that voxel -- it is no
    longer above the floor, which is exactly when it matters -- and reports a
    clean zero undershoot.  That is the bias this function is written to avoid.
    """
    plain = np.zeros((4, 4, 4), dtype=np.float32)
    plain[0, 0, 0] = 0.02
    plain[0, 0, 1] = 0.03
    tissue = plain > 0.01                     # the input's mask: 2 voxels

    driven = plain.copy()
    driven[0, 0, 0] = -0.5

    from_input_mask = overshoot_stats(driven, tissue)
    assert from_input_mask["tissue_n"] == 2
    assert from_input_mask["tissue_min"] == pytest.approx(-0.5)
    assert from_input_mask["tissue_below_0_pct"] == pytest.approx(50.0)

    rederived = overshoot_stats(driven, driven > 0.01)
    assert rederived["tissue_n"] == 1         # the below-floor voxel vanished
    assert rederived["tissue_below_0_pct"] == 0.0

    # An empty mask reports NaN rather than omitting the key.
    assert np.isnan(overshoot_stats(driven, np.zeros_like(tissue))["tissue_min"])


def test_visible_darkening_ignores_voxels_that_were_already_black():
    """The metric that can actually see a dark rim ("黑边").

    ``tissue_below_0_pct`` counts a voxel sitting at the tissue floor after a
    0.01 fall, so it is dominated by voxels that are black in any display window.
    On the real volume lowering the dark side moved that metric 0.82% -> 0.69%
    while the *visible* darkening stayed at 0.53 -- this test pins the difference
    on a case small enough to reason about.
    """
    plain = np.zeros((4, 4, 4), dtype=np.float32)
    plain[0, 0, 0] = 0.02                      # at the floor: already black
    plain[1, 1, 1] = 0.30                      # unambiguously tissue
    tissue = plain > 0.01

    driven = plain.copy()
    driven[0, 0, 0] = -0.30                    # huge fall, invisible
    stats = overshoot_stats(driven, tissue, plain)
    assert stats["tissue_below_0_pct"] == pytest.approx(50.0)   # 1 of 2 tissue voxels
    assert stats["darken_max"] == pytest.approx(0.32)
    assert stats["darken_visible_max"] == 0.0                   # nothing visible moved

    driven[1, 1, 1] = 0.10                     # now a voxel that *is* visible
    stats = overshoot_stats(driven, tissue, plain)
    assert stats["darken_visible_max"] == pytest.approx(0.20)
    assert stats["darken_visible_p999"] == pytest.approx(0.20)


def test_darkening_keys_are_nan_without_a_reference():
    """A consumer must never read "no reference" as "no darkening"."""
    volume = np.full((4, 4, 4), 0.3, dtype=np.float32)
    stats = overshoot_stats(volume, volume > 0.01)
    assert np.isnan(stats["darken_max"])
    assert np.isnan(stats["darken_visible_max"])


def _write_volume(path, shape=SHAPE) -> None:
    """A tiny volume with air, soft tissue and one higher-density blob.

    Noise is needed in the tissue: with ``--sigma-source flat`` a perfectly
    constant flat patch measures ``sigma = 0`` and ``eps`` would be zero.
    """
    rng = np.random.default_rng(0)
    vol = np.full(shape, 0.30, dtype=np.float32)
    vol[:, :8, :] = 0.0                                        # air in front
    vol += rng.normal(0.0, 0.002, size=shape).astype(np.float32)
    ramp = np.linspace(-0.05, 0.25, 24, dtype=np.float32)
    vol[4:8, 20:44, 20:44] += ramp[None, :, None]              # a texture blob
    path.write_bytes(vol.astype("<f4").tobytes())


def _write_roi_cache(path, *, flat_size=16, bone_size=12) -> None:
    """Bypass ROI auto-discovery.

    ``--flat-roi``/``--bone-roi`` only *override* a discovered ROI, so they do
    not help on a synthetic volume; a cache is read instead of searching.  The
    cache must carry ``size``/``z``/``y``/``x`` plus the ``window_mean`` and
    ``local_std`` the config log line prints.
    """
    path.write_text(
        json.dumps(
            {
                "flat": {"z": 6, "y": 32, "x": 32, "size": flat_size,
                         "window_mean": 0.30, "local_std": 0.002,
                         "label": "test flat"},
                "bone": {"z": 6, "y": 32, "x": 32, "size": bone_size,
                         "window_mean": 0.30, "local_std": 0.05,
                         "label": "test bone"},
            }
        ),
        encoding="utf-8",
    )


def test_bare_output_flags_land_under_dir_not_the_cwd(tmp_path, monkeypatch):
    """The regression: bare flag values must not be written to the CWD."""
    data = tmp_path / "data"
    data.mkdir()
    _write_volume(data / "full.bhc.r4b.raw")
    _write_roi_cache(data / "rois.json")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)             # nothing may be created here

    monkeypatch.setattr(
        "sys.argv",
        [
            "guided_sharpen_volume.py",
            "--dir", str(data),
            "--input", "full.bhc.r4b.raw",
            "--shape", *[str(v) for v in SHAPE],
            "--radius", "2", "--eps-k", "1", "--strength", "1.0",
            "--sigma", "0.002",
            "--roi-cache", "rois.json",
            "--out", "full.bhc.r4b.gfsharp.raw",
            "--json", "gfsharp.json",
            "--markdown", "gfsharp_summary.md",
            "--figures", "figures_gf",
        ],
    )
    assert main() == 0

    assert (data / "full.bhc.r4b.gfsharp.raw").is_file()
    assert (data / "gfsharp.json").is_file()
    assert (data / "gfsharp_summary.md").is_file()
    assert (data / "figures_gf" / "gfsharp_metrics.png").is_file()
    assert list(elsewhere.iterdir()) == []
