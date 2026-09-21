"""Tests for the strength A/B script (no GPU, no SDK).

``guided_sharpen_compare.py`` exists so a strength sweep costs one guided-filter
run instead of one 590 MB volume per point.  That shortcut rests on a single
property -- the field is strength-independent, so ``plain + s * field`` cropped
is bit-identical to the full volume the volume script writes -- and on path
flags behaving like every other script in the repo.  Both are pinned here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cbct_n2n.filters.guided import guided_sharpen_3d, guided_structure_field
from scripts.guided_sharpen_compare import _CropMask, main, parse_verify, sharpen_patch

SHAPE = (12, 64, 64)
SL = (slice(10, 16), slice(28, 36), slice(28, 36))  # cropping is a plain tuple


def test_parse_verify_splits_on_the_first_equals():
    parsed = parse_verify(["1.0=a/b.raw", "0.6=c.raw"])
    assert set(parsed) == {1.0, 0.6}
    assert parsed[1.0] == Path("a/b.raw")
    assert parsed[0.6] == Path("c.raw")


def test_parse_verify_rejects_a_missing_equals():
    with pytest.raises(SystemExit):
        parse_verify(["1.0"])


def _volume(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.full(SHAPE, 0.30, dtype=np.float32)
    vol[:, :3, :] = 0.0
    vol += rng.normal(0.0, 0.002, size=SHAPE).astype(np.float32)
    vol[4:9, 20:44, 20:44] += np.linspace(-0.05, 0.25, 24, dtype=np.float32)[None, :, None]
    return vol


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 2.0])
def test_crop_of_the_field_reproduces_the_written_volume(strength):
    """The whole point of the script: cropping commutes with the combine.

    Compared against ``guided_sharpen_3d`` -- the function that actually writes
    the delivered RAW -- so a drift here means the figure would show a strength
    that is not the one on disk.
    """
    vol = _volume()
    eps = 1e-5
    field = guided_structure_field(vol, radius=2, eps=eps, mode="wiener")
    written = guided_sharpen_3d(vol, radius=2, eps=eps, strength=strength, mode="wiener")

    mine = sharpen_patch(vol, field, SL, strength)
    assert np.array_equal(mine, np.asarray(written[SL], dtype=np.float32))
    assert mine.dtype == np.dtype("float32")


def test_zero_strength_is_the_identity_of_the_crop():
    vol = _volume()
    field = guided_structure_field(vol, radius=2, eps=1e-5, mode="wiener")
    assert np.array_equal(sharpen_patch(vol, field, SL, 0.0), vol[SL])


def test_lazy_panel_matches_the_eager_patch():
    """``_CropMask`` is what keeps the script from allocating full volumes."""
    vol = _volume()
    field = guided_structure_field(vol, radius=2, eps=1e-5, mode="wiener")
    lazy = _CropMask(vol, field, 0.6)
    assert np.array_equal(lazy[SL], sharpen_patch(vol, field, SL, 0.6))


def test_all_path_flags_land_under_dir_including_noisy(tmp_path, monkeypatch):
    """``--noisy`` was a plain string, so ``_under`` raised AttributeError.

    Also a CWD-purity check: the figures/sidecars must not appear next to the
    process, only under ``--dir``.
    """
    import json

    from tests.filters.test_guided_sharpen_cli import _write_roi_cache, _write_volume

    data = tmp_path / "data"
    data.mkdir()
    _write_volume(data / "full.bhc.r4b.raw", SHAPE)
    _write_volume(data / "noisy.raw", SHAPE)
    _write_roi_cache(data / "rois.json", flat_size=16, bone_size=12)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    monkeypatch.setattr(
        "sys.argv",
        [
            "guided_sharpen_compare.py",
            "--dir", str(data),
            "--input", "full.bhc.r4b.raw",
            "--noisy", "noisy.raw",
            "--shape", *[str(v) for v in SHAPE],
            "--radius", "2", "--eps-k", "1", "--strength", "0.5", "1.0",
            "--sigma", "0.002", "--slab", "8",
            "--roi-cache", "rois.json",
            "--figure", "cmp.png",
            "--curve", "cmp_curve.png",
            "--json", "cmp.json",
            "--markdown", "cmp.md",
        ],
    )
    assert main() == 0

    for name in ("cmp.png", "cmp_curve.png", "cmp.json", "cmp.md"):
        assert (data / name).is_file(), name
    assert list(elsewhere.iterdir()) == []

    payload = json.loads((data / "cmp.json").read_text(encoding="utf-8"))
    assert payload["meta"]["eps_k"] == 1.0
    assert payload["meta"]["strengths"] == [0.5, 1.0]
    # The noisy reference must be present, otherwise the figure silently loses
    # the row that says "this is what undenoised noise looks like".
    assert np.isfinite(payload["meta"]["noisy_bone"])
