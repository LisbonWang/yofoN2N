"""Unit tests for the conventional-denoise variant zoom (no GPU, no SDK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.compare_conv_zoom import _label, _named_path, _parse_args, main

SHAPE = (8, 32, 32)


def _write_volume(path, *, bone: float = 0.6, scale: float = 1.0) -> None:
    volume = np.full(SHAPE, 0.3, dtype=np.float32)
    volume[:, :8, :] = 0.0                      # air in front of the jaw
    volume[2:6, 8:16, 8:16] = bone              # a bone blob
    path.write_bytes((volume * scale).astype("<f4").tobytes())


def test_named_path_splits_on_the_first_equals():
    assert _named_path("mine=a/b.raw") == ("mine", Path("a/b.raw"))


def test_named_path_rejects_a_missing_path():
    with pytest.raises(Exception):
        _named_path("mine")


def test_label_strips_the_reference_prefix_and_suffix():
    assert _label("full.bhc.conv_rn_ani.raw") == "conv_rn_ani"
    assert _label("full.bhc.conv_rn_ani.pwls600.raw") == "conv_rn_ani.pwls600"


def test_label_keeps_the_whole_name_when_it_is_already_the_stem():
    """The undenoised reference never reaches _label -- main() gives it its own caption."""
    assert _label("conv_rn_ani.raw", prefix="") == "conv_rn_ani"


def test_zoom_main_renders_every_globbed_variant(tmp_path):
    _write_volume(tmp_path / "full.bhc.raw")
    _write_volume(tmp_path / "full.bhc.conv_rn_ani.raw", bone=0.55)
    _write_volume(tmp_path / "full.bhc.conv_rn_ani.pwls600.raw", bone=0.5)
    out = tmp_path / "figures" / "zoom.png"

    rc = main(["--dir", str(tmp_path), "--shape", *(str(v) for v in SHAPE),
               "--roi-size", "16", "--out", str(out)])

    assert rc == 0
    assert out.is_file()


def test_zoom_main_rejects_a_volume_whose_size_contradicts_shape(tmp_path):
    _write_volume(tmp_path / "full.bhc.raw")
    _write_volume(tmp_path / "full.bhc.conv_rn_ani.raw")
    (tmp_path / "full.bhc.conv_rn_ani.pwls600.raw").write_bytes(b"\0" * 1024)

    with pytest.raises(SystemExit, match="implies"):
        main(["--dir", str(tmp_path), "--shape", *(str(v) for v in SHAPE),
              "--roi-size", "16", "--out", str(tmp_path / "zoom.png")])


def test_zoom_main_needs_something_to_compare(tmp_path):
    _write_volume(tmp_path / "full.bhc.raw")

    with pytest.raises(SystemExit, match="nothing to compare"):
        main(["--dir", str(tmp_path), "--shape", *(str(v) for v in SHAPE),
              "--roi-size", "16", "--out", str(tmp_path / "zoom.png")])


def test_parse_args_requires_dir_shape_and_out():
    with pytest.raises(SystemExit):
        _parse_args(["--dir", "x"])
