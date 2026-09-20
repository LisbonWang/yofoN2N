from pathlib import Path
import struct

import pytest

from cbct_n2n.reconstruction.subset import (
    build_subset_workspace,
    numbered_raw_files,
    split_projection_indices,
    split_projection_indices_blockwise,
)


def _case(tmp_path: Path, count: int = 5):
    raw = tmp_path / "raw"
    cfg = tmp_path / "cfg"
    raw.mkdir()
    cfg.mkdir()
    for i in reversed(range(count)):
        (raw / f"{i:05d}.raw").write_bytes(bytes([i]))
    (cfg / "fdkPara.xml").write_text(f"<FDK_sys_para><noView>{count}</noView></FDK_sys_para>", encoding="utf-8")
    (cfg / "filter.xml").write_text("<FDK_filter_config/>", encoding="utf-8")
    (cfg / "projectionMatrix.bin").write_bytes(b"".join(struct.pack("<21f", *([i] * 21)) for i in range(count)))
    (cfg / "notes.txt").write_text("keep", encoding="utf-8")
    return raw, cfg


def test_numbered_files_sort_numerically_and_split_positions(tmp_path):
    raw, _ = _case(tmp_path, 5)
    assert [p.name for p in numbered_raw_files(raw)] == ["00000.raw", "00001.raw", "00002.raw", "00003.raw", "00004.raw"]
    assert split_projection_indices(5) == ([0, 2, 4], [1, 3])
    assert split_projection_indices(4) == ([0, 2], [1, 3])


def test_blockwise_split_takes_two_then_skips_two():
    assert split_projection_indices_blockwise(8, block_size=2) == (
        [0, 1, 4, 5],
        [2, 3, 6, 7],
    )
    assert split_projection_indices_blockwise(10, block_size=2) == (
        [0, 1, 4, 5, 8, 9],
        [2, 3, 6, 7],
    )


def test_subset_copies_projection_and_per_view_geometry(tmp_path):
    raw, cfg = _case(tmp_path, 5)
    out = tmp_path / "subset"
    result = build_subset_workspace(raw, cfg, [1, 3], out, link_mode="copy")
    assert result.view_count == 2
    assert [p.name for p in result.raw_files] == ["00000.raw", "00001.raw"]
    assert (out / "config" / "fdkPara.xml").read_text(encoding="utf-8").find("<noView>2</noView>") >= 0
    values = struct.unpack("<42f", (out / "config" / "projectionMatrix.bin").read_bytes())
    assert values[:21] == (1.0,) * 21
    assert values[21:] == (3.0,) * 21


def test_subset_slices_explicit_per_view_weight_and_fit_data(tmp_path):
    raw, cfg = _case(tmp_path, 4)
    (cfg / "weightR.bin").write_bytes(b"".join(struct.pack("<f", i + 0.25) for i in range(4)))
    (cfg / "fitData.bin").write_bytes(b"".join(struct.pack("<2f", i, i + 100) for i in range(4)))
    out = tmp_path / "subset"
    build_subset_workspace(raw, cfg, [1, 3], out, link_mode="copy", per_view_files={"projectionMatrix.bin": 84, "weightR.bin": 4, "fitData.bin": 8})
    assert struct.unpack("<2f", (out / "config" / "weightR.bin").read_bytes()) == (1.25, 3.25)
    assert struct.unpack("<4f", (out / "config" / "fitData.bin").read_bytes()) == (1.0, 101.0, 3.0, 103.0)


def test_subset_empty_per_view_mapping_copies_binaries_whole(tmp_path):
    """An explicit empty per-view mapping must not fall back to slicing.

    Mode1 stores projectionMatrix.bin as a shared whole file whose size does not match
    nviews x 84; passing ``per_view_files={}`` must copy it intact instead of slicing.
    """
    raw, cfg = _case(tmp_path, 5)
    (cfg / "projectionMatrix.bin").write_bytes(b"shared-matrix-data")  # size != 5 x 84
    out = tmp_path / "subset"
    result = build_subset_workspace(raw, cfg, [1, 3], out, link_mode="copy", per_view_files={})
    assert (out / "config" / "projectionMatrix.bin").read_bytes() == b"shared-matrix-data"
    assert result.view_count == 2


def test_subset_rejects_invalid_inputs(tmp_path):
    raw, cfg = _case(tmp_path, 3)
    with pytest.raises(ValueError):
        split_projection_indices(0)
    with pytest.raises(ValueError):
        build_subset_workspace(raw, cfg, [], tmp_path / "empty")
    (cfg / "projectionMatrix.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="projectionMatrix.bin"):
        build_subset_workspace(raw, cfg, [0], tmp_path / "bad", link_mode="copy")
