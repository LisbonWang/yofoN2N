"""Tests for :class:`N2N3DDataset` (3D-input / 3D-output contract)."""

import json
import unittest.mock as mock

import numpy as np
import pytest

from cbct_n2n.data.n2n3d_dataset import N2N3DDataset

# H/W must be multiples of 32 (pad_to_multiple inside the dataset).
# 8 z-slices give enough headroom to exercise head overflow (start < 0),
# tail overflow (start + depth > count) and the identity-buffer layers.
SHAPE = (8, 32, 32)


@pytest.fixture(autouse=True)
def stub_memmap():
    """Replace ``np.memmap`` with ``np.fromfile`` for the test session.

    On Windows the developer sandbox restricts I/O against
    ``C:\\Users\\admin\\AppData\\Local\\Temp\\pytest-of-admin\\garbage-...``
    (the default location pytest's ``tmp_path`` fixture points at) and
    that makes the underlying ``mmap.mmap`` call return ``WinError 8``.
    :class:`N2NDataset` only *reads* the volumes, so a plain
    ``np.fromfile`` is a faithful substitute that avoids the mmap call.
    """
    real_memmap = np.memmap

    def fake_memmap(filename, dtype=None, mode="r", shape=None, **kwargs):
        # mirror numpy's read-only mode contract without touching mmap.
        if mode != "r":
            return real_memmap(filename, dtype=dtype, mode=mode, shape=shape, **kwargs)
        data = np.fromfile(str(filename), dtype=dtype)
        if shape is not None:
            data = data.reshape(shape)
        return data

    with mock.patch("numpy.memmap", side_effect=fake_memmap):
        yield


@pytest.fixture
def workspace(tmp_path):
    """Use pytest's ``tmp_path`` for the case file directory.

    The stubbed ``np.memmap`` above means we never open the file as a
    real memmap, so ``tmp_path``'s Windows location is fine for the
    plain file I/O the tests do.
    """
    yield tmp_path


def _base_volumes():
    """Build a tiny synthetic even/odd/full volume with z-aware values.

    Each ``z`` slice has constant intensity (0.10 for even, 0.30 for odd,
    0.20 for full) so any per-slice read can be predicted from the
    anchor z.  H/W are 32x32 of the same value -- enough to verify
    padding / reshape without dominating test runtime.
    """
    base_even = np.full(SHAPE, 0.10, dtype=np.float32)
    base_odd = np.full(SHAPE, 0.30, dtype=np.float32)
    base_full = np.full(SHAPE, 0.20, dtype=np.float32)
    # add a tiny per-z drift so anchor z can be told apart.
    z_drift = (np.arange(SHAPE[0], dtype=np.float32) * 1e-3).reshape(SHAPE[0], 1, 1)
    return {
        "even": base_even + z_drift,
        "odd": base_odd + z_drift,
        "full": base_full + z_drift,
    }


def _write_volumes(workspace, subdir, volumes):
    case_dir = workspace / subdir
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, volume in volumes.items():
        volume.astype("<f4").tofile(case_dir / f"{name}.raw")
    return case_dir


def _record(key, subdir):
    return {
        "key": key,
        "case_id": key,
        "reconstruction_status": "success",
        "shape": list(SHAPE),
        "dtype": "<f4",
        "outputs": {f"{name}.raw": f"{subdir}/{name}.raw" for name in ("even", "odd", "full")},
    }


def _write_manifest(workspace, records, split_keys, split="train"):
    manifest = {"records": records, "splits": {split: split_keys, "val": [], "test": []}}
    manifest_path = workspace / "reconstruction-index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _dataset(workspace, voxel_depth, identity_layers=0, mappings=("even->full",)):
    volumes = _base_volumes()
    _write_volumes(workspace, "case", volumes)
    record = _record("uid::case::0", "case")
    manifest_path = _write_manifest(workspace, [record], ["uid::case::0"])
    return N2N3DDataset(
        {"records": [record], "splits": {"train": ["uid::case::0"], "val": [], "test": []}},
        manifest_path, "train", voxel_depth=voxel_depth,
        identity_layers=identity_layers, mappings=list(mappings),
    )


# ---- construction validation ----------------------------------------------


def test_voxel_depth_must_be_positive(workspace):
    volumes = _base_volumes()
    _write_volumes(workspace, "case_pos", volumes)
    record = _record("uid::pos::0", "case_pos")
    manifest = {"records": [record], "splits": {"train": ["uid::pos::0"], "val": [], "test": []}}
    manifest_path = _write_manifest(workspace, [record], ["uid::pos::0"])
    with pytest.raises(ValueError, match="positive"):
        N2N3DDataset(manifest, manifest_path, "train", voxel_depth=0, mappings=["even->full"])


def test_identity_layers_must_be_less_than_voxel_depth(workspace):
    volumes = _base_volumes()
    _write_volumes(workspace, "case_idtv", volumes)
    record = _record("uid::idtv::0", "case_idtv")
    manifest = {"records": [record], "splits": {"train": ["uid::idtv::0"], "val": [], "test": []}}
    manifest_path = _write_manifest(workspace, [record], ["uid::idtv::0"])
    with pytest.raises(ValueError, match="identity_layers"):
        N2N3DDataset(manifest, manifest_path, "train", voxel_depth=5,
                     identity_layers=5, mappings=["even->full"])
    with pytest.raises(ValueError, match="identity_layers"):
        N2N3DDataset(manifest, manifest_path, "train", voxel_depth=5,
                     identity_layers=-1, mappings=["even->full"])


def test_accepts_even_voxel_depth(workspace):
    """Even depths are now legal: there is no centre-slice target anymore."""
    dataset = _dataset(workspace, voxel_depth=4)
    item = dataset[0]
    assert item["source"].shape[1] == 4  # D axis is 4 (even) slices
    assert item["target"].shape[1] == 4


# ---- start-index enumeration ----------------------------------------------


def test_start_index_enumeration(workspace):
    """Windows are keyed by start index: -identity_layers, stride 2, <= count-3.

    With count=8, identity_layers=2 the starts are {-2, 0, 2, 4} (4 samples
    for a single even->full mapping).
    """
    dataset = _dataset(workspace, voxel_depth=4, identity_layers=2)
    starts = sorted(int(dataset[i]["slice_index"]) for i in range(len(dataset)))
    assert starts == [-2, 0, 2, 4]
    assert len(dataset) == 4


def test_start_index_enumeration_no_identity(workspace):
    """identity_layers=0 -> start begins at 0 (no head overflow at all)."""
    dataset = _dataset(workspace, voxel_depth=4, identity_layers=0)
    starts = sorted(int(dataset[i]["slice_index"]) for i in range(len(dataset)))
    assert starts == [0, 2, 4]
    assert len(dataset) == 3


# ---- forward / target shape ------------------------------------------------


def test_returns_5d_source_and_5d_target(workspace):
    dataset = _dataset(workspace, voxel_depth=4)
    item = dataset[0]
    # source and target are both 5D [1, D, H, W] over the same window.
    assert item["source"].shape == (1, 4, SHAPE[1], SHAPE[2])
    assert item["target"].shape == (1, 4, SHAPE[1], SHAPE[2])
    # Sample metadata stays compatible with default_collate.
    assert isinstance(item["residual"], str)


def test_middle_window_maps_source_to_target(workspace):
    """A fully in-range window maps each source slice to its paired target
    (no centre-slice collapse): source[d] = even[start+d], target[d] = full[start+d].
    """
    volumes = _base_volumes()
    _write_volumes(workspace, "case_slab", volumes)
    record = _record("uid::slab::0", "case_slab")
    manifest_path = _write_manifest(workspace, [record], ["uid::slab::0"])
    dataset = N2N3DDataset(
        {"records": [record], "splits": {"train": ["uid::slab::0"], "val": [], "test": []}},
        manifest_path, "train", voxel_depth=4, identity_layers=0, mappings=["even->full"],
    )
    item = dataset[1]  # start = 2 -> window z = 2,3,4,5 (all in range)
    assert int(item["slice_index"]) == 2
    for d in range(4):
        z = 2 + d
        np.testing.assert_allclose(item["source"].numpy()[0, d, :, :],
                                   volumes["even"][z], atol=1e-5)
        np.testing.assert_allclose(item["target"].numpy()[0, d, :, :],
                                   volumes["full"][z], atol=1e-5)


# ---- boundary handling -----------------------------------------------------


def test_head_overflow_zero_fills(workspace):
    """start=-2 with identity_layers=2: the two leading window slices (z=-2,-1)
    fall before slice 0 and must be zero-filled for both source and target.
    """
    volumes = _base_volumes()
    _write_volumes(workspace, "case_zero", volumes)
    record = _record("uid::zero::0", "case_zero")
    manifest_path = _write_manifest(workspace, [record], ["uid::zero::0"])
    dataset = N2N3DDataset(
        {"records": [record], "splits": {"train": ["uid::zero::0"], "val": [], "test": []}},
        manifest_path, "train", voxel_depth=4, identity_layers=2, mappings=["even->full"],
    )
    item = dataset[0]  # start = -2 -> z = -2,-1 (zero), 0,1 (in range)
    assert int(item["slice_index"]) == -2
    zeros = np.zeros((SHAPE[1], SHAPE[2]), dtype=np.float32)
    src = item["source"].numpy()
    tgt = item["target"].numpy()
    # z=-2,-1 zero-filled (and these are also the identity-buffer positions).
    np.testing.assert_allclose(src[0, 0], zeros, atol=1e-5)
    np.testing.assert_allclose(src[0, 1], zeros, atol=1e-5)
    np.testing.assert_allclose(tgt[0, 0], zeros, atol=1e-5)
    np.testing.assert_allclose(tgt[0, 1], zeros, atol=1e-5)
    # z=0,1 are in-range: source=even, target=full (index >= identity_layers).
    np.testing.assert_allclose(src[0, 2], volumes["even"][0], atol=1e-5)
    np.testing.assert_allclose(tgt[0, 2], volumes["full"][0], atol=1e-5)
    np.testing.assert_allclose(src[0, 3], volumes["even"][1], atol=1e-5)
    np.testing.assert_allclose(tgt[0, 3], volumes["full"][1], atol=1e-5)


def test_tail_overflow_replicates_boundary(workspace):
    """A window that sticks past the last slice replicates the boundary slice
    (start=4 with depth=5 covers z=4..8; z=8 -> replicate z=7).
    """
    volumes = _base_volumes()
    _write_volumes(workspace, "case_tail", volumes)
    record = _record("uid::tail::0", "case_tail")
    manifest_path = _write_manifest(workspace, [record], ["uid::tail::0"])
    dataset = N2N3DDataset(
        {"records": [record], "splits": {"train": ["uid::tail::0"], "val": [], "test": []}},
        manifest_path, "train", voxel_depth=5, identity_layers=0, mappings=["even->full"],
    )
    # start=4 -> z = 4,5,6,7 (in range), 8 (-> replicate z=7)
    item = dataset[2]  # starts: 0,2,4 -> index 2 is start=4
    assert int(item["slice_index"]) == 4
    src = item["source"].numpy()
    tgt = item["target"].numpy()
    for d in range(4):
        z = 4 + d
        np.testing.assert_allclose(src[0, d], volumes["even"][z], atol=1e-5)
        np.testing.assert_allclose(tgt[0, d], volumes["full"][z], atol=1e-5)
    # last slice (z=8) replicates the boundary slice z=7.
    np.testing.assert_allclose(src[0, 4], volumes["even"][7], atol=1e-5)
    np.testing.assert_allclose(tgt[0, 4], volumes["full"][7], atol=1e-5)


def test_identity_layers_mirror_source(workspace):
    """With identity_layers=2 the leading two in-range window slices have
    target==source (identity / continuity constraint); the rest use the paired
    target volume.  start=0 -> z=0,1 (identity), 2,3 (normal).
    """
    volumes = _base_volumes()
    _write_volumes(workspace, "case_idt", volumes)
    record = _record("uid::idt::0", "case_idt")
    manifest_path = _write_manifest(workspace, [record], ["uid::idt::0"])
    dataset = N2N3DDataset(
        {"records": [record], "splits": {"train": ["uid::idt::0"], "val": [], "test": []}},
        manifest_path, "train", voxel_depth=4, identity_layers=2, mappings=["even->full"],
    )
    item = dataset[1]  # starts: -2,0,2,4 -> index 1 is start=0
    assert int(item["slice_index"]) == 0
    src = item["source"].numpy()
    tgt = item["target"].numpy()
    # identity layers (index 0,1 -> z=0,1): target == source == even
    for d, z in ((0, 0), (1, 1)):
        np.testing.assert_allclose(src[0, d], volumes["even"][z], atol=1e-5)
        np.testing.assert_allclose(tgt[0, d], volumes["even"][z], atol=1e-5)
    # normal supervision (index 2,3 -> z=2,3): source=even, target=full
    for d, z in ((2, 2), (3, 3)):
        np.testing.assert_allclose(src[0, d], volumes["even"][z], atol=1e-5)
        np.testing.assert_allclose(tgt[0, d], volumes["full"][z], atol=1e-5)


# ---- metadata preservation -------------------------------------------------


def test_metadata_keys_present_and_str_compatible(workspace):
    dataset = _dataset(workspace, voxel_depth=4)
    item = dataset[0]
    for key in ("source", "target", "case_id", "slice_index", "source_kind",
                "target_kind", "mapping", "residual"):
        assert key in item
    assert item["source_kind"] == "even"
    assert item["target_kind"] == "full"
    assert item["mapping"] == "even->full"
    assert item["residual"] == ""  # base sample, empty-string sentinel
