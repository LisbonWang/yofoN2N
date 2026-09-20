"""Unit tests for scripts/denoise_test_set.py (test-set staging + denoising)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.denoise_test_set import (
    copy_input,
    denoised_name,
    expected_bytes,
    main,
    volume_stats,
)


def test_denoised_name_appends_suffix_before_extension():
    assert denoised_name("full.raw") == "full_denoised.raw"
    assert denoised_name("full.bhc.raw") == "full.bhc_denoised.raw"
    assert denoised_name("full.marfi.bhc.raw") == "full.marfi.bhc_denoised.raw"


def test_expected_bytes_matches_float32_volume():
    assert expected_bytes((32, 64, 64)) == 32 * 64 * 64 * 4


def test_copy_input_is_idempotent_and_verifies_shape(tmp_path):
    shape = (4, 8, 8)
    source = tmp_path / "full.raw"
    np.arange(int(np.prod(shape)), dtype="<f4").tofile(source)
    destination = tmp_path / "staged" / "full.raw"

    assert copy_input(source, destination, shape) == "copied"
    assert destination.stat().st_size == expected_bytes(shape)
    # second call must not re-copy
    assert copy_input(source, destination, shape) == "cached"


def test_copy_input_rejects_shape_mismatch(tmp_path):
    source = tmp_path / "full.raw"
    np.zeros(10, dtype="<f4").tofile(source)
    with pytest.raises(ValueError, match="manifest and on-disk volume disagree"):
        copy_input(source, tmp_path / "out.raw", (4, 8, 8))


def test_copy_input_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing source volume"):
        copy_input(tmp_path / "gone.raw", tmp_path / "out.raw", (4, 8, 8))


def test_volume_stats_reports_finite_range_and_noise():
    volume = np.random.default_rng(0).normal(0.1, 0.01, size=(16, 32, 32)).astype("<f4")
    stats = volume_stats(volume)
    assert stats["finite"] is True
    assert stats["min"] < stats["mean"] < stats["max"]
    assert stats["sigma_immerkaer"] > 0


def _write_volume(path, shape, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (rng.random(shape, dtype=np.float32) * 0.2 + 0.05).astype("<f4")
    data.tofile(path)


def _write_checkpoint(path):
    from cbct_n2n.models import UNet3D

    model = UNet3D(in_channels=1, out_channels=1, base_channels=4, depth=2, residual=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0}, path)


def _manifest(path, shape, keys):
    records = []
    for key, case_id in keys:
        relative = f"PDL/{case_id}/{key}/full.raw"
        records.append({
            "key": f"{key}::Mode2_1609",
            "group": "Mode2_1609",
            "case_id": case_id,
            "study_uid": key,
            "shape": list(shape),
            "outputs": {"full.raw": relative},
        })
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def test_main_end_to_end_with_tiny_model(tmp_path, monkeypatch):
    shape = (16, 64, 64)
    keys = [("uid-a", "DEV-A"), ("uid-b", "DEV-B")]
    manifest = tmp_path / "splits.json"
    _manifest(manifest, shape, keys)
    root = tmp_path / "root"
    for seed, (key, case_id) in enumerate(keys):
        _write_volume(root / f"PDL/{case_id}/{key}/full.raw", shape, seed)
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    out_root = tmp_path / "out"

    monkeypatch.setattr("sys.argv", [
        "denoise_test_set.py",
        "--manifest", str(manifest),
        "--root", str(root),
        "--out-root", str(out_root),
        "--checkpoint", str(checkpoint),
        "--domains", "full.raw",
        "--base-channels", "4",
        "--depth", "2",
        "--voxel-depth", "16",
        "--patch-hw", "32",
        "--overlap-d", "4",
        "--overlap-hw", "8",
        "--batch-size", "1",
        "--device", "cpu",
    ])

    assert main() == 0

    for key, case_id in keys:
        staged = out_root / "inputs" / f"PDL/{case_id}/{key}/full.raw"
        result = out_root / "denoised" / f"PDL/{case_id}/{key}/full_denoised.raw"
        assert staged.exists() and staged.stat().st_size == expected_bytes(shape)
        assert result.exists() and result.stat().st_size == expected_bytes(shape)
        # staged input must be byte-identical to the source
        assert staged.read_bytes() == (root / f"PDL/{case_id}/{key}/full.raw").read_bytes()
        # prediction must be finite and preserve the volume shape
        prediction = np.fromfile(result, dtype="<f4").reshape(shape)
        assert np.isfinite(prediction).all()

    report = json.loads((out_root / "denoise_summary.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in report["records"]] == ["done", "done"]
    assert report["records"][0]["bytes"] == expected_bytes(shape)
    assert all(entry["denoised"]["finite"] for entry in report["records"])
    assert report["tiled"]["voxel_depth"] == 16


def test_main_copy_only_skips_inference(tmp_path, monkeypatch):
    shape = (16, 64, 64)
    keys = [("uid-a", "DEV-A")]
    manifest = tmp_path / "splits.json"
    _manifest(manifest, shape, keys)
    root = tmp_path / "root"
    _write_volume(root / "PDL/DEV-A/uid-a/full.raw", shape, 0)
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    out_root = tmp_path / "out"

    monkeypatch.setattr("sys.argv", [
        "denoise_test_set.py",
        "--manifest", str(manifest),
        "--root", str(root),
        "--out-root", str(out_root),
        "--checkpoint", str(checkpoint),
        "--domains", "full.raw",
        "--copy-only", "--device", "cpu",
    ])

    assert main() == 0
    assert (out_root / "inputs" / "PDL/DEV-A/uid-a/full.raw").exists()
    assert not (out_root / "denoised").exists()
    assert not (out_root / "denoise_summary.json").exists()


def test_main_rejects_missing_domain_entry(tmp_path, monkeypatch):
    shape = (16, 64, 64)
    manifest = tmp_path / "splits.json"
    _manifest(manifest, shape, [("uid-a", "DEV-A")])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    monkeypatch.setattr("sys.argv", [
        "denoise_test_set.py",
        "--manifest", str(manifest),
        "--root", str(tmp_path / "root"),
        "--out-root", str(tmp_path / "out"),
        "--checkpoint", str(checkpoint),
        "--domains", "full.bhc.marfi.raw",
        "--device", "cpu",
    ])

    with pytest.raises(KeyError, match="no output entry"):
        main()
