import json
from types import SimpleNamespace

import numpy as np

from cbct_n2n.reconstruction.discovery import ScanCase
from cbct_n2n.reconstruction.prepare import PrepareOptions, prepare_case


def test_prepare_case_writes_matching_pair_atomically(tmp_path):
    raw = tmp_path / "case" / "scan_Mode2_100kV8mA1cc_S4"
    cfg = tmp_path / "case" / "config" / "Mode2"
    raw.mkdir(parents=True)
    cfg.mkdir(parents=True)
    for i in range(4):
        (raw / f"{i:05d}.raw").write_bytes(bytes([i]))
    (cfg / "filter.xml").write_text("<x/>")
    (cfg / "fdkPara.xml").write_text("<FDK_sys_para><noView>4</noView></FDK_sys_para>")
    np.arange(4 * 21, dtype=np.float32).tofile(cfg / "projectionMatrix.bin")
    np.arange(4, dtype=np.float32).tofile(cfg / "weightR.bin")
    calls = []

    def reconstruct(raw_dir, cfg_dir, kv, ma, xday_install_dir, post_procs, **kwargs):
        calls.append((raw_dir, cfg_dir, kv, ma, np.fromfile(cfg_dir / "weightR.bin", dtype=np.float32).tolist()))
        value = float(len(calls))
        return SimpleNamespace(
            volume=np.full((2, 3, 4), value, dtype=np.float32),
            volume_info=SimpleNamespace(z_spacing=0.3, y_spacing=0.2, x_spacing=0.2),
        )

    record = prepare_case(
        ScanCase("case", raw, cfg, "Mode2"),
        tmp_path / "out",
        PrepareOptions(xday_dir=tmp_path / "sdk", link_mode="copy", per_view_files={"projectionMatrix.bin": 84, "weightR.bin": 4}),
        reconstruct_fn=reconstruct,
    )
    assert record.shape == (2, 3, 4)
    assert np.load(record.even_path).shape == record.shape
    assert np.load(record.odd_path).shape == record.shape
    assert np.load(record.full_path).shape == record.shape
    metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert metadata["spacing_zyx"] == [0.3, 0.2, 0.2]
    assert metadata["full_path"] == str(record.full_path.resolve())
    assert metadata["full_projections"] == [f"{i:05d}.raw" for i in range(4)]
    assert len(calls) == 3
    assert calls[0][0] == raw
    assert calls[0][1] == cfg
    assert calls[0][4] == [0.0, 1.0, 2.0, 3.0]
    assert calls[1][4] == [0.0, 2.0]
    assert calls[2][4] == [1.0, 3.0]


def test_prepare_case_rejects_full_spacing_mismatch(tmp_path):
    raw = tmp_path / "case" / "scan_Mode2_100kV8mA1cc_S4"
    cfg = tmp_path / "case" / "config" / "Mode2"
    raw.mkdir(parents=True)
    cfg.mkdir(parents=True)
    for i in range(4):
        (raw / f"{i:05d}.raw").write_bytes(bytes([i]))
    (cfg / "filter.xml").write_text("<x/>")
    (cfg / "fdkPara.xml").write_text("<FDK_sys_para><noView>4</noView></FDK_sys_para>")
    np.arange(4 * 21, dtype=np.float32).tofile(cfg / "projectionMatrix.bin")
    np.arange(4, dtype=np.float32).tofile(cfg / "weightR.bin")

    def reconstruct(raw_dir, cfg_dir, *args, **kwargs):
        spacing = 0.4 if raw_dir == raw else 0.3
        return SimpleNamespace(
            volume=np.zeros((2, 3, 4), dtype=np.float32),
            volume_info=SimpleNamespace(z_spacing=spacing, y_spacing=0.2, x_spacing=0.2),
        )

    with np.testing.assert_raises_regex(ValueError, "spacing mismatch"):
        prepare_case(
            ScanCase("case", raw, cfg, "Mode2"),
            tmp_path / "out",
            PrepareOptions(xday_dir=tmp_path / "sdk", link_mode="copy", per_view_files={"projectionMatrix.bin": 84, "weightR.bin": 4}),
            reconstruct_fn=reconstruct,
        )
