from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import xml.etree.ElementTree as ET

from cbct_n2n.reconstruction import (
    Study,
    VariantOptions,
    apply_filter_overrides,
    build_full_workspace,
    discover_studies,
    fix_subsample_step,
    read_module_field,
    reconstruct_study,
    validated_per_view,
)


def _filter_xml(step: str = "2") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<FDK_filter_config>
    <MARFI>
        <metalThres>1.0</metalThres>
        <step>{step}</step>
    </MARFI>
    <SecondaryBP::BHC>
        <BHC><boneThresLow>0.3</boneThresLow></BHC>
        <time>12</time>
    </SecondaryBP::BHC>
    <ReduceNoise><ratio>1.0</ratio></ReduceNoise>
</FDK_filter_config>"""


#: per-mode (xNum, yNum, zNum) used by _make_study; 0.25mm spacing gives the
#: real FOV suffixes: Mode1 -> 160x150 -> "1615", Mode2 -> 160x90 -> "1609",
#: Mode3 -> 80x80 -> "0808".
_MODE_FDK = {
    "Mode1": (640, 640, 600),
    "Mode2": (640, 640, 360),
    "Mode3": (320, 320, 320),
}


def _make_study(tmp_path: Path, count: int = 5, step: str = "2", mode: str = "Mode2") -> Study:
    raw = tmp_path / "raw"
    cfg = tmp_path / "cfg"
    raw.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (raw / f"{i + 1:05d}.raw").write_bytes(bytes([i]) * 64)
    x_num, y_num, z_num = _MODE_FDK[mode]
    (cfg / "fdkPara.xml").write_text(
        f"<FDK_sys_para><noView>{count}</noView>"
        f"<xNum>{x_num}</xNum><yNum>{y_num}</yNum><zNum>{z_num}</zNum>"
        f"<xSpacing>0.25</xSpacing><ySpacing>0.25</ySpacing><zSpacing>0.25</zSpacing>"
        f"</FDK_sys_para>", encoding="utf-8")
    (cfg / "filter.xml").write_text(_filter_xml(step), encoding="utf-8")
    (cfg / "projectionMatrix.bin").write_bytes(b"".join(struct.pack("<21f", *([i] * 21)) for i in range(count)))
    (cfg / "weightR.bin").write_bytes(b"shared-calibration")
    return Study(
        case_id="CASE-1",
        study_uid="STUDY-1",
        scan_id="STUDY-1",
        raw_dir=raw,
        cfg_dir=cfg,
        mode=mode,
        kv=100.0,
        ma=8.0,
        nviews=count,
    )


def _no_view(cfg: Path) -> int:
    tree = ET.parse(cfg / "fdkPara.xml")
    return int(tree.getroot().find(".//noView").text)


def _marfi_step(cfg: Path) -> int:
    return int(read_module_field(cfg, "MARFI", "step"))


class _FakeResult:
    def __init__(self, volume: np.ndarray):
        self.volume = volume
        self.volume_info = SimpleNamespace(z_spacing=0.25, y_spacing=0.25, x_spacing=0.25)


def _make_fake_reconstruct(calls: list):
    def fake(raw_dir, cfg_dir, kv, ma, xday_install_dir, post_procs, dtype, show_progress, sdk):
        raw_dir, cfg_dir = Path(raw_dir), Path(cfg_dir)
        raw_count = len(list(raw_dir.glob("*.raw")))
        assert _no_view(cfg_dir) == raw_count, "noView must match the workspace projection count"
        calls.append({"raw_count": raw_count, "post_procs": post_procs, "kv": kv, "ma": ma})
        return _FakeResult(np.zeros((8, 6, 4), dtype="<f4"))
    return fake


def test_fix_subsample_step_rounds_odd_up_to_even(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "filter.xml").write_text(_filter_xml("3"), encoding="utf-8")
    assert fix_subsample_step(cfg) == [("MARFI", 3, 4)]
    assert _marfi_step(cfg) == 4
    # already even -> untouched
    (cfg / "filter.xml").write_text(_filter_xml("4"), encoding="utf-8")
    assert fix_subsample_step(cfg) == []
    assert _marfi_step(cfg) == 4
    # step <= 1 -> untouched
    (cfg / "filter.xml").write_text(_filter_xml("1"), encoding="utf-8")
    assert fix_subsample_step(cfg) == []
    assert _marfi_step(cfg) == 1


def test_apply_filter_overrides(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "filter.xml").write_text(_filter_xml("2"), encoding="utf-8")
    applied = apply_filter_overrides(cfg, {"MARFI.step": "4", "ReduceNoise.ratio": "0.5"})
    assert applied == [("MARFI.step", "4"), ("ReduceNoise.ratio", "0.5")]
    assert read_module_field(cfg, "MARFI", "step") == "4"
    assert read_module_field(cfg, "ReduceNoise", "ratio") == "0.5"
    # override of a BHC module with non-standard '::' tag also works
    applied = apply_filter_overrides(cfg, {"BHC.boneThresLow": "0.25"})
    assert applied == [("BHC.boneThresLow", "0.25")]
    assert read_module_field(cfg, "BHC", "boneThresLow") == "0.25"
    # comments and unrelated content are preserved by regex patching
    text = (cfg / "filter.xml").read_text(encoding="utf-8")
    assert "<?xml version" in text and "<SecondaryBP::BHC>" in text
    with pytest.raises(ValueError, match="missing_module"):
        apply_filter_overrides(cfg, {"missing_module.field": "1"})


def test_build_full_workspace_copies_and_normalizes_no_view(tmp_path):
    study = _make_study(tmp_path, count=7)
    out = tmp_path / "ws"
    raw_out, cfg_out = build_full_workspace(study.raw_dir, study.cfg_dir, out, link_mode="copy")
    assert len(list(raw_out.glob("*.raw"))) == 7
    assert _no_view(cfg_out) == 7
    assert (cfg_out / "weightR.bin").read_bytes() == b"shared-calibration"


def test_validated_per_view_drops_mismatched_files(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "projectionMatrix.bin").write_bytes(b"x" * (5 * 84))  # matches 5 views x 84
    (cfg / "other.bin").write_bytes(b"y" * 100)                  # does not match 5 x 84
    valid, warnings = validated_per_view(cfg, 5, {"projectionMatrix.bin": 84, "other.bin": 84})
    assert valid == {"projectionMatrix.bin": 84}
    assert len(warnings) == 1 and "other.bin" in warnings[0]


def test_discover_studies_pdl_layout(tmp_path):
    root = tmp_path / "src"
    case = root / "YPDL-X-0001"
    study_dir = case / "study-uid-1"
    raw_dir = study_dir / "scan_Mode2_100.0kV8mA1cc_S640_H768_V640_B"
    raw_dir.mkdir(parents=True)
    cfg_dir = study_dir / "config" / "recon" / "PDI1" / "Mode2"
    cfg_dir.mkdir(parents=True)
    for i in range(3):
        (raw_dir / f"{i + 1:05d}.raw").write_bytes(b"x" * 8)
    (cfg_dir / "fdkPara.xml").write_text("<FDK_sys_para><noView>3</noView></FDK_sys_para>", encoding="utf-8")
    (cfg_dir / "filter.xml").write_text(_filter_xml(), encoding="utf-8")
    # a study without config -> skipped
    no_cfg = case / "study-uid-2"
    (no_cfg / "scan_Mode2_100.0kV8mA1cc_S640_H768_V640_B").mkdir(parents=True)
    studies, skipped = discover_studies(root)
    assert len(studies) == 1
    assert studies[0].case_id == "YPDL-X-0001"
    assert studies[0].mode == "Mode2"
    assert studies[0].kv == 100.0 and studies[0].ma == 8.0
    assert studies[0].nviews == 3
    assert any("found 0" in s["reason"] for s in skipped)


def test_fov_suffix_and_mode_fov_key(tmp_path):
    from cbct_n2n.reconstruction import fov_suffix_from_fdk, mode_fov_key
    m1 = _make_study(tmp_path / "m1", mode="Mode1")
    m2 = _make_study(tmp_path / "m2", mode="Mode2")
    m3 = _make_study(tmp_path / "m3", mode="Mode3")
    assert fov_suffix_from_fdk(m1.cfg_dir / "fdkPara.xml") == "1615"
    assert fov_suffix_from_fdk(m2.cfg_dir / "fdkPara.xml") == "1609"
    assert fov_suffix_from_fdk(m3.cfg_dir / "fdkPara.xml") == "0808"
    assert mode_fov_key("Mode2", m2.cfg_dir / "fdkPara.xml") == "Mode2_1609"
    with pytest.raises(FileNotFoundError):
        fov_suffix_from_fdk(tmp_path / "missing.xml")
    # missing dimension field raises ValueError
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "fdkPara.xml").write_text("<FDK_sys_para><noView>3</noView></FDK_sys_para>", encoding="utf-8")
    with pytest.raises(ValueError):
        fov_suffix_from_fdk(bare / "fdkPara.xml")


def test_reconstruct_study_writes_6_variants_and_fixes_step(tmp_path):
    study = _make_study(tmp_path, count=5, step="3")
    out = tmp_path / "out"
    calls = []
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True)
    record = reconstruct_study(study, out, options, reconstruct_fn=_make_fake_reconstruct(calls))

    assert record["status"] == "success"
    assert record["mode_fov"] == "Mode2_1609"
    assert record["fov_suffix"] == "1609"
    assert record["shape"] == [8, 6, 4]
    assert record["spacing_zyx"] == [0.25, 0.25, 0.25]
    assert len(record["outputs"]) == 6
    assert set(record["outputs"]) == {
        "even.raw", "odd.raw", "full.raw",
        "full.marfi.raw", "full.bhc.raw", "full.marfi.bhc.raw",
    }
    # every output lives under <Mode>_<FOV>/<case_id>/<study_uid>/
    for name in record["outputs"]:
        rel = record["outputs"][name]
        assert rel == f"Mode2_1609/CASE-1/STUDY-1/{name}"
        path = out / rel
        assert path.is_file() and path.stat().st_size == 8 * 6 * 4 * 4

    # 6 reconstructions: even/odd plain only, full all four combos
    assert len(calls) == 6
    assert sorted(c["post_procs"] for c in calls) == sorted(
        ["", "", "", "BHC", "MARFI", "BHC MARFI"]
    )
    # full(5 views), even(3 views: indices 0,2,4), odd(2 views: indices 1,3)
    counts = sorted(c["raw_count"] for c in calls)
    assert counts == [2, 3, 5, 5, 5, 5]

    # step fixed only for downsampled subsets; full keeps the original odd step
    ws = out / "Mode2_1609" / "CASE-1" / "STUDY-1" / "workspace"
    assert _no_view(ws / "even" / "config") == 3
    assert _no_view(ws / "odd" / "config") == 2
    assert _no_view(ws / "full" / "config") == 5
    assert _marfi_step(ws / "even" / "config") == 4
    assert _marfi_step(ws / "odd" / "config") == 4
    assert _marfi_step(ws / "full" / "config") == 3
    assert record["step_fixes"]["even"] == [("MARFI", 3, 4)]
    assert record["step_fixes"]["odd"] == [("MARFI", 3, 4)]
    assert "full" not in record["step_fixes"]

    # second run without overwrite skips
    calls.clear()
    record2 = reconstruct_study(study, out, options, reconstruct_fn=_make_fake_reconstruct(calls))
    assert record2["skipped"] is True and calls == []


def test_reconstruct_study_applies_filter_overrides_everywhere(tmp_path):
    study = _make_study(tmp_path, count=3, step="2")
    out = tmp_path / "out"
    calls = []
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True,
                             filter_set={"MARFI.step": "6", "ReduceNoise.ratio": "0.5"})
    record = reconstruct_study(study, out, options, reconstruct_fn=_make_fake_reconstruct(calls))
    assert record["status"] == "success"
    assert record["filter_set"] == {"MARFI.step": "6", "ReduceNoise.ratio": "0.5"}
    ws = out / "Mode2_1609" / "CASE-1" / "STUDY-1" / "workspace"
    for subset in ("full", "even", "odd"):
        assert _marfi_step(ws / subset / "config") == 6
        assert read_module_field(ws / subset / "config", "ReduceNoise", "ratio") == "0.5"
    # even/odd are already even so no step_fixes recorded
    assert record["step_fixes"] == {}
