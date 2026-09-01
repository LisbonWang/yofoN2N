from pathlib import Path
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import xml.etree.ElementTree as ET

from cbct_n2n.reconstruction import (
    ReconGroup,
    Study,
    VariantOptions,
    apply_fdk_para,
    apply_filter_file,
    build_full_workspace,
    discover_studies,
    fix_subsample_step,
    groups_for_mode,
    load_recon_groups,
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
#: real FOV suffixes: Mode0/Mode1 -> 160x150 -> "1615", Mode2 -> 160x90 -> "1609",
#: Mode3 -> 80x80 -> "0808".
_MODE_FDK = {
    "Mode0": (640, 640, 600),
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


def _make_group(name: str = "Mode2_1609", modes: tuple[str, ...] = ("Mode2",),
                fdk: dict[str, float] | None = None, filter_file: str | Path | None = None) -> ReconGroup:
    return ReconGroup(name=name, source_modes=modes,
                      fdk_para=fdk or {}, filter_file=str(filter_file) if filter_file else None)


def _fdk_field(cfg: Path, field: str) -> str:
    tree = ET.parse(cfg / "fdkPara.xml")
    node = tree.getroot().find(field)
    return node.text if node is not None else None


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


def test_apply_filter_file_replaces_whole_file(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "filter.xml").write_text(_filter_xml("2"), encoding="utf-8")
    std = tmp_path / "std.filter.xml"
    std.write_text(_filter_xml("4"), encoding="utf-8")
    # different content -> replaced, reported as changed
    assert apply_filter_file(cfg, std) is True
    assert (cfg / "filter.xml").read_text(encoding="utf-8") == _filter_xml("4")
    # identical content -> copied again, reported as unchanged
    assert apply_filter_file(cfg, std) is False
    assert (cfg / "filter.xml").read_text(encoding="utf-8") == _filter_xml("4")
    # missing standard file raises
    with pytest.raises(ValueError, match="not found"):
        apply_filter_file(cfg, tmp_path / "nope.filter.xml")


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


def test_discover_studies_mode0(tmp_path):
    """A Mode0 scan is discovered like any other Mode<N>."""
    root = tmp_path / "src"
    case = root / "YPDL-X-0001"
    study_dir = case / "study-uid-1"
    raw_dir = study_dir / "scan_Mode0_100.0kV8mA1cc_S640_H768_V640_B"
    raw_dir.mkdir(parents=True)
    cfg_dir = study_dir / "config" / "recon" / "PDI1" / "Mode0"
    cfg_dir.mkdir(parents=True)
    for i in range(3):
        (raw_dir / f"{i + 1:05d}.raw").write_bytes(b"x" * 8)
    (cfg_dir / "fdkPara.xml").write_text("<FDK_sys_para><noView>3</noView></FDK_sys_para>", encoding="utf-8")
    (cfg_dir / "filter.xml").write_text(_filter_xml(), encoding="utf-8")
    studies, skipped = discover_studies(root, mode="Mode0")
    assert len(studies) == 1
    assert studies[0].mode == "Mode0"
    assert skipped == []


def test_fov_suffix_and_mode_fov_key(tmp_path):
    from cbct_n2n.reconstruction import fov_suffix_from_fdk, mode_fov_key
    m0 = _make_study(tmp_path / "m0", mode="Mode0")
    m1 = _make_study(tmp_path / "m1", mode="Mode1")
    m2 = _make_study(tmp_path / "m2", mode="Mode2")
    m3 = _make_study(tmp_path / "m3", mode="Mode3")
    assert fov_suffix_from_fdk(m0.cfg_dir / "fdkPara.xml") == "1615"
    assert fov_suffix_from_fdk(m1.cfg_dir / "fdkPara.xml") == "1615"
    assert fov_suffix_from_fdk(m2.cfg_dir / "fdkPara.xml") == "1609"
    assert fov_suffix_from_fdk(m3.cfg_dir / "fdkPara.xml") == "0808"
    assert mode_fov_key("Mode0", m0.cfg_dir / "fdkPara.xml") == "Mode0_1615"
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
    group = _make_group()
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True, top_dir="PDL")
    record = reconstruct_study(study, group, out, options, reconstruct_fn=_make_fake_reconstruct(calls))

    assert record["status"] == "success"
    assert record["top_dir"] == "PDL"
    assert record["group"] == "Mode2_1609"
    assert record["shape"] == [8, 6, 4]
    assert record["spacing_zyx"] == [0.25, 0.25, 0.25]
    assert len(record["outputs"]) == 6
    assert set(record["outputs"]) == {
        "even.raw", "odd.raw", "full.raw",
        "full.marfi.raw", "full.bhc.raw", "full.marfi.bhc.raw",
    }
    # every output lives under <top>/<group>/<case_id>/<study_uid>/
    for name in record["outputs"]:
        rel = record["outputs"][name]
        assert rel == f"PDL/Mode2_1609/CASE-1/STUDY-1/{name}"
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
    ws = out / "PDL" / "Mode2_1609" / "CASE-1" / "STUDY-1" / "workspace"
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
    record2 = reconstruct_study(study, group, out, options, reconstruct_fn=_make_fake_reconstruct(calls))
    assert record2["skipped"] is True and calls == []


def test_reconstruct_study_applies_group_fdk_para_and_standard_filter(tmp_path):
    """Group fdkPara overrides reshape the volume (Mode2 -> 0808) and the standard filter file replaces filter.xml."""
    study = _make_study(tmp_path, count=5, step="2")
    out = tmp_path / "out"
    calls = []
    std_filter = tmp_path / "std.filter.xml"
    std_filter.write_text(_filter_xml("4"), encoding="utf-8")  # differs from the source step="2"
    group = _make_group(
        name="Mode2_0808",
        fdk={"xNum": 640, "yNum": 640, "zNum": 640, "xSpacing": 0.125, "ySpacing": 0.125, "zSpacing": 0.125},
        filter_file=std_filter,
    )
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True, top_dir="PDL")
    record = reconstruct_study(study, group, out, options, reconstruct_fn=_make_fake_reconstruct(calls))

    assert record["status"] == "success"
    assert record["group"] == "Mode2_0808"
    assert record["fdk_para"]["xNum"] == 640 and record["fdk_para"]["zSpacing"] == 0.125
    assert "xNum" in record["fdk_changed"] and "zSpacing" in record["fdk_changed"]
    assert record["filter_file"] == str(std_filter)
    assert record["filter_replaced"] is True
    # fdkPara geometry patched and standard filter.xml copied in every subset's workspace
    ws = out / "PDL" / "Mode2_0808" / "CASE-1" / "STUDY-1" / "workspace"
    for subset in ("full", "even", "odd"):
        assert _fdk_field(ws / subset / "config", "xNum") == "640"
        assert _fdk_field(ws / subset / "config", "zNum") == "640"
        assert _fdk_field(ws / subset / "config", "zSpacing") == "0.125"
        assert (ws / subset / "config" / "filter.xml").read_text(encoding="utf-8") == _filter_xml("4")
    # source config untouched
    assert _fdk_field(study.cfg_dir, "zNum") == "360"
    assert _fdk_field(study.cfg_dir, "zSpacing") == "0.25"
    assert _marfi_step(study.cfg_dir) == 2
    for name in record["outputs"]:
        assert record["outputs"][name] == f"PDL/Mode2_0808/CASE-1/STUDY-1/{name}"


def test_apply_fdk_para_overwrites_and_validates(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "fdkPara.xml").write_text(
        "<FDK_sys_para><xNum>640</xNum><zNum>360</zNum><zSpacing>0.25</zSpacing></FDK_sys_para>",
        encoding="utf-8")
    changed = apply_fdk_para(cfg, {"xNum": 320, "zSpacing": 0.125})
    assert changed == ["xNum", "zSpacing"]
    assert _fdk_field(cfg, "xNum") == "320"
    assert _fdk_field(cfg, "zSpacing") == "0.125"
    # untouched fields preserved
    assert _fdk_field(cfg, "zNum") == "360"
    # unknown field raises
    with pytest.raises(ValueError, match="noSuchField"):
        apply_fdk_para(cfg, {"noSuchField": 1})
    # missing file raises
    with pytest.raises(FileNotFoundError):
        apply_fdk_para(tmp_path / "nope", {"xNum": 1})


def test_load_recon_groups_and_groups_for_mode(tmp_path):
    filters = tmp_path / "recon_filters"
    filters.mkdir()
    (filters / "Mode2.filter.xml").write_text(_filter_xml("4"), encoding="utf-8")
    cfg_file = tmp_path / "groups.json"
    cfg_file.write_text(json.dumps({
        "schema_version": 1,
        "groups": {
            "Mode2_1609": {"source_modes": ["Mode2"],
                           "fdk_para": {"xNum": 640, "zNum": 360, "zSpacing": 0.25},
                           "filter_file": "recon_filters/Mode2.filter.xml"},
            "Mode2_0808": {"source_modes": ["Mode2"],
                           "fdk_para": {"xNum": 640, "zNum": 640, "zSpacing": 0.125},
                           "filter_file": "recon_filters/Mode2.filter.xml"},
            "Mode0_1615": {"source_modes": ["Mode0"],
                           "fdk_para": {"zNum": 600}, "filter_file": None},
        },
    }), encoding="utf-8")
    groups = load_recon_groups(cfg_file)
    assert set(groups) == {"Mode2_1609", "Mode2_0808", "Mode0_1615"}
    assert groups["Mode2_0808"].fdk_para["zSpacing"] == 0.125
    # relative filter_file is resolved against the JSON's directory
    assert groups["Mode2_0808"].filter_file == str((filters / "Mode2.filter.xml").resolve())
    assert groups["Mode0_1615"].filter_file is None
    # a Mode2 study matches both Mode2 groups, in file order
    assert [g.name for g in groups_for_mode(groups, "Mode2")] == ["Mode2_1609", "Mode2_0808"]
    assert [g.name for g in groups_for_mode(groups, "Mode0")] == ["Mode0_1615"]
    assert groups_for_mode(groups, "Mode1") == []
    # missing standard filter file raises
    cfg_file.write_text(json.dumps({
        "schema_version": 1,
        "groups": {"Mode2_1609": {"source_modes": ["Mode2"], "fdk_para": {},
                                  "filter_file": "recon_filters/nope.xml"}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="filter_file not found"):
        load_recon_groups(cfg_file)
    # schema guard
    cfg_file.write_text('{"schema_version": 2, "groups": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_recon_groups(cfg_file)


def test_reconstruct_study_mode0_no_per_view(tmp_path):
    """Mode0 (no per-view config) must reconstruct fine and copy ancillary files whole."""
    study = _make_study(tmp_path, count=5, mode="Mode0")
    out = tmp_path / "out"
    calls = []
    group = _make_group(name="Mode0_1615", modes=("Mode0",))
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True, top_dir="PDL")
    record = reconstruct_study(study, group, out, options, reconstruct_fn=_make_fake_reconstruct(calls))

    assert record["status"] == "success"
    assert record["group"] == "Mode0_1615"
    assert record["top_dir"] == "PDL"
    # no per-view mapping for Mode0 -> nothing sliced, no warnings
    assert record["per_view_sliced"] == {}
    assert record["per_view_warnings"] == []
    for name in record["outputs"]:
        rel = record["outputs"][name]
        assert rel == f"PDL/Mode0_1615/CASE-1/STUDY-1/{name}"
        assert (out / rel).is_file()
    assert len(calls) == 6


def test_reconstruct_study_standard_filter_replaced_or_kept(tmp_path):
    # group with a standard filter file -> filter.xml replaced in every subset
    study = _make_study(tmp_path / "a", count=3, step="2")
    out = tmp_path / "out"
    calls = []
    std_filter = tmp_path / "std.filter.xml"
    std_filter.write_text(_filter_xml("6"), encoding="utf-8")
    group = _make_group(filter_file=std_filter)
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True, top_dir="PDL")
    record = reconstruct_study(study, group, out, options, reconstruct_fn=_make_fake_reconstruct(calls))
    assert record["status"] == "success"
    assert record["filter_file"] == str(std_filter)
    assert record["filter_replaced"] is True
    ws = out / "PDL" / "Mode2_1609" / "CASE-1" / "STUDY-1" / "workspace"
    for subset in ("full", "even", "odd"):
        assert (ws / subset / "config" / "filter.xml").read_text(encoding="utf-8") == _filter_xml("6")

    # group without a standard file -> the source filter.xml is kept untouched
    study2 = _make_study(tmp_path / "b", count=3, step="2")
    calls.clear()
    record2 = reconstruct_study(study2, _make_group(name="Mode2_0808"),
                                out, options, reconstruct_fn=_make_fake_reconstruct(calls))
    assert record2["status"] == "success"
    assert record2["filter_file"] is None
    assert record2["filter_replaced"] is False
    ws2 = out / "PDL" / "Mode2_0808" / "CASE-1" / "STUDY-1" / "workspace"
    for subset in ("full", "even", "odd"):
        assert _marfi_step(ws2 / subset / "config") == 2  # source step preserved
    # even/odd are already even so no step_fixes recorded
    assert record2["step_fixes"] == {}
