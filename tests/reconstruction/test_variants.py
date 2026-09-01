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


def _make_study(tmp_path: Path, count: int = 5, step: str = "2", mode: str = "Mode2") -> Study:
    raw = tmp_path / "raw"
    cfg = tmp_path / "cfg"
    raw.mkdir()
    cfg.mkdir()
    for i in range(count):
        (raw / f"{i + 1:05d}.raw").write_bytes(bytes([i]) * 64)
    (cfg / "fdkPara.xml").write_text(f"<FDK_sys_para><noView>{count}</noView></FDK_sys_para>", encoding="utf-8")
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


def test_reconstruct_study_writes_12_variants_and_fixes_step(tmp_path):
    study = _make_study(tmp_path, count=5, step="3")
    out = tmp_path / "out"
    calls = []
    options = VariantOptions(xday_dir=tmp_path / "sdk", keep_workspace=True)
    record = reconstruct_study(study, out, options, reconstruct_fn=_make_fake_reconstruct(calls))

    assert record["status"] == "success"
    assert record["shape"] == [8, 6, 4]
    assert record["spacing_zyx"] == [0.25, 0.25, 0.25]
    assert len(record["outputs"]) == 12
    assert set(record["outputs"]) == {
        f"{subset}_{tag}.raw"
        for subset in ("full", "even", "odd")
        for tag in ("none", "bhc", "marfi", "bhc_marfi")
    }
    for name in record["outputs"]:
        path = out / record["outputs"][name]
        assert path.is_file() and path.stat().st_size == 8 * 6 * 4 * 4

    # 12 reconstructions: 3 subsets x 4 combos
    assert len(calls) == 12
    assert sorted(c["post_procs"] for c in calls) == sorted(
        [p for _s in range(3) for p in ("", "BHC", "MARFI", "BHC MARFI")]
    )
    # each subset reconstructed with the right view count
    counts = sorted(c["raw_count"] for c in calls)
    assert counts == [2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5]

    # step fixed only for downsampled subsets; full keeps the original odd step
    ws = out / "CASE-1" / "STUDY-1" / "workspace"
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
    ws = out / "CASE-1" / "STUDY-1" / "workspace"
    for subset in ("full", "even", "odd"):
        assert _marfi_step(ws / subset / "config") == 6
        assert read_module_field(ws / subset / "config", "ReduceNoise", "ratio") == "0.5"
    # even/odd are already even so no step_fixes recorded
    assert record["step_fixes"] == {}
