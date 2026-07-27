import pytest

from cbct_n2n.reconstruction.discovery import discover_cases


def test_discover_observed_uid_layout(tmp_path):
    case = tmp_path / "patient.uid"
    raw = case / "series_Mode2_100.0kV8mA1cc_S4"
    cfg = case / "config" / "recon" / "PDI1313" / "Mode2"
    raw.mkdir(parents=True)
    cfg.mkdir(parents=True)
    for i in range(4):
        (raw / f"{i:05d}.raw").write_bytes(b"x")
    (cfg / "filter.xml").write_text("<x/>")
    (cfg / "fdkPara.xml").write_text("<FDK_sys_para><noView>4</noView></FDK_sys_para>")
    found = discover_cases(tmp_path)
    assert len(found) == 1
    assert found[0].case_id == "patient.uid"
    assert found[0].mode == "Mode2"
    assert found[0].raw_dir == raw
    assert found[0].cfg_dir == cfg


def test_discovery_can_skip_ambiguous_case_when_requested(tmp_path):
    case = tmp_path / "patient.uid"
    raw = case / "series_Mode2_100.0kV8mA1cc_S4"
    raw.mkdir(parents=True)
    for i in range(2):
        (raw / f"{i:05d}.raw").write_bytes(b"x")
    for name in ("Mode2", "Mode2_backup"):
        cfg = case / "config" / name
        cfg.mkdir(parents=True)
        (cfg / "filter.xml").write_text("<x/>")
        (cfg / "fdkPara.xml").write_text("<FDK_sys_para><noView>2</noView></FDK_sys_para>")
    assert discover_cases(tmp_path, continue_on_error=True) == []


def test_discovery_filters_cases_by_mode_case_insensitively(tmp_path):
    for mode in ("Mode1", "Mode2"):
        case = tmp_path / f"patient-{mode}"
        raw = case / f"series_{mode}_100.0kV8mA1cc_S4"
        cfg = case / "config" / mode
        raw.mkdir(parents=True)
        cfg.mkdir(parents=True)
        (raw / "00000.raw").write_bytes(b"x")
        (cfg / "filter.xml").write_text("<x/>")
        (cfg / "fdkPara.xml").write_text("<FDK_sys_para><noView>1</noView></FDK_sys_para>")
    found = discover_cases(tmp_path, mode="mode2")
    assert [case.mode for case in found] == ["Mode2"]


def test_discovery_rejects_requested_mode_without_cases(tmp_path):
    with pytest.raises(ValueError, match="Mode2"):
        discover_cases(tmp_path, mode="Mode2")
