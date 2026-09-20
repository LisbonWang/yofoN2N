"""Unit tests for the single-study conventional-denoise driver (no GPU, no SDK)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.recon_conventional_denoise import (
    DEFAULT_POST_PROCS,
    _block_snapshot,
    _denoise_params_snapshot,
    _flag,
    _parse_args,
    _probe_study,
)


def _make_study(root: Path, case: str, study: str, *, raw_name: str, cfg_mode: str = "Mode2",
                n_views: int = 4, with_raw: bool = True, extra_raw: str | None = None) -> Path:
    study_dir = root / case / study
    if with_raw:
        raw_dir = study_dir / raw_name
        raw_dir.mkdir(parents=True)
        for index in range(1, n_views + 1):
            (raw_dir / f"{index:05d}.raw").write_bytes(b"\0\0")
    if extra_raw:
        (study_dir / extra_raw).mkdir(parents=True, exist_ok=True)
        (study_dir / extra_raw / "00001.raw").write_bytes(b"\0\0")
    cfg_dir = study_dir / "config" / "recon" / "PDI1313" / cfg_mode
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "filter.xml").write_text("<root/>", encoding="utf-8")
    (cfg_dir / "fdkPara.xml").write_text("<root/>", encoding="utf-8")
    return study_dir


RAW_NAME = "1.2.156.1.2_Mode2_100.0kV8mA1cc_S640_H768_V640_B"


def test_probe_study_parses_mode_kv_ma_and_view_count(tmp_path):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=RAW_NAME, n_views=4)
    study = _probe_study(tmp_path, "CASE1", "STUDY1")
    assert study is not None
    assert study.mode == "Mode2"
    assert study.kv == 100.0
    # 8mA1cc means mA x cc = 8 x 1 = 8, not 81.
    assert study.ma == 8.0
    assert study.nviews == 4
    assert study.case_id == "CASE1"
    assert study.study_uid == "STUDY1"


def test_probe_study_applies_cc_factor(tmp_path):
    name = "s_Mode3_85.0kV12.5mA2cc_S500_H768_V640_A"
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=name, cfg_mode="Mode3")
    study = _probe_study(tmp_path, "CASE1", "STUDY1")
    assert study is not None
    assert (study.kv, study.ma) == (85.0, 25.0)


def test_probe_study_accepts_names_without_cc(tmp_path):
    name = "s_Mode2_90.0kV10mA_S900_H768_V640_B"
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=name)
    study = _probe_study(tmp_path, "CASE1", "STUDY1")
    assert study is not None
    assert (study.kv, study.ma) == (90.0, 10.0)


def test_probe_study_returns_none_for_unknown_case(tmp_path):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=RAW_NAME)
    assert _probe_study(tmp_path, "CASE2", "STUDY1") is None


def test_probe_study_returns_none_when_two_raw_dirs(tmp_path):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=RAW_NAME,
                extra_raw="other_Mode2_100.0kV8mA_S640_H768_V640_B")
    assert _probe_study(tmp_path, "CASE1", "STUDY1") is None


def test_probe_study_returns_none_on_mode_mismatch(tmp_path):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=RAW_NAME, cfg_mode="Mode3")
    assert _probe_study(tmp_path, "CASE1", "STUDY1") is None


def test_probe_study_returns_none_without_raw_dir(tmp_path):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=RAW_NAME, with_raw=False)
    assert _probe_study(tmp_path, "CASE1", "STUDY1") is None


@pytest.mark.parametrize("name", ["1.2.156.1.2_Mode2_S640_H768_V640_B", "plain_dir"])
def test_probe_study_returns_none_when_kv_ma_unparseable(tmp_path, name):
    _make_study(tmp_path, "CASE1", "STUDY1", raw_name=name)
    assert _probe_study(tmp_path, "CASE1", "STUDY1") is None


def test_default_chain_keeps_bhc_and_enables_both_denoisers():
    chain = DEFAULT_POST_PROCS.split()
    # BHC keeps the beam-hardening correction the full.bhc AI input already carries;
    # both conventional denoisers must be switched on for a like-for-like comparison.
    assert chain == ["BHC", "ReduceNoise", "Anisotropic"]


# ------------------------------------------------------------------ filter.xml swapping


def _write_filter(path: Path, *, med: int = 7, weight_z: str = "0.0015") -> Path:
    """Write a file shaped like a real filter.xml, double-colon tag and all."""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- commented out on purpose: <ReduceNoise><medFilterLen>999</medFilterLen>"
        "</ReduceNoise> -->\n"
        "<FDK_filter_config>\n"
        "\t<SecondaryBP::BHC>\n"
        "\t\t<BHC>\n\t\t\t<pwb>0.03</pwb>\n\t\t</BHC>\n"
        "\t</SecondaryBP::BHC>\n"
        "\t<ReduceNoise>\n"
        "\t\t<ratio>1.0</ratio>\n"
        "\t\t<gradientThresh>0.01</gradientThresh>\n"
        "\t\t<time>6</time>\n"
        f"\t\t<medFilterLen>{med}</medFilterLen>\n"
        "\t</ReduceNoise>\n"
        "\t<Anisotropic>\n"
        "\t\t<stepNum>1</stepNum>\n"
        "\t\t<filterLen>5</filterLen>\n"
        "\t\t<sigma>3.0</sigma>\n"
        f"\t\t<weightZ>{weight_z}</weightZ>\n"
        "\t\t<time>1</time>\n"
        "\t</Anisotropic>\n"
        "</FDK_filter_config>\n",
        encoding="utf-8",
    )
    return path


def test_denoise_snapshot_reads_an_annotated_filter_file(tmp_path):
    snap = _denoise_params_snapshot(_write_filter(tmp_path / "filter.xml"))
    assert snap["ReduceNoise"] == {
        "ratio": "1.0", "gradientThresh": "0.01", "time": "6", "medFilterLen": "7",
    }
    assert snap["Anisotropic"]["weightZ"] == "0.0015"
    assert snap["Anisotropic"]["sigma"] == "3.0"


def test_denoise_snapshot_ignores_a_commented_out_block(tmp_path):
    """The sweep copy carries TUNABLE comments; a commented block must never win."""
    snap = _denoise_params_snapshot(_write_filter(tmp_path / "filter.xml"))
    assert snap["ReduceNoise"]["medFilterLen"] == "7"  # not the 999 inside the comment


def test_denoise_snapshot_survives_the_double_colon_tag(tmp_path):
    """<SecondaryBP::BHC> is not valid XML, so an XML parser cannot even open the file."""
    import xml.etree.ElementTree as ET

    path = _write_filter(tmp_path / "filter.xml")
    with pytest.raises(ET.ParseError):
        ET.parse(path)
    assert set(_denoise_params_snapshot(path)) == {"ReduceNoise", "Anisotropic"}


def test_denoise_snapshot_tracks_parameter_edits(tmp_path):
    snap = _denoise_params_snapshot(_write_filter(tmp_path / "filter.xml", med=3, weight_z="0.02"))
    assert snap["ReduceNoise"]["medFilterLen"] == "3"
    assert snap["Anisotropic"]["weightZ"] == "0.02"


def test_denoise_snapshot_returns_empty_blocks_when_absent(tmp_path):
    path = tmp_path / "filter.xml"
    path.write_text("<FDK_filter_config>\n\t<BHC/>\n</FDK_filter_config>\n", encoding="utf-8")
    assert _denoise_params_snapshot(path) == {"ReduceNoise": {}, "Anisotropic": {}}


def test_parse_args_accepts_a_custom_filter_file():
    args = _parse_args([
        "--case", "C", "--study", "S", "--out", "o.raw",
        "--filter-file", "configs/recon_filters/Mode2_1609.convdenoise.filter.xml",
    ])
    assert args.filter_file == Path("configs/recon_filters/Mode2_1609.convdenoise.filter.xml")


def _write_pwls_filter(path: Path, *, enable: str = "1", beta: str = "600.0") -> Path:
    path.write_text(
        "<FDK_filter_config>\n"
        "\t<Pwls>\n"
        f"\t\t<isEnable>{enable}</isEnable>\n"
        "\t\t<iterCount>20</iterCount>\n"
        "\t\t<delta>0.5</delta>\n"
        f"\t\t<beta>{beta}</beta>\n"
        "\t</Pwls>\n"
        "\t<Ramp>\n\t\t<isHamEnable>0</isHamEnable>\n\t</Ramp>\n"
        "</FDK_filter_config>\n",
        encoding="utf-8",
    )
    return path


def test_pwls_block_is_read_from_the_filter_file(tmp_path):
    """Without --pwls-scale the file decides, so its Pwls block must be reportable."""
    block = _block_snapshot(_write_pwls_filter(tmp_path / "filter.xml"), ("Pwls",))["Pwls"]
    assert block["isEnable"] == "1"
    assert block["beta"] == "600.0"
    assert _flag(block["isEnable"]) is True


def test_pwls_block_absent_reads_as_disabled(tmp_path):
    path = tmp_path / "filter.xml"
    path.write_text("<FDK_filter_config>\n\t<BHC/>\n</FDK_filter_config>\n", encoding="utf-8")
    block = _block_snapshot(path, ("Pwls",))["Pwls"]
    assert block == {}
    assert _flag(block.get("isEnable")) is False


def test_pwls_block_ignores_a_commented_out_copy(tmp_path):
    path = tmp_path / "filter.xml"
    path.write_text(
        "<FDK_filter_config>\n"
        "\t<!--\n\t<Pwls>\n\t\t<isEnable>0</isEnable>\n\t</Pwls>\n\t-->\n"
        "\t<Pwls>\n\t\t<isEnable>1</isEnable>\n\t\t<beta>600.0</beta>\n\t</Pwls>\n"
        "</FDK_filter_config>\n",
        encoding="utf-8",
    )
    assert _block_snapshot(path, ("Pwls",))["Pwls"]["isEnable"] == "1"


def test_flag_accepts_the_usual_spellings():
    assert _flag("1") and _flag("true") and _flag("ON")
    assert not _flag("0") and not _flag("false") and not _flag(None)


def test_parse_args_leaves_the_filter_file_unset_by_default():
    args = _parse_args(["--case", "C", "--study", "S", "--out", "o.raw"])
    assert args.filter_file is None
