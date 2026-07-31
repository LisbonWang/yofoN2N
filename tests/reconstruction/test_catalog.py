import json

from cbct_n2n.reconstruction.catalog import build_catalog


def test_catalog_filters_z_spacing_and_keeps_relative_paths(tmp_path):
    case = tmp_path / "case-a"
    raw = case / "scan_Mode2_90kV8mA_"
    cfg = case / "config" / "Mode2"
    raw.mkdir(parents=True)
    cfg.mkdir(parents=True)
    (raw / "00001.raw").write_bytes(b"1")
    (cfg / "filter.xml").write_text("<filter />", encoding="utf-8")
    (cfg / "fdkPara.xml").write_text("<fdk><zSpacing>0.25</zSpacing></fdk>", encoding="utf-8")

    catalog = build_catalog(tmp_path, z_spacing=0.25, ratios=(1.0, 0.0, 0.0), seed=42)

    assert catalog["source_root"] == str(tmp_path.resolve())
    assert len(catalog["records"]) == 1
    record = catalog["records"][0]
    assert record["raw_dir"] == "case-a/scan_Mode2_90kV8mA_"
    assert record["cfg_dir"] == "case-a/config/Mode2"
    assert catalog["splits"]["train"] == [record["case_id"]]


def test_catalog_marks_unmatched_config_without_absolute_record_paths(tmp_path):
    raw = tmp_path / "case" / "scan_Mode2_90kV8mA_"
    raw.mkdir(parents=True)
    (raw / "00001.raw").write_bytes(b"1")

    catalog = build_catalog(tmp_path, z_spacing=0.25, ratios=(1.0, 0.0, 0.0), seed=1)

    assert catalog["records"][0]["status"] == "unmatched_config"
    assert "raw_dir" in catalog["records"][0]
    assert not any(str(tmp_path) in json.dumps(record) for record in catalog["records"])
