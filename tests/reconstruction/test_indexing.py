from pathlib import Path

from cbct_n2n.reconstruction.indexing import scan_reconstructable


def test_scan_reconstructable_matches_mode_and_z_spacing(tmp_path):
    case = tmp_path / "case-a"
    raw = case / "scan_100kV10mA_Mode2"
    cfg = case / "config" / "recon" / "Mode2"
    raw.mkdir(parents=True)
    cfg.mkdir(parents=True)
    (raw / "00001.raw").write_bytes(b"0")
    (raw / "00002.raw").write_bytes(b"0")
    (cfg / "filter.xml").write_text("<filter />", encoding="utf-8")
    (cfg / "fdkPara.xml").write_text(
        "<fdk><xSpacing>0.25</xSpacing><ySpacing>0.25</ySpacing><zSpacing>0.25</zSpacing><zNum>360</zNum></fdk>",
        encoding="utf-8",
    )

    dataset = scan_reconstructable(tmp_path)

    assert len(dataset["records"]) == 1
    assert dataset["unmatched"] == []
    record = dataset["records"][0]
    assert record["raw_dir"] == Path("case-a/scan_100kV10mA_Mode2").as_posix()
    assert record["cfg_dir"] == Path("case-a/config/recon/Mode2").as_posix()
    assert record["projection_count"] == 2
    assert record["z_spacing"] == 0.25
