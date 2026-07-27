from pathlib import Path

import pytest

from cbct_n2n.config import load_config


def test_load_config_has_required_sections():
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert {"reconstruction", "data", "model", "training"} <= set(config)
    assert "patch_size" not in config["data"]


def test_load_config_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
