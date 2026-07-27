from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {"reconstruction", "data", "model", "training"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(sorted(missing))}")
    if "patch_size" in config["data"]:
        raise ValueError("2D patch sampling is not supported; remove patch_size")
    return config
