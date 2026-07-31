from __future__ import annotations

import hashlib
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_RAW_NAME = re.compile(r"\d+\.raw$", re.IGNORECASE)
_MODE = re.compile(r"(?:^|_)(Mode\d+)(?:_|$)", re.IGNORECASE)
_KV_MA = re.compile(r"_(\d+(?:\.\d+)?)kV(\d+(?:\.\d+)?)mA(?:(\d+(?:\.\d+)?)cc)?(?:_|$)", re.IGNORECASE)


def _mode(path: Path) -> str | None:
    match = _MODE.search(path.name)
    return match.group(1) if match else None


def _parse_kv_ma(name: str) -> tuple[float, float] | None:
    match = _KV_MA.search(name)
    if match is None:
        return None
    kv, ma, cc = match.groups()
    return float(kv), float(ma) * (float(cc) if cc else 1.0)


def _z_spacing(cfg_dir: Path) -> float | None:
    try:
        root = ET.parse(cfg_dir / "fdkPara.xml").getroot()
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] == "zSpacing":
                return float((node.text or "").strip())
    except (ET.ParseError, OSError, ValueError):
        return None
    return None


def _case_id(raw_rel: str) -> str:
    return "scan-" + hashlib.sha1(raw_rel.encode("utf-8")).hexdigest()[:16]


def _split(ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must be three non-negative values summing to one")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count = int(len(shuffled) * ratios[0]), int(len(shuffled) * ratios[1])
    return {"train": shuffled[:train_count], "val": shuffled[train_count:train_count + val_count], "test": shuffled[train_count + val_count:]}


def build_catalog(source_root: str | Path, *, z_spacing: float = 0.25,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> dict[str, Any]:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source root not found: {root}")
    raw_dirs = sorted({path.parent for path in root.rglob("*.raw") if _RAW_NAME.fullmatch(path.name)})
    records: list[dict[str, Any]] = []
    for raw_dir in raw_dirs:
        raw_rel = raw_dir.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "case_id": _case_id(raw_rel),
            "raw_dir": raw_rel,
            "projection_count": sum(1 for path in raw_dir.iterdir() if path.is_file() and _RAW_NAME.fullmatch(path.name)),
            "mode": _mode(raw_dir),
        }
        kv_ma = _parse_kv_ma(raw_dir.name)
        if kv_ma is not None:
            record["kv"], record["ma"] = kv_ma
        candidates = []
        for fdk in raw_dir.parent.rglob("fdkPara.xml"):
            cfg_dir = fdk.parent
            if (cfg_dir / "filter.xml").is_file() and _z_spacing(cfg_dir) is not None and abs(_z_spacing(cfg_dir) - z_spacing) < 1e-9:
                candidates.append(cfg_dir)
        mode = _mode(raw_dir)
        if mode is not None:
            mode_candidates = [path for path in candidates if (_mode(path) or "").casefold() == mode.casefold()]
            if mode_candidates:
                candidates = mode_candidates
        candidates = sorted(set(candidates))
        if len(candidates) == 1 and kv_ma is not None:
            cfg_dir = candidates[0]
            record.update({"cfg_dir": cfg_dir.relative_to(root).as_posix(), "z_spacing": z_spacing, "status": "ready"})
        elif len(candidates) != 1:
            record.update({"status": "unmatched_config", "candidate_config_count": len(candidates)})
        else:
            record.update({"cfg_dir": candidates[0].relative_to(root).as_posix(), "z_spacing": z_spacing, "status": "missing_kv_ma"})
        records.append(record)
    ready_ids = [record["case_id"] for record in records if record["status"] == "ready"]
    return {
        "schema_version": 1,
        "source_root": str(root),
        "selection": {"z_spacing": z_spacing, "raw_pattern": "numeric .raw projections", "requires_filter_xml": True, "requires_fdkPara_xml": True},
        "ratios": list(ratios),
        "seed": seed,
        "stats": {"raw_dirs": len(raw_dirs), "ready": len(ready_ids), "excluded": len(records) - len(ready_ids)},
        "records": records,
        "splits": _split(ready_ids, ratios, seed),
    }
