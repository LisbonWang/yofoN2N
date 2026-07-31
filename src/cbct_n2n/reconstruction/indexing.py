from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .subset import numbered_raw_files

_MODE = re.compile(r"(?:^|[_\\-])(Mode\d+)(?:[_\\-]|$)", re.IGNORECASE)
_RAW = re.compile(r"\d+\.raw$", re.IGNORECASE)


def _mode(path: Path) -> str | None:
    match = _MODE.search(path.name)
    return match.group(1) if match else None


def _xml_values(path: Path) -> dict[str, float]:
    tree = ET.parse(path)
    values: dict[str, float] = {}
    for node in tree.getroot().iter():
        name = node.tag.rsplit("}", 1)[-1]
        if name in {"xSpacing", "ySpacing", "zSpacing", "zNum"} and node.text:
            try:
                values[name] = float(node.text.strip())
            except ValueError:
                continue
    return values


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _case_id(raw_rel: str) -> str:
    digest = hashlib.sha1(raw_rel.encode("utf-8")).hexdigest()[:12]
    return f"scan-{digest}"


def _pair_score(raw_dir: Path, cfg_dir: Path, mode: str) -> tuple[int, int] | None:
    if mode is None or _mode(cfg_dir) is None or _mode(cfg_dir).casefold() != mode.casefold():
        return None
    raw_parts, cfg_parts = raw_dir.resolve().parts, cfg_dir.resolve().parts
    common = 0
    for left, right in zip(raw_parts, cfg_parts):
        if left.casefold() != right.casefold():
            break
        common += 1
    distance = (len(raw_parts) - common) + (len(cfg_parts) - common)
    return common, -distance


def scan_reconstructable(root: str | Path, *, z_spacing: float = 0.25, tolerance: float = 1e-6) -> dict[str, Any]:
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    raw_dirs = sorted({path.parent for path in source_root.rglob("*.raw") if _RAW.fullmatch(path.name)})
    configs: list[tuple[Path, dict[str, float]]] = []
    for fdk_path in source_root.rglob("fdkPara.xml"):
        cfg_dir = fdk_path.parent
        if not (cfg_dir / "filter.xml").is_file():
            continue
        try:
            values = _xml_values(fdk_path)
        except (OSError, ET.ParseError):
            continue
        if abs(values.get("zSpacing", float("nan")) - z_spacing) <= tolerance:
            configs.append((cfg_dir, values))

    records: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for raw_dir in raw_dirs:
        raw_rel = _relative(raw_dir, source_root)
        mode = _mode(raw_dir)
        candidates = [(cfg, values, _pair_score(raw_dir, cfg, mode)) for cfg, values in configs]
        candidates = [item for item in candidates if item[2] is not None]
        if not candidates:
            unmatched.append({"raw_dir": raw_rel, "reason": "no_unique_mode_and_spacing_config"})
            continue
        best_score = max(item[2] for item in candidates)
        best = [item for item in candidates if item[2] == best_score]
        if len(best) != 1:
            unmatched.append({"raw_dir": raw_rel, "reason": "ambiguous_config_match"})
            continue
        cfg_dir, values, _ = best[0]
        try:
            projections = numbered_raw_files(raw_dir)
        except (OSError, ValueError) as exc:
            unmatched.append({"raw_dir": raw_rel, "reason": str(exc)})
            continue
        cfg_rel = _relative(cfg_dir, source_root)
        case_id = _case_id(raw_rel)
        if case_id in used_ids:
            unmatched.append({"raw_dir": raw_rel, "reason": "duplicate_case_id"})
            continue
        used_ids.add(case_id)
        records.append({
            "case_id": case_id,
            "raw_dir": raw_rel,
            "cfg_dir": cfg_rel,
            "mode": mode or "unknown",
            "raw_name": raw_dir.name,
            "projection_count": len(projections),
            "z_spacing": values.get("zSpacing"),
            "x_spacing": values.get("xSpacing"),
            "y_spacing": values.get("ySpacing"),
            "z_num": values.get("zNum"),
            "reconstruction_status": "pending",
        })
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "selection": {"zSpacing": z_spacing, "tolerance": tolerance},
        "records": records,
        "unmatched": unmatched,
    }
