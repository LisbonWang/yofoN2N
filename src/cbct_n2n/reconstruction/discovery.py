from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .subset import numbered_raw_files

_MODE = re.compile(r"(?:^|_)(Mode\d+)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ScanCase:
    case_id: str
    raw_dir: Path
    cfg_dir: Path
    mode: str


def _mode(path: Path) -> str | None:
    match = _MODE.search(path.name)
    return match.group(1) if match else None


def discover_cases(root: str | Path, *, continue_on_error: bool = False, mode: str | None = None) -> list[ScanCase]:
    source = Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"Source root not found: {source}")
    cases: list[ScanCase] = []
    requested_mode = mode.casefold() if mode is not None else None
    for case_dir in sorted((p for p in source.iterdir() if p.is_dir()), key=lambda p: p.name):
        raw_dirs = [p for p in case_dir.iterdir() if p.is_dir() and numbered_raw_files(p)]
        configs = [p.parent for p in case_dir.rglob("filter.xml") if (p.parent / "fdkPara.xml").is_file()]
        for raw_dir in raw_dirs:
            case_mode = _mode(raw_dir)
            if requested_mode is not None and (case_mode is None or case_mode.casefold() != requested_mode):
                continue
            matches = [cfg for cfg in configs if _mode(cfg) == case_mode] if case_mode else configs
            if len(matches) != 1:
                details = ", ".join(str(p) for p in matches or configs)
                if continue_on_error:
                    continue
                raise ValueError(f"Expected one config for {raw_dir}; found {len(matches)}: {details}")
            cases.append(ScanCase(case_dir.name, raw_dir, matches[0], case_mode or "unknown"))
    if requested_mode is not None and not cases:
        raise ValueError(f"No cases found for mode {mode!r}")
    return cases
