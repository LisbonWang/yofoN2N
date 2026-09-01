"""Batch-run helpers: index persistence, result merging and output verification.

These are pure functions so the CLI stays thin and the logic is unit-testable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_index(index_path: str | Path) -> dict[str, Any] | None:
    """Load a batch index written by ``reconstruct_pdl_variants.py``, or None.

    Returns None when the file is missing, not valid JSON, or its schema version
    does not match the one this module writes (schema_version == 1).
    """
    path = Path(index_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return None
    if not isinstance(data.get("records"), list):
        return None
    return data


def merge_index_records(existing: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge batch records keyed by ``study_uid``; fresh records win.

    Keeps the original order of the existing list, appends new study uids at the
    end, and replaces records whose study was re-processed in this run.
    """
    by_uid = {str(r.get("study_uid")): r for r in existing if isinstance(r, dict) and r.get("study_uid")}
    for r in fresh:
        uid = r.get("study_uid") if isinstance(r, dict) else None
        if not uid:
            continue
        by_uid[str(uid)] = r
    return list(by_uid.values())


def verify_volumes(record: dict[str, Any], output_root: str | Path) -> list[str]:
    """Check that every volume in ``record["outputs"]`` exists with the expected size.

    Expected size is ``prod(shape) * dtype_size`` using ``record["shape"]`` and
    ``record["dtype"]`` (e.g. ``"<f4"``).  Returns a list of human-readable issues;
    an empty list means everything checks out.
    """
    root = Path(output_root)
    issues: list[str] = []
    shape = record.get("shape")
    outputs = record.get("outputs")
    if not isinstance(shape, list) or not isinstance(outputs, dict):
        return ["record missing shape/outputs"]
    try:
        expected = math.prod(int(d) for d in shape)
    except TypeError:
        return [f"invalid shape {shape!r}"]
    dtype = record.get("dtype", "<f4")
    size = {"<f2": 2, "int16": 2, "uint16": 2, "<f4": 4, "float32": 4}.get(str(dtype), 4)
    expected_bytes = expected * size
    for name, rel in outputs.items():
        path = root / str(rel)
        if not path.is_file():
            issues.append(f"{name}: missing {path}")
        elif path.stat().st_size != expected_bytes:
            issues.append(f"{name}: size {path.stat().st_size} != expected {expected_bytes}")
    return issues
