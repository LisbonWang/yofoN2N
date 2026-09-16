"""Copy the original ``full.raw`` source volume into each eval case dir.

Places the source ``full.raw`` next to ``full_denoised.raw`` and
``full_metrics.json`` so every case dir is a self-contained triple:
``full.raw`` (input) / ``full_denoised.raw`` (denoised) / metrics json.

Streamed copy (shutil.copyfile) — does not load the ~0.59 GiB volume into
RAM.  Case dir naming matches ``test_raw_3d.py`` (``{group}__{study_uid}``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _resolve_output(record: dict[str, Any], logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (f"{logical}.raw", logical):
        if key in outputs:
            return outputs[key]
    return None


def _case_dir_name(record: dict[str, Any]) -> str:
    group = record.get("group") or "unknown_group"
    study_uid = record.get("study_uid") or record.get("key") or "unknown"
    return f"{group}__{study_uid}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy full.raw into eval case dirs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.parent
    records = {str(record["key"]): record for record in manifest["records"]}
    split_keys = list(manifest["splits"][args.split])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[copy] split={args.split} studies={len(split_keys)} -> {output_dir}", flush=True)

    for key in split_keys:
        record = records[key]
        relative = _resolve_output(record, "full")
        if relative is None:
            print(f"[skip] {key}: no full path in outputs", flush=True)
            continue
        src = base / relative
        if not src.exists():
            print(f"[skip] {key}: full missing {src}", flush=True)
            continue
        case_dir = output_dir / _case_dir_name(record)
        case_dir.mkdir(parents=True, exist_ok=True)
        dst = case_dir / "full.raw"
        if dst.exists():
            print(f"[exists] {case_dir.name}/full.raw, skip", flush=True)
            continue
        shutil.copyfile(src, dst)  # streamed, no RAM load
        print(f"[copied] {case_dir.name}/full.raw <- {relative}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
