from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.data.manifest import create_case_splits, save_manifest


def _rebase_case_path(value: str, case_id: str, source_root: Path) -> str:
    marker = f"/{case_id}/"
    _, found, suffix = value.replace("\\", "/").partition(marker)
    if not found:
        return value
    return str(source_root / case_id / suffix)


def load_pair_record(metadata_path: Path, source_root: Path | None = None) -> dict:
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    case_dir = metadata_path.parent
    for name in ("even", "odd", "full"):
        record[f"{name}_path"] = str(case_dir / f"{name}.npy")
    if source_root is not None:
        case_id = str(record["case_id"])
        for name in ("raw_dir", "cfg_dir"):
            if name in record:
                record[name] = _rebase_case_path(str(record[name]), case_id, source_root)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Create case-level train/val/test splits from pair.json files.")
    parser.add_argument("pairs_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--source-root", type=Path, help="Rebase raw_dir and cfg_dir below this source root")
    args = parser.parse_args()
    records = [load_pair_record(path, args.source_root) for path in sorted(args.pairs_root.glob("*/pair.json"))]
    save_manifest(create_case_splits(records, tuple(args.ratios), args.seed), args.output)
    print(f"Wrote {args.output} with {len(records)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
