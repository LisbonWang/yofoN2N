from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.data.manifest import create_case_splits
from cbct_n2n.reconstruction import scan_reconstructable


def main() -> int:
    parser = argparse.ArgumentParser(description="Index reconstructable scans with zSpacing filtering.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--z-spacing", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    args = parser.parse_args()
    scanned = scan_reconstructable(args.source_root, z_spacing=args.z_spacing)
    dataset = create_case_splits(scanned["records"], tuple(args.ratios), args.seed) | {
        "schema_version": 1,
        "source_root": str(Path(args.source_root).resolve()),
        "selection": {"zSpacing": args.z_spacing, "tolerance": 1e-6},
        "unmatched": scanned["unmatched"],
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"wrote {destination} records={len(dataset['records'])} unmatched={len(dataset['unmatched'])}")
    print({name: len(ids) for name, ids in dataset["splits"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
