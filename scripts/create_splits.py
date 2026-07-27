from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.data.manifest import create_case_splits, save_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create case-level train/val/test splits from pair.json files.")
    parser.add_argument("pairs_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.pairs_root.glob("*/pair.json"))]
    save_manifest(create_case_splits(records, tuple(args.ratios), args.seed), args.output)
    print(f"Wrote {args.output} with {len(records)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
