from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction.catalog import build_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalog reconstructable scans with zSpacing filtering.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--z-spacing", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    catalog = build_catalog(args.source_root, z_spacing=args.z_spacing, ratios=tuple(args.ratios), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: raw_dirs={catalog['stats']['raw_dirs']} ready={catalog['stats']['ready']} excluded={catalog['stats']['excluded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
