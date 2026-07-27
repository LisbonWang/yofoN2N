from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import PrepareOptions, discover_cases, prepare_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare odd/even XDay CBCT reconstruction pairs.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--xday-dir", type=Path, default=os.environ.get("XDAY_DIR"))
    parser.add_argument("--post-procs", default="")
    parser.add_argument("--dtype", choices=["float32", "int16", "uint16"], default="float32")
    parser.add_argument("--link-mode", choices=["auto", "copy", "hardlink"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--mode", help="Only process cases with this acquisition mode")
    parser.add_argument("--per-view-file", action="append", metavar="NAME=BYTES", help="Explicitly slice an ancillary per-view binary file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = discover_cases(args.source_root, continue_on_error=args.continue_on_error, mode=args.mode)
    for case in cases:
        print(f"{case.case_id}\t{case.mode}\t{case.raw_dir}\t{case.cfg_dir}")
    if args.dry_run:
        mode_label = f" {args.mode}" if args.mode else ""
        print(f"Discovered {len(cases)}{mode_label} case(s); no reconstruction performed.")
        return 0
    if args.xday_dir is None:
        raise SystemExit("--xday-dir or XDAY_DIR is required")
    per_view_files = {"projectionMatrix.bin": 84}
    for item in args.per_view_file or []:
        try:
            name, width = item.rsplit("=", 1)
            per_view_files[name] = int(width)
        except ValueError as exc:
            raise SystemExit("--per-view-file must use NAME=BYTES") from exc
    options = PrepareOptions(args.xday_dir, args.post_procs, args.dtype, args.link_mode, args.overwrite, args.keep_workspace, per_view_files)
    failures = 0
    for case in cases:
        try:
            record = prepare_case(case, args.output_root, options)
            print(f"Wrote {record.metadata_path}")
        except Exception as exc:
            failures += 1
            if not args.continue_on_error:
                raise
            print(f"ERROR {case.case_id}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
