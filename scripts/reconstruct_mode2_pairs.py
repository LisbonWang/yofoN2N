from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.config import load_config
from cbct_n2n.reconstruction import PrepareOptions, discover_cases, prepare_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct Mode2 CBCT even/odd/full pairs from configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--per-view-file", action="append", metavar="NAME=BYTES", help="Override an ancillary per-view binary file width")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reconstruction = load_config(args.config)["reconstruction"]
    source_root, output_root = Path(reconstruction["source_root"]), Path(reconstruction["output_root"])
    cases = discover_cases(source_root, continue_on_error=args.continue_on_error, mode="Mode2")
    for case in cases:
        print(f"{case.case_id}\t{case.mode}\t{case.raw_dir}\t{case.cfg_dir}")
    if args.dry_run:
        print(f"Discovered {len(cases)} Mode2 case(s); no reconstruction performed.")
        return 0
    per_view_files = dict(reconstruction.get("per_view_files", {}))
    for item in args.per_view_file or []:
        try:
            name, width = item.rsplit("=", 1)
            per_view_files[name] = int(width)
        except ValueError as exc:
            raise SystemExit("--per-view-file must use NAME=BYTES") from exc
    options = PrepareOptions(
        xday_dir=Path(reconstruction["xday_dir"]),
        post_procs=str(reconstruction.get("post_procs", "")),
        dtype=str(reconstruction.get("dtype", "float32")),
        link_mode=str(reconstruction.get("link_mode", "auto")),
        overwrite=args.overwrite,
        keep_workspace=args.keep_workspace,
        per_view_files=per_view_files,
    )
    failures = 0
    for case in cases:
        try:
            record = prepare_case(case, output_root, options)
            print(f"Wrote {record.metadata_path}")
        except Exception as exc:
            failures += 1
            if not args.continue_on_error:
                raise
            print(f"ERROR {case.case_id}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
