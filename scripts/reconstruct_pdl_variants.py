"""Batch CBCT reconstruction of PDL studies into 12 variants each.

Each study (raw projections + config) is reconstructed as full / even / odd subsets and,
for every subset, with the four MARFI/BHC post-processing combos, yielding 12 volumes::

    <output_root>/<case_id>/<study_uid>/{full,even,odd}_{none,bhc,marfi,bhc_marfi}.raw

Raw and config are first copied to a temp workspace; even/odd subsets slice the projections
and per-view binaries and update ``noView``.  For downsampled subsets the MARFI/BHC ``step``
parameter is rounded up to an even value when it is > 1 (see ``fix_subsample_step``).

Examples
--------
Dry-run discovery::

    python scripts\\reconstruct_pdl_variants.py --dry-run

Real batch::

    python scripts\\reconstruct_pdl_variants.py

With extra filter.xml overrides (repeatable)::

    python scripts\\reconstruct_pdl_variants.py --filter-set MARFI.step=2 --filter-set BHC.boneThresLow=0.25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import (
    DEFAULT_COMBOS,
    DEFAULT_PER_VIEW_FILES,
    DEFAULT_STEP_MODULES,
    SUBSETS,
    VariantOptions,
    discover_studies,
    reconstruct_study,
)

INDEX_NAME = "reconstruction-index.json"


def _parse_combo(spec: str) -> tuple[str, str]:
    tag, sep, chain = spec.partition("=")
    if not sep:
        raise SystemExit(f"--combo must be TAG=CHAIN, got {spec!r}")
    return tag, chain.strip()


def _parse_per_view(spec: str) -> tuple[str, int]:
    try:
        name, width = spec.rsplit("=", 1)
        return name, int(width)
    except ValueError as exc:
        raise SystemExit(f"--per-view-file must be NAME=BYTES, got {spec!r}") from exc


def _parse_filter_set(spec: str) -> tuple[str, str]:
    key, sep, value = spec.partition("=")
    if not sep:
        raise SystemExit(f"--filter-set must be MODULE.FIELD=VALUE, got {spec!r}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch CBCT reconstruction: 3 subsets x 4 MARFI/BHC combos = 12 volumes per study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path,
                        default=r"\\192.168.36.235\团队文件-牙科ct\data_raw\PDL",
                        help="Root containing <case>/<study>/<series>_ModeX_*kV*mA* raw dirs and config/")
    parser.add_argument("--output-root", type=Path,
                        default=r"Z:\raid5\data_tmp\AIReduceNoise",
                        help="Where <case_id>/<study_uid>/*.raw volumes are written")
    parser.add_argument("--xday-dir", type=Path, default=Path(os.environ.get("XDAY_DIR", "D:\\XDay Release\\XDay617.2")),
                        help="XDay SDK directory (xday.dll + dependencies)")
    parser.add_argument("--combo", action="append", metavar="TAG=CHAIN", default=None,
                        help=f"Post-processing combo; repeatable. Default: {'; '.join(f'{t}={c!r}' for t, c in DEFAULT_COMBOS)}")
    parser.add_argument("--dtype", default="float32", choices=("int16", "uint16", "float32"))
    parser.add_argument("--link-mode", default="copy", choices=("auto", "copy", "hardlink"),
                        help="How raw projections are placed into the temp workspace")
    parser.add_argument("--per-view-file", action="append", metavar="NAME=BYTES", default=None,
                        help="Ancillary per-view binary width to slice in subsets; repeatable. "
                             "Default by mode: Mode2/Mode3 projectionMatrix.bin=84, Mode1 none")
    parser.add_argument("--filter-set", action="append", metavar="MODULE.FIELD=VALUE", default=None,
                        help="Override a filter.xml parameter in every temp config; repeatable, e.g. MARFI.step=2")
    parser.add_argument("--step-modules", default=",".join(DEFAULT_STEP_MODULES),
                        help="Comma-separated modules whose step must stay even for downsampled subsets")
    parser.add_argument("--mode", default=None, help="Only reconstruct studies of this mode, e.g. Mode2")
    parser.add_argument("--kv", type=float, default=None, help="Override tube kV for all studies")
    parser.add_argument("--ma", type=float, default=None, help="Override tube mA for all studies")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many studies (useful for smoke tests)")
    parser.add_argument("--overwrite", action="store_true", help="Reconstruct studies even if all outputs already exist")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going after a failing study")
    parser.add_argument("--dry-run", action="store_true", help="Discover studies and print the plan without reconstructing")
    parser.add_argument("--keep-workspace", action="store_true",
                        help="Keep each study's temp workspace under <output>/<case>/<study>/workspace for debugging")
    parser.add_argument("--workspace-root", type=Path, default=None,
                        help="Base directory for temp workspaces (default: system temp)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    combos = tuple(_parse_combo(spec) for spec in args.combo) if args.combo else DEFAULT_COMBOS
    if not combos:
        raise SystemExit("at least one --combo is required")
    for tag, _chain in combos:
        if not tag.isidentifier() and not all(c.isalnum() or c in "-_" for c in tag):
            raise SystemExit(f"invalid combo tag {tag!r}; use letters, digits, '-' or '_'")
    per_view_files = {mode: dict(files) for mode, files in DEFAULT_PER_VIEW_FILES.items()}
    for spec in args.per_view_file or []:
        name, width = _parse_per_view(spec)
        for mode_files in per_view_files.values():
            mode_files[name] = width
    filter_set = dict(_parse_filter_set(spec) for spec in args.filter_set or [])
    step_modules = tuple(m.strip() for m in args.step_modules.split(",") if m.strip())

    studies, skipped = discover_studies(
        args.source_root,
        mode=args.mode,
        kv=args.kv,
        ma=args.ma,
    )
    if args.limit is not None:
        studies = studies[: args.limit]

    print(f"Discovered {len(studies)} study(ies), skipped {len(skipped)} without usable config/kV-mA", flush=True)
    for study in studies:
        print(f"  {study.case_id}/{study.study_uid}\t{study.mode}\t{study.nviews} views\t{study.kv}kV {study.ma}mA\t{study.raw_dir.name}", flush=True)
    if args.dry_run:
        print(f"Dry run: {len(studies)} study(ies) x {len(combos)} combos x {len(SUBSETS)} subsets = "
              f"{len(studies) * len(combos) * len(SUBSETS)} volumes would be reconstructed.", flush=True)
        return 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from xday import load_xday
    except ImportError as exc:
        raise SystemExit("xday-kits is required for reconstruction; install it in the torch environment") from exc
    sdk = load_xday(args.xday_dir)
    options = VariantOptions(
        xday_dir=args.xday_dir,
        combos=combos,
        dtype=args.dtype,
        link_mode=args.link_mode,
        per_view_files=per_view_files,
        filter_set=filter_set,
        step_modules=step_modules,
        kv=args.kv,
        ma=args.ma,
        overwrite=args.overwrite,
        keep_workspace=args.keep_workspace,
        workspace_root=args.workspace_root,
    )

    records, failures = [], 0
    for study in studies:
        try:
            record = reconstruct_study(study, output_root, options, sdk=sdk)
            records.append(record)
            state = "skipped (complete)" if record.get("skipped") else "ok"
            print(f"{study.case_id}/{study.study_uid}: {state}", flush=True)
        except Exception as exc:
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            records.append({"case_id": study.case_id, "study_uid": study.study_uid, "scan_id": study.scan_id,
                            "mode": study.mode, "status": "error", "error": error})
            print(f"{study.case_id}/{study.study_uid}: ERROR {error}", flush=True)
            if not args.continue_on_error:
                break

    index = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "combos": [{"tag": tag, "post_procs": chain} for tag, chain in combos],
        "records": records,
        "skipped": skipped,
    }
    index_path = output_root / INDEX_NAME
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(index_path)
    ok = sum(1 for r in records if r.get("status") == "success")
    print(f"Wrote {index_path}: success={ok} error={failures} skipped_studies={len(skipped)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
