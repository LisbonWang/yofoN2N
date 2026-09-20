"""Batch CBCT reconstruction of PDL studies into 6 output variants per recon group.

Each study (raw projections + config) is reconstructed once per matching *recon group*
(see ``configs/recon_groups.json``): a group lists the scan modes it applies to, the
fdkPara.xml geometry to overwrite, and the *standard* filter.xml file to use (the file is
copied verbatim, no runtime parameter overrides), so the same scan can be rebuilt at
multiple FOVs.  A Mode2 study, for example, is rebuilt both as ``Mode2_1609`` (native) and
``Mode2_0808`` (patched to an 80x80mm volume)::

    <output_root>/<source>/<group>/<case_id>/<study_uid>/{even,odd,full}.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/full.{marfi,bhc,marfi.bhc}.raw

The top-level ``<source>`` directory is the source root name (``PDL`` by default).  Any
``Mode<N>`` is accepted (Mode0 included); modes without a matching group are reported as
skipped.

Raw and config are first copied to a temp workspace (deleted when the study finishes);
even/odd subsets slice the projections and per-view binaries and update ``noView``.  The
group's fdkPara geometry and standard filter file are applied to every subset's config
copy.  For downsampled subsets the MARFI/BHC ``step`` parameter is rounded up to an even
value when it is > 1 (see ``fix_subsample_step``).

Examples
--------
Dry-run discovery::

    python scripts\\reconstruct_pdl_variants.py --dry-run

Real batch::

    python scripts\\reconstruct_pdl_variants.py

Only one group::

    python scripts\\reconstruct_pdl_variants.py --groups Mode2_0808
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import (
    DEFAULT_COMBOS,
    DEFAULT_PER_VIEW_FILES,
    DEFAULT_SPLIT_MODES,
    DEFAULT_STEP_MODULES,
    OUTPUT_NAMES,
    VariantOptions,
    discover_studies,
    groups_for_mode,
    load_index,
    load_recon_groups,
    merge_index_records,
    output_names_for_mode,
    reconstruct_study,
    validate_study_config,
    verify_volumes,
)

INDEX_NAME = "reconstruction-index.json"
DEFAULT_GROUPS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "recon_groups.json"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch CBCT reconstruction: 6 output variants per study, grouped by <Mode>_<FOV>.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path,
                        default=r"\\192.168.36.235\团队文件-牙科ct\data_raw\PDL",
                        help="Root containing <case>/<study>/<series>_ModeX_*kV*mA* raw dirs and config/")
    parser.add_argument("--output-root", type=Path,
                        default=r"Z:\raid0\yyc\AIReduceNoise",
                        help="Where <source>/<Mode>_<FOV>/<case_id>/<study_uid>/*.raw volumes are written "
                             "(top-level <source> defaults to the source root name, e.g. PDL)")
    parser.add_argument("--xday-dir", type=Path, default=Path(os.environ.get("XDAY_DIR", "D:\\XDay Release\\XDay617.2")),
                        help="XDay SDK directory (xday.dll + dependencies)")
    parser.add_argument("--combo", action="append", metavar="TAG=CHAIN", default=None,
                        help=f"Post-processing combo; repeatable. Default: {'; '.join(f'{t}={c!r}' for t, c in DEFAULT_COMBOS)}")
    parser.add_argument("--dtype", default="float32", choices=("int16", "uint16", "float32"))
    parser.add_argument("--link-mode", default="copy", choices=("auto", "copy", "hardlink"),
                        help="How raw projections are placed into the temp workspace")
    parser.add_argument("--per-view-file", action="append", metavar="NAME=BYTES", default=None,
                        help="Ancillary per-view binary width to slice in subsets; repeatable. "
                             "Default by mode: Mode1/Mode2/Mode3 projectionMatrix.bin=84; "
                             "modes without an entry (e.g. Mode0) copy such files whole")
    parser.add_argument("--step-modules", default=",".join(DEFAULT_STEP_MODULES),
                        help="Comma-separated modules whose step must stay even for downsampled subsets")
    parser.add_argument("--split-modes", default=",".join(DEFAULT_SPLIT_MODES),
                        help="Comma-separated modes whose scans are split into complementary subsets. "
                             "Mode1 uses two-frame alternating source blocks; Mode2/Mode3 use scalar parity. "
                             "Modes not listed (e.g. Mode0) produce full volumes only")
    parser.add_argument("--mode", default=None, help="Only reconstruct studies of this mode, e.g. Mode2")
    parser.add_argument("--groups-config", type=Path, default=DEFAULT_GROUPS_CONFIG,
                        help="JSON file defining the reconstruction groups (source modes + fdkPara geometry + standard filter file)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated subset of group names to reconstruct (default: all groups matching each study's mode)")
    parser.add_argument("--kv", type=float, default=None, help="Override tube kV for all studies")
    parser.add_argument("--ma", type=float, default=None, help="Override tube mA for all studies")
    parser.add_argument("--pwls-scale", type=float, default=0.01,
                        help="Data2-style per-study PWLS patch: beta = scale * mean(fieldMatrixMax.bin), "
                             "PWLS forced on and Hamming window (0.54/0.46) enabled in every subset's filter.xml")
    parser.add_argument("--no-pwls", action="store_true",
                        help="Disable the PWLS+Hamming patch (data1-style: standard filter used verbatim)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many studies (useful for smoke tests)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Exclude a (study_uid, group) work item: repeat the flag as "
                             "'<study_uid>:<group>' or just '<study_uid>' to drop every group of that study. "
                             "Use for items that hang the engine (e.g. corrupt/mis-calibrated projection data)")
    parser.add_argument("--overwrite", action="store_true", help="Reconstruct studies even if all outputs already exist")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going after a failing study")
    parser.add_argument("--retries", type=int, default=2,
                        help="Per-study retry count before it is reported as an error (default: 2)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip checking every output volume's file size after reconstruction")
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
    step_modules = tuple(m.strip() for m in args.step_modules.split(",") if m.strip())
    split_modes = tuple(m.strip() for m in args.split_modes.split(",") if m.strip()) or DEFAULT_SPLIT_MODES
    groups = load_recon_groups(args.groups_config)
    selected_names = set(args.groups.split(",")) if args.groups else None
    if selected_names is not None:
        missing = selected_names - set(groups)
        if missing:
            raise SystemExit(f"unknown groups: {sorted(missing)}; available: {sorted(groups)}")
        groups = {name: g for name, g in groups.items() if name in selected_names}

    studies, skipped = discover_studies(
        args.source_root,
        mode=args.mode,
        kv=args.kv,
        ma=args.ma,
    )
    if args.limit is not None:
        studies = studies[: args.limit]

    excluded = set(args.exclude)
    # (study, group) work items; studies whose mode matches no group are reported as skipped
    work: list[tuple[Any, Any]] = []
    for study in studies:
        matched = groups_for_mode(groups, study.mode)
        if not matched:
            skipped.append({"case_id": study.case_id, "study_uid": study.study_uid,
                            "reason": f"mode {study.mode} matches no recon group"})
            continue
        for group in matched:
            if study.study_uid in excluded or f"{study.study_uid}:{group.name}" in excluded:
                skipped.append({"case_id": study.case_id, "study_uid": study.study_uid,
                                "group": group.name, "reason": "excluded via --exclude (engine hang)"})
                print(f"{study.case_id}/{study.study_uid} [{group.name}]: SKIPPED excluded "
                      "(engine hang)", flush=True)
                continue
            work.append((study, group))

    # Drop studies whose config is an incomplete copy: the engine dies with an opaque
    # access violation on blank current.xml / filterPara.xml instead of failing cleanly.
    checked: dict[str, list[str]] = {}
    usable: list[tuple[Any, Any]] = []
    for study, group in work:
        key = str(study.cfg_dir)
        if key not in checked:
            checked[key] = validate_study_config(study.cfg_dir)
        if checked[key]:
            continue
        usable.append((study, group))
    for study, _group in work:
        problems = checked[str(study.cfg_dir)]
        if problems and not any(s.get("study_uid") == study.study_uid for s in skipped):
            skipped.append({"case_id": study.case_id, "study_uid": study.study_uid,
                            "mode": study.mode, "cfg_dir": str(study.cfg_dir),
                            "reason": "corrupt config: " + "; ".join(problems)})
            print(f"{study.case_id}/{study.study_uid} [{study.mode}]: SKIPPED corrupt config "
                  f"({'; '.join(problems)})", flush=True)
    work = usable

    print(f"Discovered {len(studies)} study(ies) -> {len(work)} (study, group) item(s), "
          f"skipped {len(skipped)} without usable config/kV-mA or matching group", flush=True)
    top_dir = Path(args.source_root).name or "SOURCE"
    for study, group in work:
        print(f"  {top_dir}/{group.name:<14} {study.case_id}/{study.study_uid}\t{study.mode}\t{study.nviews} views\t"
              f"{study.kv}kV {study.ma}mA\t{study.raw_dir.name}", flush=True)
    if args.dry_run:
        volumes = sum(len(output_names_for_mode(study.mode, combos, split_modes)) for study, _group in work)
        print(f"Dry run: {len(work)} (study, group) item(s) x per-mode output sets = "
              f"{volumes} volumes would be reconstructed.", flush=True)
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
        step_modules=step_modules,
        split_modes=split_modes,
        kv=args.kv,
        ma=args.ma,
        overwrite=args.overwrite,
        keep_workspace=args.keep_workspace,
        workspace_root=args.workspace_root,
        top_dir=top_dir,
        pwls_scale=None if args.no_pwls else args.pwls_scale,
    )

    records, failures = [], 0
    started = time.monotonic()
    total = len(work)
    for i, (study, group) in enumerate(work, 1):
        record = None
        last_error: str | None = None
        for attempt in range(args.retries + 1):
            try:
                record = reconstruct_study(study, group, output_root, options, sdk=sdk)
                last_error = None
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    print(f"  retry {attempt + 1}/{args.retries} after {last_error}", flush=True)
        if record is None:
            failures += 1
            records.append({"case_id": study.case_id, "study_uid": study.study_uid, "scan_id": study.scan_id,
                            "mode": study.mode, "group": group.name, "status": "error", "error": last_error})
            print(f"{study.case_id}/{study.study_uid} [{group.name}]: ERROR {last_error}", flush=True)
            if not args.continue_on_error:
                break
            continue
        records.append(record)
        issues = [] if args.no_verify else verify_volumes(record, output_root)
        if issues:
            failures += 1
            record["status"] = "verify_failed"
            record["verify_issues"] = issues
            print(f"{study.case_id}/{study.study_uid} [{group.name}]: VERIFY FAILED {issues}", flush=True)
        else:
            state = "skipped (complete)" if record.get("skipped") else "ok"
            print(f"{study.case_id}/{study.study_uid} [{group.name}]: {state}", flush=True)
        elapsed = time.monotonic() - started
        per_item = elapsed / i
        remaining = max(0, (total - i) * per_item)
        print(f"  [{i}/{total}] {elapsed / 60:.1f} min elapsed, "
              f"{per_item:.1f} s/item, ETA {remaining / 60:.1f} min", flush=True)

    existing = load_index(index_path) if (index_path := output_root / INDEX_NAME).is_file() else None
    merged_records = merge_index_records(existing.get("records", []) if existing else [], records)
    index = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "groups_config": str(args.groups_config),
        "combos": [{"tag": tag, "post_procs": chain} for tag, chain in combos],
        "pwls": {"enabled": not args.no_pwls, "scale": None if args.no_pwls else args.pwls_scale},
        "records": merged_records,
        "skipped": skipped,
    }
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(index_path)
    ok = sum(1 for r in records if r.get("status") == "success")
    verify_failed = sum(1 for r in records if r.get("status") == "verify_failed")
    print(f"Wrote {index_path}: total_records={len(merged_records)} this_run: success={ok} "
          f"verify_failed={verify_failed} error={failures} skipped_studies={len(skipped)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
