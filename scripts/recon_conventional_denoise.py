"""Rebuild one study through the scanner's *conventional* (non-AI) denoising chain.

The AIReduceNoise data set stores four post-processing variants of every study
(``full``, ``full.bhc``, ``full.marfi``, ``full.marfi.bhc``).  None of them switches on
the vendor's own denoising filters: ``ReduceNoise`` and ``Anisotropic`` are declared in
the group's standard ``filter.xml`` but are only executed when they are explicitly
requested as post-processing modules -- exactly like ``BHC`` / ``MARFI``.

The rebuild goes through the *same* reconstruction group as the batch pipeline (same
fdkPara geometry, same full projection set), with ``--post-procs`` chosen by the caller.
The default chain ``"BHC ReduceNoise Anisotropic"`` keeps the ``BHC`` that the
``full.bhc`` AI input already carries and switches on both conventional denoisers, so
the result is a like-for-like non-AI reference against the denoised volumes.

One study's raw projections and config are copied to a temporary workspace first, so
nothing in the source data set or in ``configs/recon_filters/`` is modified.

Parameter sweeps go through ``--filter-file``: point it at a *copy* of the group's
standard ``filter.xml`` (the repo ships ``Mode2_1609.convdenoise.filter.xml`` for this)
and edit the copy's ``ReduceNoise`` / ``Anisotropic`` blocks.  The group file that the
batch pipeline references is never touched.  The sidecar JSON records the filter file's
md5 plus a snapshot of both blocks, so every swept volume stays traceable to the exact
parameter values that produced it.

Example::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/recon_conventional_denoise.py \
      --case YPDL-26A-0027 \
      --study 1.2.156.89797079.1182197106.1232956091.447186561 \
      --group Mode2_1609 \
      --out outputs/air3_ypdl0027_bhc_5models/full.bhc.conv_rn_ani.raw

    # same study, but rebuilt from an edited copy of the filter file
    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/recon_conventional_denoise.py \
      --case YPDL-26A-0027 \
      --study 1.2.156.89797079.1182197106.1232956091.447186561 \
      --group Mode2_1609 \
      --filter-file configs/recon_filters/Mode2_1609.convdenoise.filter.xml \
      --pwls-scale 0.01 \
      --out outputs/air3_ypdl0027_bhc_5models/full.bhc.conv_rn_ani.raw --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import (
    Study,
    apply_fdk_para,
    apply_filter_file,
    brightness_mean,
    build_full_workspace,
    discover_studies,
    load_recon_groups,
    patch_pwls_hamming,
    validate_study_config,
)
from cbct_n2n.reconstruction.variants import (
    _find_configs,
    _has_raw_files,
    _MODE_NAME,
    _parse_kv_ma,
    _raw_file_count,
)

DEFAULT_SOURCE_ROOT = r"\\192.168.36.235\团队文件-牙科ct\data_raw\PDL"
DEFAULT_GROUPS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "recon_groups.json"
DEFAULT_POST_PROCS = "BHC ReduceNoise Anisotropic"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT),
                        help="Root containing <case>/<study>/<series>_ModeX_* raw dirs and config/")
    parser.add_argument("--case", required=True, help="case id, e.g. YPDL-26A-0027")
    parser.add_argument("--study", required=True, help="study uid (the directory under <case>/)")
    parser.add_argument("--group", default="Mode2_1609",
                        help="reconstruction group name from --groups-config")
    parser.add_argument("--groups-config", type=Path, default=DEFAULT_GROUPS_CONFIG)
    parser.add_argument("--post-procs", default=DEFAULT_POST_PROCS,
                        help="space-separated XDay post-processing chain")
    parser.add_argument("--filter-file", type=Path, default=None,
                        help="use this filter.xml instead of the group's standard file "
                             "(whole-file replacement, so it must be a complete config).  "
                             "Use it to sweep ReduceNoise / Anisotropic parameters without "
                             "touching configs/recon_filters/<group>.filter.xml, which the "
                             "batch pipeline depends on.")
    parser.add_argument("--out", type=Path, required=True, help="destination .raw file")
    parser.add_argument("--xday-dir", type=Path,
                        default=Path(os.environ.get("XDAY_DIR", r"D:\XDay Release\XDay 617.2_polyc")))
    parser.add_argument("--dtype", default="float32", choices=("int16", "uint16", "float32"))
    parser.add_argument("--pwls-scale", type=float, default=None,
                        help="Enable the data2-style PWLS patch: beta = SCALE * mean(fieldMatrixMax*.bin) "
                             "plus the Hamming window (0.54/0.46).  AIReduceNoise3 was built with 0.01 -- "
                             "pass the same value here or the comparison baseline will not match the AI input.")
    parser.add_argument("--keep-workspace", action="store_true",
                        help="keep the temporary raw/config copy for inspection")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    return parser.parse_args(argv)


def _probe_study(source_root: Path, case_id: str, study_uid: str) -> Study | None:
    """Look the study up directly, without walking the whole source tree.

    ``discover_studies`` scans every case and study on the network share (~90 s); a
    single-study run only needs ``<root>/<case>/<study>``.  The same private helpers the
    batch discovery uses are reused here so the resulting :class:`Study` is built by
    identical rules (kv/mA parsing, raw/config detection, effective view count).
    """
    study_dir = source_root / case_id / study_uid
    if not study_dir.is_dir():
        return None
    raw_dirs = [Path(e.path) for e in os.scandir(study_dir) if e.is_dir() and _has_raw_files(Path(e.path))]
    configs = _find_configs(study_dir)
    if len(raw_dirs) != 1 or len(configs) != 1:
        return None
    raw_dir, cfg_dir = raw_dirs[0], configs[0]
    raw_match = _MODE_NAME.search(raw_dir.name)
    cfg_match = _MODE_NAME.search(cfg_dir.name)
    if raw_match is None or cfg_match is None:
        return None
    mode = raw_match.group(1)
    if mode.casefold() != cfg_match.group(1).casefold():
        return None
    try:
        kv, ma = _parse_kv_ma(raw_dir.name)
    except ValueError:
        return None
    return Study(
        case_id=case_id,
        study_uid=study_uid,
        scan_id=study_uid,
        raw_dir=raw_dir,
        cfg_dir=cfg_dir,
        mode=mode,
        kv=kv,
        ma=ma,
        nviews=_raw_file_count(raw_dir),
    )


def _find_study(source_root: Path, case_id: str, study_uid: str) -> Study:
    """Locate the (raw, config) pair of one study, whichever Mode it carries."""
    probed = _probe_study(source_root, case_id, study_uid)
    if probed is not None:
        print(f"located study directly in {probed.raw_dir}")
        return probed

    print("direct probe inconclusive; falling back to a full source-tree scan (~90 s)")
    studies, _skipped = discover_studies(source_root)
    matches = [s for s in studies if s.case_id == case_id and s.study_uid == study_uid]
    if not matches:
        raise SystemExit(f"study not found: {case_id}/{study_uid} under {source_root}")
    if len(matches) > 1:
        modes = sorted(s.mode for s in matches)
        raise SystemExit(f"{case_id}/{study_uid} matched {len(matches)} raw dirs: {modes}")
    return matches[0]


def _file_md5(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_snapshot(path: Path, blocks: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Pull named top-level blocks out of a filter.xml as ``{key: value}`` maps.

    A real filter.xml is *not* standard XML -- ``<SecondaryBP::BHC>`` has a double
    colon, which every conforming parser rejects -- so the blocks are pulled out with
    a regular expression instead of an XML parser.  Comments are stripped first so an
    annotated block (like the TUNABLE markers in the sweep copy) cannot match.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    snapshot: dict[str, dict[str, str]] = {}
    for block in blocks:
        match = re.search(rf"<{block}>(.*?)</{block}>", text, flags=re.DOTALL)
        if match is None:
            snapshot[block] = {}
            continue
        snapshot[block] = {
            key: value.strip()
            for key, value in re.findall(r"<(\w+)>(.*?)</\1>", match.group(1), flags=re.DOTALL)
        }
    return snapshot


def _denoise_params_snapshot(path: Path) -> dict[str, dict[str, str]]:
    """Snapshot the ``ReduceNoise`` / ``Anisotropic`` blocks of a filter.xml.

    The whole file is what the SDK consumes, but these two blocks are the ones that
    get swept when tuning the conventional denoiser, so they are copied into the
    sidecar verbatim.  That way a rebuilt volume can always be traced back to the
    exact parameter values that produced it.
    """
    return _block_snapshot(path, ("ReduceNoise", "Anisotropic"))


def _flag(value: str | None) -> bool:
    """Interpret an XDay boolean-ish field (``isEnable`` and friends)."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out.is_file() and not args.force:
        print(f"skip: {args.out} exists (use --force)")
        return 0

    groups = load_recon_groups(args.groups_config)
    if args.group not in groups:
        raise SystemExit(f"unknown group {args.group!r}; available: {sorted(groups)}")
    group = groups[args.group]

    study = _find_study(args.source_root, args.case, args.study)
    if not group.applies_to(study.mode):
        raise SystemExit(f"group {group.name} does not apply to {study.mode}")

    filter_file = args.filter_file if args.filter_file is not None else group.filter_file
    if filter_file is None:
        raise SystemExit(f"group {group.name} has no standard filter.xml and --filter-file was not given")
    filter_path = Path(filter_file)
    if not filter_path.is_file():
        raise SystemExit(f"filter.xml not found: {filter_path}")

    warnings = validate_study_config(study.cfg_dir)
    for warning in warnings:
        print(f"[config warning] {warning}")

    print(f"case       : {study.case_id}")
    print(f"study      : {study.study_uid}")
    print(f"mode       : {study.mode}  {study.kv} kV / {study.ma} mA / {study.nviews} views")
    print(f"raw dir    : {study.raw_dir}")
    print(f"cfg dir    : {study.cfg_dir}")
    print(f"group      : {group.name}  fdk_para={group.fdk_para}")
    print(f"filter.xml : {filter_path}"
          f"{'  (custom --filter-file)' if args.filter_file is not None else '  (group standard)'}")
    print(f"post-procs : {args.post_procs!r}")
    print(f"dtype      : {args.dtype}")

    workspace = Path(tempfile.mkdtemp(prefix=f"convdenoise_{study.case_id}_"))
    try:
        full_raw, full_cfg = build_full_workspace(
            study.raw_dir, study.cfg_dir, workspace / "full", "copy",
            max_views=study.nviews,
        )
        changed = apply_fdk_para(full_cfg, group.fdk_para)
        replaced = apply_filter_file(full_cfg, filter_path)
        filter_md5 = _file_md5(filter_path)
        denoise_params = _denoise_params_snapshot(filter_path)
        print(f"fdkPara fields changed: {changed}")
        print(f"filter.xml replaced   : {replaced}")
        print(f"filter.xml md5        : {filter_md5}")
        for block, values in denoise_params.items():
            print(f"  {block:12s}: {values}")

        pwls_info = None
        if args.pwls_scale is not None:
            mean = brightness_mean(study.cfg_dir)
            beta = args.pwls_scale * mean
            notes = patch_pwls_hamming(full_cfg, beta)
            pwls_info = {"enabled": True, "scale": args.pwls_scale, "beta": beta,
                         "brightness_mean": mean, "source": "cli_patch"}
            print(f"pwls        : beta={beta:g} (scale {args.pwls_scale:g} x brightness {mean:.4f})")
            for note in notes:
                print(f"  {note}")
        else:
            # No patch: whatever the filter.xml itself says decides whether Pwls runs.
            # Report it verbatim -- assuming "off" because no --pwls-scale was passed is
            # wrong as soon as someone flips <Pwls><isEnable> by hand in the sweep copy.
            block = _block_snapshot(filter_path, ("Pwls",))["Pwls"]
            enabled = _flag(block.get("isEnable"))
            pwls_info = {"enabled": enabled, "source": "filter_file", "filter_block": block}
            if enabled:
                print(f"pwls        : enabled by {filter_path.name} "
                      f"beta={block.get('beta', '?')} (no --pwls-scale -> no patch, "
                      f"Ramp left as written in the file)")
            else:
                print("pwls        : disabled (filter.xml isEnable=0 and no --pwls-scale)")

        from xday import invoke_xday

        started = time.time()
        result = invoke_xday(
            full_raw, full_cfg, study.kv, study.ma,
            post_procs=args.post_procs,
            xday_install_dir=args.xday_dir,
            dtype=args.dtype,
            show_progress=False,
        )
        elapsed = time.time() - started

        volume = result.volume
        shape = tuple(int(v) for v in volume.shape)
        info = result.volume_info
        spacing = [float(info.z_spacing), float(info.y_spacing), float(info.x_spacing)]
        print(f"wrote shape={shape} spacing_zyx={spacing} in {elapsed:.1f}s")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        temporary = args.out.with_name(args.out.name + ".tmp")
        np.asarray(volume, dtype="<f4").tofile(temporary)
        os.replace(temporary, args.out)

        expected = int(np.prod(shape)) * 4
        actual = args.out.stat().st_size
        if actual != expected:
            raise SystemExit(f"size mismatch: {args.out} is {actual} B, expected {expected} B")

        record = {
            "output": str(args.out),
            "bytes": actual,
            "md5": _file_md5(args.out),
            "shape": list(shape),
            "dtype": "<f4",
            "spacing_zyx": spacing,
            "case_id": study.case_id,
            "study_uid": study.study_uid,
            "mode": study.mode,
            "group": group.name,
            "kv": study.kv,
            "ma": study.ma,
            "nviews": study.nviews,
            "raw_dir": str(study.raw_dir),
            "cfg_dir": str(study.cfg_dir),
            "fdk_para": dict(group.fdk_para),
            "filter_file": str(filter_path),
            "filter_file_md5": filter_md5,
            "filter_file_is_custom": args.filter_file is not None,
            "filter_file_group_standard": group.filter_file,
            "denoise_params": denoise_params,
            "filter_replaced": replaced,
            "post_procs": args.post_procs,
            "pwls": pwls_info,
            "dtype_requested": args.dtype,
            "seconds": elapsed,
            "config_warnings": warnings,
            "volume_min": float(np.min(volume)),
            "volume_max": float(np.max(volume)),
            "volume_mean": float(np.mean(volume)),
            "volume_std": float(np.std(volume)),
        }
        sidecar = args.out.with_name(args.out.name + ".json")
        sidecar.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}  ({actual} B, md5 {record['md5']})")
        print(f"wrote {sidecar}")
        return 0
    finally:
        if args.keep_workspace:
            print(f"workspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
