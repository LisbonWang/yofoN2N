"""AIReduceNoise2 batch reconstruction: 6 output variants per 1609 study,
with per-study PWLS beta = BRIGHTNESS_SCALE * mean(fieldMatrixMax.bin) and a
Hamming window forced on in the Ramp section.

Mirrors cbct_n2n.reconstruction.reconstruct_study but, instead of replacing
filter.xml with a group-level template, it patches every workspace's filter.xml
copy in place (source configs on the share are never touched):

  <Pwls>  isEnable -> 1, beta -> scale * brightness-of-this-study
  <Ramp>  isHamEnable -> 1, paraHam1 -> 0.54, paraHam2 -> 0.46   (Hamming)

Output layout mirrors AIReduceNoise:
  <output_root>/PDL/<group>/<case_id>/<study_uid>/{even,odd,full,
  full.bhc,full.marfi,full.marfi.bhc}.raw + metadata.json

Usage (Windows box with XDay SDK + GPU):
  python scripts/reconstruct_aired2_pwls.py --dry-run
  python scripts/reconstruct_aired2_pwls.py --limit 1
  python scripts/reconstruct_aired2_pwls.py --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import (  # noqa: E402
    DEFAULT_COMBOS,
    DEFAULT_PER_VIEW_FILES,
    DEFAULT_SPLIT_MODES,
    DEFAULT_STEP_MODULES,
    OUTPUT_NAMES,
    load_index,
    merge_index_records,
    output_names_for_mode,
    split_projection_indices,
    validated_per_view,
    verify_volumes,
)
from cbct_n2n.reconstruction.variants import (  # noqa: E402
    apply_fdk_para,
    build_full_workspace,
    build_subset_workspace,
    fix_subsample_step,
    numbered_raw_files,
    subsets_for_mode,
    _read_no_view,
)

INDEX_NAME = "reconstruction-index.json"
BRIGHTNESS_SCALE = 0.01  # PWLS beta = this * mean(fieldMatrixMax.bin)

# 1609 reconstruction geometry (same target as configs/recon_groups.json Mode*/1609)
FDK_1609 = {"xNum": 640, "yNum": 640, "zNum": 360,
            "xSpacing": 0.25, "ySpacing": 0.25, "zSpacing": 0.25}


def _brightness_mean(cfg_dir: Path) -> float:
    f = cfg_dir / "fieldMatrixMax.bin"
    if not f.is_file():
        raise FileNotFoundError(f"fieldMatrixMax.bin not found in {cfg_dir}")
    return float(np.fromfile(f, dtype="<f4").mean())


def patch_pwls_hamming(filter_xml: Path, beta: float) -> list[str]:
    """Force PWLS on with *beta* and enable the Hamming window; edits in place."""
    text = filter_xml.read_text(encoding="utf-8")
    changes: list[str] = []

    def _block(tag: str) -> tuple[int, int] | None:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        return (m.start(1), m.end(1)) if m else None

    # ---- <Pwls>: isEnable=1 + beta ----
    span = _block("Pwls")
    if span is None:
        raise ValueError(f"{filter_xml}: no <Pwls> section")
    block = text[span[0]:span[1]]
    if re.search(r"<isEnable>\s*0\s*</isEnable>", block):
        block = re.sub(r"<isEnable>\s*0\s*</isEnable>", "<isEnable>1</isEnable>", block, count=1)
        changes.append("pwls.isEnable 0->1")
    elif not re.search(r"<isEnable>", block):
        lead = re.search(r"^(\s*)", block).group(1)
        block = "<isEnable>1</isEnable>\n" + lead + block.lstrip("\r\n\t ")
        changes.append("pwls.isEnable inserted=1")
    if re.search(r"<beta>", block):
        block = re.sub(r"(<beta>)[^<]*(</beta>)", rf"\g<1>{beta:g}\g<2>", block, count=1)
        changes.append(f"pwls.beta={beta:g}")
    text = text[:span[0]] + block + text[span[1]:]

    # ---- <Ramp>: isHamEnable=1, Hamming coefficients 0.54/0.46 ----
    span = _block("Ramp")
    if span is None:
        raise ValueError(f"{filter_xml}: no <Ramp> section")
    block = text[span[0]:span[1]]
    if re.search(r"<isHamEnable>\s*0\s*</isHamEnable>", block):
        block = re.sub(r"<isHamEnable>\s*0\s*</isHamEnable>", "<isHamEnable>1</isHamEnable>", block, count=1)
        changes.append("ramp.isHamEnable 0->1")
    for tag, value in (("paraHam1", "0.54"), ("paraHam2", "0.46")):
        if re.search(rf"<{tag}>", block):
            block = re.sub(rf"(<{tag}>)[^<]*(</{tag}>)", rf"\g<1>{value}\g<2>", block, count=1)
            changes.append(f"ramp.{tag}={value}")
    text = text[:span[0]] + block + text[span[1]:]

    filter_xml.write_text(text, encoding="utf-8")
    return changes


def _atomic_raw(path: Path, volume: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    np.asarray(volume, dtype="<f4").tofile(tmp)
    for attempt in range(4):
        try:
            tmp.replace(path)
            return
        except OSError:
            # SMB shares may transiently refuse the overwrite-rename; drop a stale
            # target and retry before giving up.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 3:
                raise
            import time as _t
            _t.sleep(3 + 2 * attempt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIReduceNoise2: PWLS(0.01*brightness)+Hamming batch reconstruction")
    p.add_argument("--index", type=Path, default=r"Z:\raid0\yyc\AIReduceNoise\reconstruction-index.json",
                   help="AIReduceNoise reconstruction-index.json (raw/cfg/kv/ma source)")
    p.add_argument("--manifest", type=Path, default=r"Z:\raid0\yyc\AIReduceNoise\splits_1609v2.json",
                   help="1609 split manifest selecting the studies to rebuild")
    p.add_argument("--output-root", type=Path, default=r"Z:\raid0\yyc\AIReduceNoise2")
    p.add_argument("--xday-dir", type=Path, default=Path(r"D:\XDay Release\XDay 617.2_polyc"))
    p.add_argument("--combo", action="append", metavar="TAG=CHAIN", default=None,
                   help=f"Repeatable. Default: {'; '.join(f'{t}={c!r}' for t, c in DEFAULT_COMBOS)}")
    p.add_argument("--dtype", default="float32", choices=("int16", "uint16", "float32"))
    p.add_argument("--link-mode", default="copy", choices=("auto", "copy", "hardlink"))
    p.add_argument("--scale", type=float, default=BRIGHTNESS_SCALE, help="PWLS beta = scale * brightness mean")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0, help="Skip first N selected studies (parallel shards)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    return p


def _parse_combo(spec: str) -> tuple[str, str]:
    tag, sep, chain = spec.partition("=")
    if not sep:
        raise SystemExit(f"--combo must be TAG=CHAIN, got {spec!r}")
    return tag, chain.strip()


def main() -> int:
    args = build_parser().parse_args()
    scale = args.scale
    combos = tuple(_parse_combo(s) for s in args.combo) if args.combo else DEFAULT_COMBOS
    split_modes = DEFAULT_SPLIT_MODES
    per_view = {mode: dict(files) for mode, files in DEFAULT_PER_VIEW_FILES.items()}
    step_modules = DEFAULT_STEP_MODULES

    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    by_key = {(r["study_uid"], r["group"]): r for r in index["records"]}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected, missing = [], []
    for r in manifest["records"]:
        rec = by_key.get((r["study_uid"], r["group"]))
        (selected if rec else missing).append(r)
    if missing:
        print(f"[warn] {len(missing)} manifest records absent from index", file=sys.stderr)
    if args.limit:
        selected = selected[args.offset: args.offset + args.limit]
    else:
        selected = selected[args.offset:]

    if args.dry_run:
        total_vols = 0
        for r in selected:
            rec = by_key[(r["study_uid"], r["group"])]
            names = output_names_for_mode(rec["mode"], combos, split_modes)
            total_vols += len(names)
            print(f"  {rec['group']:<12} {rec['case_id']} {rec['study_uid']}  -> {len(names)} volumes")
        print(f"dry-run: {len(selected)} studies, {total_vols} volumes")
        return 0

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        from xday import load_xday, invoke_xday
    except ImportError:
        print("xday-kits required: pip install -e D:\\workspace2\\xday_kits", file=sys.stderr)
        return 2
    sdk = load_xday(args.xday_dir)

    records, failures, started = [], 0, time.monotonic()
    total = len(selected)
    for i, split_rec in enumerate(selected, 1):
        rec = by_key[(split_rec["study_uid"], split_rec["group"])]
        group_name = rec["group"]
        prefix = ["PDL", group_name, rec["case_id"], rec["study_uid"]]
        case_out = output_root.joinpath(*prefix)
        rel_prefix = "/".join(prefix)
        expected = output_names_for_mode(rec["mode"], combos, split_modes)
        record: dict | None = None
        try:
            if not args.overwrite and all((case_out / n).is_file() for n in expected):
                record = {"case_id": rec["case_id"], "study_uid": rec["study_uid"],
                          "scan_id": rec.get("scan_id") or rec["study_uid"], "mode": rec["mode"],
                          "group": group_name, "kv": rec["kv"], "ma": rec["ma"],
                          "status": "success", "skipped": True,
                          "outputs": {n: f"{rel_prefix}/{n}" for n in expected}}
            else:
                raw_dir, cfg_dir = Path(rec["raw_dir"]), Path(rec["cfg_dir"])
                kv, ma = rec["kv"], rec["ma"]
                if not kv or not ma:
                    from cbct_n2n.reconstruction.prepare import _parse_kv_ma
                    kv, ma = _parse_kv_ma(raw_dir.name)
                brightness = _brightness_mean(cfg_dir)
                beta = scale * brightness
                source_no_view = _read_no_view(cfg_dir / "fdkPara.xml")
                nviews = int(rec.get("nviews") or len(numbered_raw_files(raw_dir)))
                case_out.mkdir(parents=True, exist_ok=True)
                workspace = Path(tempfile.mkdtemp(prefix=f"aired2_{rec['study_uid'][-12:]}_"))
                try:
                    full_raw, full_cfg = build_full_workspace(raw_dir, cfg_dir, workspace / "full",
                                                              args.link_mode, max_views=source_no_view)
                    workspaces = {"full": (full_raw, full_cfg)}
                    subsets = subsets_for_mode(rec["mode"], split_modes)
                    if "even" in subsets:
                        even_idx, odd_idx = split_projection_indices(nviews)
                        pv, pv_warns = validated_per_view(cfg_dir, nviews, per_view.get(rec["mode"], {}))
                        for name, idxs in (("even", even_idx), ("odd", odd_idx)):
                            ws = build_subset_workspace(raw_dir, cfg_dir, idxs, workspace / name,
                                                        args.link_mode, pv)
                            workspaces[name] = (ws.raw_dir, ws.cfg_dir)

                    fdk_changed: list[str] = []
                    filter_patches: dict[str, list[str]] = {}
                    for subset, (_raw, _cfg) in workspaces.items():
                        fdk_changed += apply_fdk_para(_cfg, FDK_1609)
                        filter_patches[subset] = patch_pwls_hamming(Path(_cfg) / "filter.xml", beta)
                        if subset != "full":
                            fix_subsample_step(_cfg, step_modules)

                    common = dict(xday_install_dir=str(args.xday_dir), dtype=args.dtype,
                                  show_progress=False, sdk=sdk)
                    results = {}
                    for subset, (_raw, _cfg) in workspaces.items():
                        for tag, chain in combos:
                            if (subset, tag) in OUTPUT_NAMES:
                                results[(subset, tag)] = invoke_xday(_raw, _cfg, kv, ma,
                                                                     post_procs=chain, **common)
                    shapes = {k: tuple(int(v) for v in res.volume.shape) for k, res in results.items()}
                    if len(set(shapes.values())) != 1:
                        raise ValueError(f"shape mismatch {shapes}")
                    info0 = results[("full", combos[0][0])].volume_info
                    spacing = [float(info0.z_spacing), float(info0.y_spacing), float(info0.x_spacing)]
                    outputs = {}
                    for (s, t), res in results.items():
                        name = OUTPUT_NAMES[(s, t)]
                        _atomic_raw(case_out / name, res.volume)
                        outputs[name] = f"{rel_prefix}/{name}"
                    record = {"case_id": rec["case_id"], "study_uid": rec["study_uid"],
                              "scan_id": rec.get("scan_id") or rec["study_uid"], "mode": rec["mode"],
                              "group": group_name, "kv": kv, "ma": ma,
                              "status": "success", "skipped": False,
                              "shape": list(shapes[("full", combos[0][0])]), "dtype": "<f4",
                              "spacing_zyx": spacing, "outputs": outputs,
                              "nviews": nviews, "brightness": brightness,
                              "pwls_beta": beta, "pwls_scale": scale,
                              "filter_patches": filter_patches, "fdk_changed": list(dict.fromkeys(fdk_changed)),
                              "filter_file": None, "pwls_hamming": True}
                    meta = dict(record)
                    meta["source_root"] = str(index.get("source_root", ""))
                    tmp_meta = case_out / "metadata.json.tmp"
                    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp_meta.replace(case_out / "metadata.json")
                    issues = [] if args.no_verify else verify_volumes(record, output_root)
                    if issues:
                        record["status"] = "verify_failed"
                        record["verify_issues"] = issues
                finally:
                    shutil.rmtree(workspace, ignore_errors=True)
        except Exception as exc:
            failures += 1
            record = {"case_id": rec["case_id"], "study_uid": rec["study_uid"],
                      "mode": rec["mode"], "group": group_name, "kv": rec.get("kv"), "ma": rec.get("ma"),
                      "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{i}/{total}] {rec['case_id']}/{rec['study_uid']} [{group_name}]: ERROR {exc}", flush=True)
            if not args.continue_on_error:
                break
        records.append(record)
        if record["status"] == "success":
            beta_v = record.get("pwls_beta")
            beta_txt = f"{beta_v:g}" if isinstance(beta_v, (int, float)) else "-"
            print(f"[{i}/{total}] {rec['case_id']}/{rec['study_uid']} [{group_name}]: "
                  f"{'skipped (complete)' if record.get('skipped') else 'ok'} "
                  f"beta={beta_txt}", flush=True)
        else:
            print(f"[{i}/{total}] {rec['case_id']}/{rec['study_uid']}: {record['status']}", flush=True)
        elapsed = time.monotonic() - started
        eta = max(0.0, (total - i) * elapsed / i)
        print(f"    {elapsed/60:.1f} min elapsed, {elapsed/i:.1f} s/item, ETA {eta/60:.1f} min", flush=True)

    existing = load_index(output_root / INDEX_NAME) if (output_root / INDEX_NAME).is_file() else None
    merged = merge_index_records(existing.get("records", []) if existing else [], records)
    out_index = {
        "schema_version": 1,
        "description": "AIReduceNoise2: 1609 studies, PWLS beta=0.01*mean(fieldMatrixMax.bin) + Hamming window",
        "source_root": str(index.get("source_root", "")),
        "output_root": str(output_root),
        "pwls": {"isEnable": 1, "iterCount": 20, "delta": 0.5, "beta": "0.01 * mean(fieldMatrixMax.bin)"},
        "ramp": {"isHamEnable": 1, "paraHam1": 0.54, "paraHam2": 0.46},
        "records": merged,
    }
    tmp = output_root / (INDEX_NAME + ".tmp")
    tmp.write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_root / INDEX_NAME)
    ok = sum(1 for r in records if r["status"] == "success")
    print(f"Wrote {output_root / INDEX_NAME}: ok={ok} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
