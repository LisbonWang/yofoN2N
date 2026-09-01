"""Batch CBCT reconstruction: 6 volumes per study, stored under a ``<Mode>_<FOV>`` top-level key.

Each study is a ``(raw projections, config)`` pair discovered under a source root with the
layout used by the PDL dataset::

    <source_root>/<case_id>/<study_uid>/<series>_ModeX_<kv>kV<ma>mA.../<NNNNN>.raw
    <source_root>/<case_id>/<study_uid>/config/recon/<dev>/<ModeX>/filter.xml + fdkPara.xml + ...

Outputs are grouped by *mode + field of view* derived from ``fdkPara.xml``
(``xNum*xSpacing`` and ``zNum*zSpacing`` in cm, two digits each, e.g. 160x150mm -> ``1615``)::

    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/even.raw
    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/odd.raw
    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/full.raw
    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/full.marfi.raw
    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/full.bhc.raw
    <output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/full.marfi.bhc.raw

The even/odd subsets only get the plain (no post-processing) variant; the full subset gets
all four MARFI/BHC combos (see ``OUTPUT_NAMES``).

Reconstruction always works on a *temporary copy* of raw and config.  Even/odd subsets are
built by ``build_subset_workspace`` (copies the selected projections, updates ``noView`` in
fdkPara.xml, slices per-view binaries such as ``projectionMatrix.bin``).  The full subset is a
plain copy whose ``noView`` is normalized to the actual projection count.

MARFI/BHC post-processing is controlled by the ``--post-procs`` chain of ``xday``; the four
combos are produced by toggling BHC and MARFI.  For downsampled (even/odd) subsets, the
``step`` parameter of MARFI (and BHC when present) must stay even when it is > 1; the temp
config copy is patched accordingly before reconstruction.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .prepare import _parse_kv_ma
from .subset import build_subset_workspace, numbered_raw_files, split_projection_indices

#: subset name -> display name used in output file names
SUBSETS = ("full", "even", "odd")

#: default post-processing combos: tag -> xday --post-procs chain
DEFAULT_COMBOS: tuple[tuple[str, str], ...] = (
    ("none", ""),
    ("bhc", "BHC"),
    ("marfi", "MARFI"),
    ("bhc_marfi", "BHC MARFI"),
)

#: (subset, combo tag) -> output file name.  even/odd only get the plain variant;
#: full gets all four MARFI/BHC combos.  ``bhc_marfi`` is written as ``marfi.bhc``.
OUTPUT_NAMES: dict[tuple[str, str], str] = {
    ("even", "none"): "even.raw",
    ("odd", "none"): "odd.raw",
    ("full", "none"): "full.raw",
    ("full", "marfi"): "full.marfi.raw",
    ("full", "bhc"): "full.bhc.raw",
    ("full", "bhc_marfi"): "full.marfi.bhc.raw",
}

#: modules whose ``step`` must be even after projection downsampling (even/odd subsets)
DEFAULT_STEP_MODULES = ("MARFI", "BHC")

#: per-view binary files sliced per projection index, keyed by scan mode.  Mode1 stores
#: projectionMatrix.bin as a shared/whole file (size != 1046 x 84), so it is copied intact.
DEFAULT_PER_VIEW_FILES: dict[str, dict[str, int]] = {
    "Mode1": {},
    "Mode2": {"projectionMatrix.bin": 21 * 4},
    "Mode3": {"projectionMatrix.bin": 21 * 4},
}

_RAW_NAME = re.compile(r"\d+\.raw$", re.IGNORECASE)
_MODE_NAME = re.compile(r"(?:^|_)(Mode\d+)(?:_|$)", re.IGNORECASE)
_TAG_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class Study:
    """A single reconstructable (raw, config) pair."""

    case_id: str
    study_uid: str
    scan_id: str
    raw_dir: Path
    cfg_dir: Path
    mode: str
    kv: float | None
    ma: float | None
    nviews: int


@dataclass
class VariantOptions:
    """Options shared by all studies of one batch run."""

    xday_dir: Path
    combos: tuple[tuple[str, str], ...] = DEFAULT_COMBOS
    dtype: str = "float32"
    link_mode: str = "copy"
    per_view_files: dict[str, dict[str, int]] = field(default_factory=lambda: {m: dict(v) for m, v in DEFAULT_PER_VIEW_FILES.items()})
    filter_set: dict[str, str] = field(default_factory=dict)
    step_modules: tuple[str, ...] = DEFAULT_STEP_MODULES
    kv: float | None = None
    ma: float | None = None
    overwrite: bool = False
    keep_workspace: bool = False
    workspace_root: str | Path | None = None


# --------------------------------------------------------------------------- discovery


def _has_raw_files(directory: Path) -> bool:
    try:
        with os.scandir(directory) as entries:
            return any(e.is_file() and _RAW_NAME.search(e.name) for e in entries)
    except OSError:
        return False


def _raw_file_count(directory: Path) -> int:
    with os.scandir(directory) as entries:
        return sum(1 for e in entries if e.is_file() and _RAW_NAME.search(e.name))


def _find_configs(study_dir: Path) -> list[Path]:
    """Return config dirs that contain both filter.xml and fdkPara.xml."""
    results: list[Path] = []
    for base, _dirs, files in os.walk(study_dir):
        if "filter.xml" in files and (Path(base) / "fdkPara.xml").is_file():
            results.append(Path(base))
    return results


def discover_studies(
    source_root: str | Path,
    *,
    mode: str | None = None,
    kv: float | None = None,
    ma: float | None = None,
) -> tuple[list[Study], list[dict[str, Any]]]:
    """Discover studies in the PDL layout; return ``(studies, skipped)``.

    A study is usable when it has exactly one raw projection directory and exactly one
    matching config directory.  Anything else is reported in ``skipped`` with a reason.
    """
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Source root not found: {root}")
    studies: list[Study] = []
    skipped: list[dict[str, Any]] = []
    requested = mode.casefold() if mode else None
    for case_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for study_dir in sorted((p for p in case_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            raw_dirs = [Path(e.path) for e in os.scandir(study_dir) if e.is_dir() and _has_raw_files(Path(e.path))]
            configs = _find_configs(study_dir)
            if len(raw_dirs) != 1:
                skipped.append({"case_id": case_dir.name, "study_uid": study_dir.name,
                                "reason": f"expected 1 raw dir, found {len(raw_dirs)}"})
                continue
            if len(configs) != 1:
                skipped.append({"case_id": case_dir.name, "study_uid": study_dir.name,
                                "reason": f"expected 1 config dir, found {len(configs)}"})
                continue
            raw_dir, cfg_dir = raw_dirs[0], configs[0]
            raw_match = _MODE_NAME.search(raw_dir.name)
            cfg_mode = _MODE_NAME.search(cfg_dir.name)
            raw_mode = raw_match.group(1) if raw_match else None
            cfg_mode_name = cfg_mode.group(1) if cfg_mode else None
            if raw_mode is None or cfg_mode_name is None or raw_mode.casefold() != cfg_mode_name.casefold():
                skipped.append({"case_id": case_dir.name, "study_uid": study_dir.name,
                                "reason": f"mode mismatch raw={raw_mode} cfg={cfg_mode_name}"})
                continue
            if requested is not None and raw_mode.casefold() != requested:
                continue
            scan_id = study_dir.name
            try:
                parsed_kv, parsed_ma = _parse_kv_ma(raw_dir.name)
            except ValueError:
                if kv is None or ma is None:
                    skipped.append({"case_id": case_dir.name, "study_uid": study_dir.name,
                                    "reason": f"cannot parse kV/mA from {raw_dir.name}"})
                    continue
                parsed_kv, parsed_ma = kv, ma
            studies.append(Study(
                case_id=case_dir.name,
                study_uid=study_dir.name,
                scan_id=scan_id,
                raw_dir=raw_dir,
                cfg_dir=cfg_dir,
                mode=raw_mode,
                kv=parsed_kv,
                ma=parsed_ma,
                nviews=_raw_file_count(raw_dir),
            ))
    return studies, skipped


# --------------------------------------------------------------------------- FOV classification


def fov_suffix_from_fdk(fdk_path: str | Path) -> str:
    """Derive the 4-digit FOV suffix from a ``fdkPara.xml``.

    The suffix is ``xFOV`` x ``zFOV`` in cm, two zero-padded digits each, e.g.
    160x150 mm -> ``1615``, 160x90 mm -> ``1609``, 80x80 mm -> ``0808``.
    """
    tree = ET.parse(Path(fdk_path))
    root = tree.getroot()

    def _field(tag: str) -> float:
        node = root.findtext(tag)
        if node is None:
            raise ValueError(f"Missing <{tag}> in {fdk_path}")
        return float(node)

    x_mm = _field("xNum") * _field("xSpacing")
    z_mm = _field("zNum") * _field("zSpacing")
    return f"{round(x_mm / 10):02d}{round(z_mm / 10):02d}"


def mode_fov_key(mode: str, fdk_path: str | Path) -> str:
    """Top-level output key, e.g. ``Mode2_1609``."""
    return f"{mode}_{fov_suffix_from_fdk(fdk_path)}"


# --------------------------------------------------------------------------- workspaces


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode not in {"auto", "copy", "hardlink"}:
        raise ValueError("link_mode must be auto, copy, or hardlink")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(src, dst)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)


def _update_no_view(path: Path, view_count: int) -> None:
    tree = ET.parse(path)
    node = tree.getroot().find(".//noView")
    if node is None:
        raise ValueError(f"Missing noView in {path}")
    node.text = str(view_count)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_full_workspace(
    raw_dir: str | Path,
    cfg_dir: str | Path,
    destination: str | Path,
    link_mode: str = "copy",
) -> tuple[Path, Path]:
    """Copy all raw projections and the full config into a temp workspace.

    ``noView`` in the copied fdkPara.xml is normalized to the actual projection count so
    that every reconstruction uses the real number of views.  Returns ``(raw_dir, cfg_dir)``
    of the freshly built workspace.
    """
    raw_files = numbered_raw_files(raw_dir)
    cfg_path = Path(cfg_dir)
    if not cfg_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {cfg_path}")
    out = Path(destination)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Workspace destination is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    out_raw, out_cfg = out / "raw", out / "config"
    out_raw.mkdir()
    out_cfg.mkdir()
    for src in raw_files:
        _copy_or_link(src, out_raw / src.name, link_mode)
    for item in cfg_path.iterdir():
        if item.is_file():
            shutil.copy2(item, out_cfg / item.name)
    fdk = out_cfg / "fdkPara.xml"
    if not fdk.is_file():
        raise ValueError(f"Missing fdkPara.xml in {cfg_path}")
    _update_no_view(fdk, len(raw_files))
    return out_raw, out_cfg


# --------------------------------------------------------------------------- config patching


def validated_per_view(cfg_dir: str | Path, nviews: int, mapping: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    """Drop per-view entries whose size does not match ``nviews * width``.

    Layout varies by scan mode (e.g. Mode1 stores projectionMatrix.bin as a shared file).
    A mismatching file cannot be sliced and is copied whole instead; the reason is returned
    as a warning so the batch keeps running.
    """
    cfg_path = Path(cfg_dir)
    valid: dict[str, int] = {}
    warnings: list[str] = []
    for name, width in mapping.items():
        path = cfg_path / name
        if not path.is_file():
            warnings.append(f"{name}: file missing, treated as absent")
            continue
        size = path.stat().st_size
        if size != nviews * width:
            warnings.append(f"{name}: size {size} != {nviews} x {width} bytes, treated as shared (copied whole)")
            continue
        valid[name] = width
    return valid, warnings


def _module_block_pattern(module: str) -> re.Pattern:
    """Match ``<TAG ...>...</TAG>`` where TAG equals *module* or ends with ``::module``.

    Real filter.xml files contain non-standard tags such as ``<SecondaryBP::BHC>`` which
    xml.etree cannot parse; patching is therefore regex-based and preserves the file
    byte-for-byte except for the edited values.
    """
    return re.compile(
        rf"(?P<open><(?P<tag>(?:[A-Za-z0-9_:]+::)?{re.escape(module)})(?:\s[^>]*)?>)"
        rf"(?P<body>.*?)(?P<close></(?P=tag)>)",
        re.DOTALL | re.IGNORECASE,
    )


def _find_module_blocks(text: str, module: str) -> list[re.Match]:
    """Return module blocks, keeping only the outermost when the tag nests (e.g. ``<BHC>``
    inside ``<SecondaryBP::BHC>``) so replacements never use stale offsets."""
    matches = list(_module_block_pattern(module).finditer(text))
    outermost: list[re.Match] = []
    for match in matches:
        if any(prev.start() <= match.start() and match.end() <= prev.end() for prev in outermost):
            continue
        outermost.append(match)
    return outermost


def read_module_field(cfg_dir: str | Path, module: str, field: str) -> str | None:
    """Return the text of ``<field>`` inside the first *module* block, or None."""
    text = (Path(cfg_dir) / "filter.xml").read_text(encoding="utf-8")
    for match in _find_module_blocks(text, module):
        field_match = re.search(rf"<{re.escape(field)}(?:\s[^>]*)?>(.*?)</{re.escape(field)}>", match.group("body"), re.DOTALL | re.IGNORECASE)
        if field_match:
            return field_match.group(1).strip()
    return None


def _patch_filter_xml(
    cfg_dir: str | Path,
    patch: Callable[[str], str],
) -> str:
    """Read filter.xml, apply *patch* to the full text, atomically write it back."""
    path = Path(cfg_dir) / "filter.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing filter.xml in {cfg_dir}")
    text = path.read_text(encoding="utf-8")
    patched = patch(text)
    if patched != text:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(path)
    return patched


def fix_subsample_step(cfg_dir: str | Path, modules: Iterable[str] = DEFAULT_STEP_MODULES) -> list[tuple[str, int, int]]:
    """Round odd ``step > 1`` up to even for the given modules in filter.xml.

    After even/odd downsampling the MARFI (or BHC) ``step`` must stay even; this patches the
    *workspace* config copy only.  Returns ``(module_tag, old_value, new_value)`` changes.
    """
    wanted = tuple(m.casefold() for m in modules)
    changes: list[tuple[str, int, int]] = []

    def patch(text: str) -> str:
        out = text
        for module in wanted:
            for match in reversed(_find_module_blocks(out, module)):
                body = match.group("body")
                step_match = re.search(r"<step(?:\s[^>]*)?>\s*(\d+)\s*</step>", body, re.IGNORECASE)
                if step_match is None:
                    continue
                value = int(step_match.group(1))
                if value > 1 and value % 2 == 1:
                    new_body = body[: step_match.start()] + f"<step>{value + 1}</step>" + body[step_match.end():]
                    out = out[: match.start()] + match.group("open") + new_body + match.group("close") + out[match.end():]
                    changes.append((match.group("tag"), value, value + 1))
        return out

    _patch_filter_xml(cfg_dir, patch)
    return changes


def apply_filter_overrides(cfg_dir: str | Path, overrides: dict[str, str]) -> list[tuple[str, str]]:
    """Apply ``MODULE.FIELD=value`` overrides to the workspace filter.xml copy."""
    applied: list[tuple[str, str]] = []

    def patch(text: str) -> str:
        out = text
        for key, value in overrides.items():
            module_name, sep, field_name = key.partition(".")
            if not sep or not field_name:
                raise ValueError(f"override must be MODULE.FIELD=value, got {key!r}")
            found_module = False
            for match in reversed(_find_module_blocks(out, module_name)):
                found_module = True
                body = match.group("body")
                field_pattern = re.compile(rf"<{re.escape(field_name)}(?:\s[^>]*)?>.*?</{re.escape(field_name)}>", re.DOTALL | re.IGNORECASE)
                field_match = field_pattern.search(body)
                if field_match:
                    new_body = body[: field_match.start()] + f"<{field_name}>{value}</{field_name}>" + body[field_match.end():]
                else:
                    new_body = body + f"<{field_name}>{value}</{field_name}>"
                out = out[: match.start()] + match.group("open") + new_body + match.group("close") + out[match.end():]
            if not found_module:
                raise ValueError(f"module {module_name!r} not found in filter.xml")
            applied.append((key, str(value)))
        return out

    _patch_filter_xml(cfg_dir, patch)
    return applied


# --------------------------------------------------------------------------- orchestration


def _default_reconstruct(*args: Any, **kwargs: Any):
    try:
        from xday import invoke_xday
    except ImportError as exc:
        raise RuntimeError("xday-kits is required for reconstruction; install it in the torch environment") from exc
    return invoke_xday(*args, **kwargs)


def _atomic_raw(path: Path, volume: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    np.asarray(volume, dtype="<f4").tofile(temporary)
    os.replace(temporary, path)


def _workspace_context(study: Study, output_dir: Path, options: VariantOptions, mode_fov: str):
    if options.keep_workspace:
        parent = output_dir / mode_fov / study.case_id / study.study_uid / "workspace"
        if parent.exists() and options.overwrite:
            shutil.rmtree(parent)
        parent.mkdir(parents=True, exist_ok=True)
        return parent
    return Path(tempfile.mkdtemp(prefix=f"xday12_{study.scan_id[:16]}_", dir=options.workspace_root))


def reconstruct_study(
    study: Study,
    output_dir: str | Path,
    options: VariantOptions,
    *,
    reconstruct_fn: Callable[..., Any] = _default_reconstruct,
    sdk: Any = None,
) -> dict[str, Any]:
    """Reconstruct the 6 output variants (``OUTPUT_NAMES``) for one study and write them to disk.

    Volumes are stored under ``<output_root>/<Mode>_<FOV>/<case_id>/<study_uid>/`` where the
    FOV suffix is derived from the study's fdkPara.xml.  Reconstructs the full subset from a
    plain copy of the raw/config and the even/odd subsets from sliced workspaces; applies
    ``--filter-set`` overrides everywhere and the even-step rule to even/odd.
    """
    output_root = Path(output_dir)
    mode_fov = mode_fov_key(study.mode, study.cfg_dir / "fdkPara.xml")
    case_out = output_root / mode_fov / study.case_id / study.study_uid
    expected = list(OUTPUT_NAMES.values())
    if not options.overwrite and all((case_out / name).is_file() for name in expected) and (case_out / "metadata.json").is_file():
        record = _base_record(study, options, mode_fov)
        record.update({"status": "success", "skipped": True,
                       "outputs": {name: f"{mode_fov}/{study.case_id}/{study.study_uid}/{name}" for name in expected}})
        return record
    if not study.nviews:
        raise ValueError("study has no projections")
    if study.kv is None or study.ma is None:
        raise ValueError("kV/mA unknown; pass --kv/--ma or fix the raw directory name")

    case_out.mkdir(parents=True, exist_ok=True)
    per_view, per_view_warnings = validated_per_view(
        study.cfg_dir, study.nviews, options.per_view_files.get(study.mode, {})
    )
    workspace = _workspace_context(study, output_root, options, mode_fov)
    try:
        full_raw, full_cfg = build_full_workspace(study.raw_dir, study.cfg_dir, workspace / "full", options.link_mode)
        even_indices, odd_indices = split_projection_indices(study.nviews)
        even_ws = build_subset_workspace(study.raw_dir, study.cfg_dir, even_indices, workspace / "even", options.link_mode, per_view)
        odd_ws = build_subset_workspace(study.raw_dir, study.cfg_dir, odd_indices, workspace / "odd", options.link_mode, per_view)
        workspaces = {"full": (full_raw, full_cfg), "even": (even_ws.raw_dir, even_ws.cfg_dir), "odd": (odd_ws.raw_dir, odd_ws.cfg_dir)}

        step_fixes: dict[str, list[tuple[str, int, int]]] = {}
        for subset, (_raw, _cfg) in workspaces.items():
            if options.filter_set:
                apply_filter_overrides(_cfg, options.filter_set)
            if subset != "full":
                fixes = fix_subsample_step(_cfg, options.step_modules)
                if fixes:
                    step_fixes[subset] = fixes

        common = dict(xday_install_dir=options.xday_dir, dtype=options.dtype, show_progress=False, sdk=sdk)
        wanted = set(OUTPUT_NAMES)
        results: dict[tuple[str, str], Any] = {}
        for subset, (_raw, _cfg) in workspaces.items():
            for tag, chain in options.combos:
                if (subset, tag) not in wanted:
                    continue
                results[(subset, tag)] = reconstruct_fn(_raw, _cfg, study.kv, study.ma, post_procs=chain, **common)

        shapes = {key: tuple(int(v) for v in result.volume.shape) for key, result in results.items()}
        if len({tuple(s) for s in shapes.values()}) != 1:
            raise ValueError(f"reconstructed shapes differ: {shapes}")
        spacings = {}
        for (subset, tag), result in results.items():
            info = result.volume_info
            spacings[f"{subset}_{tag}"] = [float(info.z_spacing), float(info.y_spacing), float(info.x_spacing)]
        if len({tuple(s) for s in spacings.values()}) != 1:
            raise ValueError(f"reconstructed spacings differ: {spacings}")

        outputs: dict[str, str] = {}
        for (subset, tag), result in results.items():
            name = OUTPUT_NAMES[(subset, tag)]
            _atomic_raw(case_out / name, result.volume)
            outputs[name] = f"{mode_fov}/{study.case_id}/{study.study_uid}/{name}"

        record = _base_record(study, options, mode_fov)
        record.update({
            "status": "success",
            "skipped": False,
            "shape": list(shapes["full", "none"]),
            "dtype": "<f4",
            "spacing_zyx": spacings["full_none"],
            "outputs": outputs,
            "step_fixes": step_fixes,
            "filter_set": dict(options.filter_set),
            "per_view_sliced": dict(per_view),
            "per_view_warnings": per_view_warnings,
        })
        _write_json(case_out / "metadata.json", record)
        return record
    finally:
        if not options.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def _base_record(study: Study, options: VariantOptions, mode_fov: str | None = None) -> dict[str, Any]:
    record = {
        "case_id": study.case_id,
        "study_uid": study.study_uid,
        "scan_id": study.scan_id,
        "mode": study.mode,
        "raw_dir": str(study.raw_dir),
        "cfg_dir": str(study.cfg_dir),
        "kv": study.kv,
        "ma": study.ma,
        "nviews": study.nviews,
        "combos": [{"tag": tag, "post_procs": chain} for tag, chain in options.combos],
        "even_views": len(split_projection_indices(study.nviews)[0]),
        "odd_views": len(split_projection_indices(study.nviews)[1]),
    }
    if mode_fov is not None:
        record["mode_fov"] = mode_fov
        record["fov_suffix"] = mode_fov.split("_", 1)[1]
    return record


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json_dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
