"""Batch CBCT reconstruction: 6 volumes per study per recon group.

Each study is a ``(raw projections, config)`` pair discovered under a source root with the
layout used by the PDL dataset::

    <source_root>/<case_id>/<study_uid>/<series>_ModeX_<kv>kV<ma>mA.../<NNNNN>.raw
    <source_root>/<case_id>/<study_uid>/config/recon/<dev>/<ModeX>/filter.xml + fdkPara.xml + ...

Reconstruction happens per *group* (see ``configs/recon_groups.json``).  A group defines a
target geometry (fdkPara.xml field values) plus optional filter.xml overrides, and lists the
scan modes it applies to.  A study is reconstructed once for every group whose
``source_modes`` contains the study's mode — e.g. a Mode2 study is rebuilt both as
``Mode2_1609`` (native) and ``Mode2_0808`` (fdkPara patched to an 80x80mm volume)::

    <output_root>/<source>/<group>/<case_id>/<study_uid>/even.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/odd.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/full.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/full.marfi.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/full.bhc.raw
    <output_root>/<source>/<group>/<case_id>/<study_uid>/full.marfi.bhc.raw

Any ``Mode<N>`` is supported (Mode0 included): the mode token is read from the scan/config
directory names via a ``Mode\\d+`` regex, and modes without an entry in
``DEFAULT_PER_VIEW_FILES`` simply get no per-view slicing (ancillary binaries are copied
whole), which is the safe fallback for unknown layouts.

The even/odd subsets only get the plain (no post-processing) variant; the full subset gets
all four MARFI/BHC combos (see ``OUTPUT_NAMES``).

Reconstruction always works on a *temporary copy* of raw and config; the copy is deleted when
the study finishes (or with ``keep_workspace`` it is kept for debugging).  Even/odd subsets
are built by ``build_subset_workspace`` (copies the selected projections, updates ``noView``
in fdkPara.xml, slices per-view binaries such as ``projectionMatrix.bin``).  The full subset
is a plain copy whose ``noView`` is normalized to the actual projection count.  The group's
``fdk_para`` and ``filter`` overrides are then applied to every subset's config copy.

MARFI/BHC post-processing is controlled by the ``--post-procs`` chain of ``xday``; the four
combos are produced by toggling BHC and MARFI.  For downsampled (even/odd) subsets, the
``step`` parameter of MARFI (and BHC when present) must stay even when it is > 1; the temp
config copy is patched accordingly before reconstruction.
"""

from __future__ import annotations

import json
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

#: modes whose scans get even/odd projection splitting.  Mode2/Mode3 data is
#: self-consistent (raw views == fdkPara noView == projectionMatrix rows) so the
#: downsampled subsets reconstruct correctly.  Mode1 is an odd/even dual-geometry
#: scan (fdkPara carries SIDOdd/SIDEven, zSrcOffsetOdd/zSrcOffsetEven, ... and the
#: projection matrix alternates odd/even rows) whose raw view count (1046) does not
#: match noView/matrix rows (1020); splitting it produces misaligned volumes, so
#: Mode0/Mode1 are reconstructed from the full set only.
DEFAULT_SPLIT_MODES: tuple[str, ...] = ("Mode2", "Mode3")

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


def subsets_for_mode(mode: str, split_modes: Iterable[str] = DEFAULT_SPLIT_MODES) -> tuple[str, ...]:
    """Return the subset names reconstructed for *mode*.

    Modes listed in *split_modes* get the full even/odd/full trio; all other modes
    (e.g. the odd/even dual-geometry Mode1 scan) are reconstructed from the full set
    only, so no even/odd outputs are produced and no per-view slicing applies.
    """
    if any(m.casefold() == mode.casefold() for m in split_modes):
        return SUBSETS
    return ("full",)


def output_names_for_mode(mode: str, combos: Iterable[tuple[str, str]], split_modes: Iterable[str] = DEFAULT_SPLIT_MODES) -> tuple[str, ...]:
    """Output file names produced for a study of *mode* under *combos*.

    Order follows ``SUBSETS`` (full first) then combo order, matching
    ``OUTPUT_NAMES`` for the reconstructed subsets only.
    """
    subsets = subsets_for_mode(mode, split_modes)
    names: list[str] = []
    for subset in subsets:
        for tag, _chain in combos:
            name = OUTPUT_NAMES.get((subset, tag))
            if name is not None and name not in names:
                names.append(name)
    return tuple(names)

#: modules whose ``step`` must be even after projection downsampling (even/odd subsets)
DEFAULT_STEP_MODULES = ("MARFI", "BHC")

#: per-view binary files sliced per projection index, keyed by scan mode.  Mode1 stores
#: projectionMatrix.bin as a shared/whole file (size != 1046 x 84), so it is copied intact.
#: Modes not listed here (e.g. a future Mode0) get no per-view slicing and copy such files
#: whole as well — the safe fallback for unknown layouts.
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
    step_modules: tuple[str, ...] = DEFAULT_STEP_MODULES
    split_modes: tuple[str, ...] = DEFAULT_SPLIT_MODES
    kv: float | None = None
    ma: float | None = None
    overwrite: bool = False
    keep_workspace: bool = False
    workspace_root: str | Path | None = None
    #: top-level output directory name (e.g. ``PDL``, taken from the source root name by the CLI).
    #: ``None`` keeps the legacy flat layout without a top-level source directory.
    top_dir: str | None = None
    #: Data2-style per-study PWLS patch: ``beta = pwls_scale * mean(fieldMatrixMax.bin)``
    #: with PWLS forced on and the Hamming window (0.54/0.46) enabled in every subset's
    #: filter.xml copy (after the group's standard filter replacement).  ``None`` disables
    #: the patch (standard filter used verbatim = data1-style no-PWLS reconstruction).
    pwls_scale: float | None = None


@dataclass(frozen=True)
class ReconGroup:
    """One reconstruction group: a target geometry plus the scan modes it applies to.

    ``name`` is the output directory key below ``top_dir`` (e.g. ``Mode2_0808``).
    ``source_modes`` lists the scan modes whose studies are rebuilt for this group.
    ``fdk_para`` maps fdkPara.xml field name -> target value (applied to every subset's
    config copy).  ``filter_file`` is the absolute path of the group's *standard*
    filter.xml: the file is copied verbatim into every subset's config, replacing the
    source one.  ``None`` keeps the source study's filter.xml untouched (safe fallback
    when no standard file exists yet, e.g. Mode0).
    """

    name: str
    source_modes: tuple[str, ...]
    fdk_para: dict[str, float]
    filter_file: str | None

    def applies_to(self, mode: str) -> bool:
        return any(m.casefold() == mode.casefold() for m in self.source_modes)


def load_recon_groups(path: str | Path) -> dict[str, ReconGroup]:
    """Load group definitions from a JSON file; return ``{name: ReconGroup}``.

    The file layout is ``{"schema_version": 1, "groups": {name: {source_modes, fdk_para,
    filter_file}}}``.  Field values in ``fdk_para`` are coerced to float.  ``filter_file``
    may be ``null`` (no standard file) or a path relative to the JSON file's directory,
    which is resolved to an absolute path and must exist.
    """
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported recon groups schema: {raw.get('schema_version')!r}")
    groups: dict[str, ReconGroup] = {}
    for name, spec in raw.get("groups", {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"group {name!r}: expected an object, got {type(spec).__name__}")
        modes = tuple(spec.get("source_modes", ()))
        if not modes:
            raise ValueError(f"group {name!r}: empty source_modes")
        fdk = {str(k): float(v) for k, v in spec.get("fdk_para", {}).items()}
        filter_file = spec.get("filter_file")
        if filter_file is not None:
            filter_file = (cfg_path.parent / str(filter_file)).resolve()
            if not filter_file.is_file():
                raise ValueError(f"group {name!r}: filter_file not found: {filter_file}")
        groups[name] = ReconGroup(name=name, source_modes=modes, fdk_para=fdk, filter_file=str(filter_file) if filter_file else None)
    return groups


def groups_for_mode(groups: dict[str, ReconGroup], mode: str) -> list[ReconGroup]:
    """Return the groups that apply to *mode*, in file order."""
    return [g for g in groups.values() if g.applies_to(mode)]


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


def _read_no_view(fdk_path: str | Path) -> int | None:
    """Return the ``noView`` value from an fdkPara.xml, or None when absent."""
    try:
        tree = ET.parse(Path(fdk_path))
    except (OSError, ET.ParseError):
        return None
    node = tree.getroot().find(".//noView")
    if node is None or node.text is None:
        return None
    try:
        return int(node.text)
    except ValueError:
        return None


def build_full_workspace(
    raw_dir: str | Path,
    cfg_dir: str | Path,
    destination: str | Path,
    link_mode: str = "copy",
    *,
    max_views: int | None = None,
) -> tuple[Path, Path]:
    """Copy raw projections and the full config into a temp workspace.

    ``noView`` in the copied fdkPara.xml is normalized to the number of projections
    actually copied.  When *max_views* is given (e.g. the source fdkPara's own
    ``noView``), only the first *max_views* projections are copied — raw directories
    may hold more frames than the config declares (the extra ones are ignored by the
    SDK), and copying them all would desynchronize the projection count from the
    per-view ``projectionMatrix.bin``.  Returns ``(raw_dir, cfg_dir)`` of the freshly
    built workspace.
    """
    raw_files = numbered_raw_files(raw_dir)
    if max_views is not None and len(raw_files) > max_views:
        raw_files = raw_files[:max_views]
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


def validate_study_config(cfg_dir: str | Path) -> list[str]:
    """Return problems that would make the XDay engine crash on this config.

    The engine crashes with an opaque ``access violation`` when ``current.xml`` or
    ``filterPara.xml`` are blank (incomplete copies) instead of a proper XML document, so
    such studies must be filtered out *before* reconstruction.  ``filter.xml`` is
    deliberately not checked: real files use non-standard tags (``<SecondaryBP::BHC>``)
    that no XML parser accepts.
    """
    problems: list[str] = []
    for name in ("current.xml", "filterPara.xml", "fdkPara.xml"):
        f = Path(cfg_dir) / name
        if not f.is_file():
            problems.append(f"{name}: missing")
            continue
        data = f.read_bytes()
        if not data.strip():
            problems.append(f"{name}: blank ({len(data)} bytes, incomplete copy)")
            continue
        try:
            ET.fromstring(data.decode("utf-8", errors="replace"))
        except ET.ParseError as exc:
            problems.append(f"{name}: unparsable ({exc})")
    return problems


def brightness_mean(cfg_dir: str | Path) -> float:
    """Mean of the study config's fieldMatrixMax brightness bins (the data2 PWLS beta source).

    Accepts both layouts: a single ``fieldMatrixMax.bin`` (data2-style) or per-panel
    ``fieldMatrixMaxA.bin`` / ``fieldMatrixMaxB.bin`` (PDL dual-panel detectors).  The mean
    is taken over the concatenated values of all matches, which reduces to the plain
    single-file mean when only one file exists.
    """
    files = sorted(Path(cfg_dir).glob("fieldMatrixMax*.bin"))
    if not files:
        raise FileNotFoundError(f"no fieldMatrixMax*.bin in {cfg_dir}")
    total = 0.0
    count = 0
    for f in files:
        values = np.fromfile(f, dtype="<f4")
        total += float(values.sum(dtype=np.float64))
        count += values.size
    return total / count


def patch_pwls_hamming(cfg_dir: str | Path, beta: float) -> list[str]:
    """Force PWLS on with *beta* and enable the Hamming window (0.54/0.46).

    Edits the workspace config copy's ``filter.xml`` in place (atomic write).  This is the
    data2-style per-study patch ported from ``reconstruct_aired2_pwls.py``: the PWLS
    ``iterCount``/``delta`` values already present in the filter file are kept, only
    ``isEnable`` and ``beta`` are touched.  Returns human-readable change notes.
    """
    changes: list[str] = []

    def patch(text: str) -> str:
        out = text

        def _block(tag: str, src: str) -> tuple[int, int] | None:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", src, re.S)
            return (m.start(1), m.end(1)) if m else None

        # ---- <Pwls>: isEnable=1 + beta ----
        span = _block("Pwls", out)
        if span is None:
            raise ValueError(f"{cfg_dir}: filter.xml has no <Pwls> section")
        block = out[span[0]:span[1]]
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
        out = out[:span[0]] + block + out[span[1]:]

        # ---- <Ramp>: isHamEnable=1, Hamming coefficients 0.54/0.46 ----
        span = _block("Ramp", out)
        if span is None:
            raise ValueError(f"{cfg_dir}: filter.xml has no <Ramp> section")
        block = out[span[0]:span[1]]
        if re.search(r"<isHamEnable>\s*0\s*</isHamEnable>", block):
            block = re.sub(r"<isHamEnable>\s*0\s*</isHamEnable>", "<isHamEnable>1</isHamEnable>", block, count=1)
            changes.append("ramp.isHamEnable 0->1")
        for tag, value in (("paraHam1", "0.54"), ("paraHam2", "0.46")):
            if re.search(rf"<{tag}>", block):
                block = re.sub(rf"(<{tag}>)[^<]*(</{tag}>)", rf"\g<1>{value}\g<2>", block, count=1)
                changes.append(f"ramp.{tag}={value}")
        return out[:span[0]] + block + out[span[1]:]

    _patch_filter_xml(cfg_dir, patch)
    return changes


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


def apply_fdk_para(cfg_dir: str | Path, overrides: dict[str, float]) -> list[str]:
    """Overwrite fdkPara.xml fields (e.g. ``xNum``, ``zSpacing``) in a workspace copy.

    Only the fields present in *overrides* are touched; the rest of the file is preserved
    byte-for-byte (via a single text replacement per field).  Returns the field names that
    were actually replaced.  A field missing from fdkPara.xml raises ``ValueError``.
    """
    path = Path(cfg_dir) / "fdkPara.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fdkPara.xml in {cfg_dir}")
    text = path.read_text(encoding="utf-8")
    changed: list[str] = []
    for name, value in overrides.items():
        pattern = re.compile(rf"<{re.escape(name)}(?:\s[^>]*)?>.*?</{re.escape(name)}>", re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"field <{name}> not found in fdkPara.xml")
        text = pattern.sub(f"<{name}>{value:g}</{name}>", text)
        changed.append(name)
    if changed:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return changed


def apply_filter_file(cfg_dir: str | Path, filter_file: str | Path) -> bool:
    """Replace the workspace filter.xml copy with the group's standard file.

    The standard file is copied verbatim (no runtime parameter overrides); returns
    ``True`` when the copy actually replaced the existing filter.xml.
    """
    target = Path(cfg_dir) / "filter.xml"
    source = Path(filter_file)
    if not source.is_file():
        raise ValueError(f"standard filter file not found: {source}")
    same = target.is_file() and target.read_bytes() == source.read_bytes()
    shutil.copy2(source, target)
    return not same


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


def _workspace_context(study: Study, output_dir: Path, options: VariantOptions, prefix_parts: list[str]):
    if options.keep_workspace:
        parent = output_dir.joinpath(*prefix_parts) / "workspace"
        if parent.exists() and options.overwrite:
            shutil.rmtree(parent)
        parent.mkdir(parents=True, exist_ok=True)
        return parent
    return Path(tempfile.mkdtemp(prefix=f"xday12_{study.scan_id[:16]}_", dir=options.workspace_root))


def reconstruct_study(
    study: Study,
    group: ReconGroup,
    output_dir: str | Path,
    options: VariantOptions,
    *,
    reconstruct_fn: Callable[..., Any] = _default_reconstruct,
    sdk: Any = None,
) -> dict[str, Any]:
    """Reconstruct the 6 output variants (``OUTPUT_NAMES``) of one study for one group.

    Volumes are stored under ``<output_root>/[<top_dir>/]<group>/<case_id>/<study_uid>/``
    where ``<group>`` is the group name (e.g. ``Mode2_0808``) and the optional top-level
    directory comes from ``options.top_dir`` (``PDL`` for the default source root).

    The group's ``fdk_para`` geometry is applied to every subset's fdkPara.xml copy and its
    standard ``filter_file`` (if any) replaces every subset's filter.xml copy verbatim, so a
    single scan can be rebuilt at different FOVs (e.g. Mode2 at 1609 and 0808) without
    touching the source files.  Reconstructs the full subset from a plain copy of the
    raw/config and the even/odd subsets from sliced workspaces; applies the even-step rule
    to even/odd.
    """
    output_root = Path(output_dir)
    prefix_parts = ([options.top_dir] if options.top_dir else []) + [group.name, study.case_id, study.study_uid]
    case_out = output_root.joinpath(*prefix_parts)
    rel_prefix = "/".join(prefix_parts)
    expected = output_names_for_mode(study.mode, options.combos, options.split_modes)
    if not options.overwrite and all((case_out / name).is_file() for name in expected) and (case_out / "metadata.json").is_file():
        try:
            import json
            prev = json.loads((case_out / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            prev = None
        # A previous output produced under *different* PWLS settings is not complete for
        # this run: fall through to a full reconstruction instead of skipping it.
        prev_pwls = (prev or {}).get("pwls")
        if options.pwls_scale is None:
            if prev_pwls:
                prev = None
        elif not (isinstance(prev_pwls, dict) and prev_pwls.get("enabled")
                  and abs(float(prev_pwls.get("scale", -1.0)) - options.pwls_scale) < 1e-12):
            prev = None
        if prev is not None:
            record = _base_record(study, options, group)
            # Inherit shape/dtype/outputs from the previous run's metadata.json so that
            # verify_volumes can re-check the existing volumes (size check) instead of
            # flagging them as "record missing shape/outputs" on every resumable run.
            for key in ("shape", "dtype", "spacing_zyx", "outputs", "per_view_sliced",
                        "per_view_warnings", "step_fixes", "fdk_changed", "fdk_para",
                        "filter_file", "filter_replaced", "pwls"):
                if key in prev:
                    record[key] = prev[key]
            record.setdefault("outputs", {name: f"{rel_prefix}/{name}" for name in expected})
            record.update({"status": "success", "skipped": True})
            return record
    if not study.nviews:
        raise ValueError("study has no projections")
    if study.kv is None or study.ma is None:
        raise ValueError("kV/mA unknown; pass --kv/--ma or fix the raw directory name")

    case_out.mkdir(parents=True, exist_ok=True)
    split_subsets = any(m.casefold() == study.mode.casefold() for m in options.split_modes)
    per_view, per_view_warnings = validated_per_view(
        study.cfg_dir, study.nviews, options.per_view_files.get(study.mode, {})
    )
    # The SDK consumes only the projections declared by fdkPara's noView; raw dirs may
    # hold more frames (Mode1: 1046 raw vs noView=1020) which must be dropped, otherwise
    # the full subset would run with more projections than the per-view matrix rows.
    source_no_view = _read_no_view(study.cfg_dir / "fdkPara.xml")
    workspace = _workspace_context(study, output_root, options, prefix_parts)
    try:
        full_raw, full_cfg = build_full_workspace(
            study.raw_dir, study.cfg_dir, workspace / "full", options.link_mode,
            max_views=source_no_view,
        )
        workspaces = {"full": (full_raw, full_cfg)}
        if split_subsets:
            even_indices, odd_indices = split_projection_indices(study.nviews)
            even_ws = build_subset_workspace(study.raw_dir, study.cfg_dir, even_indices, workspace / "even", options.link_mode, per_view)
            odd_ws = build_subset_workspace(study.raw_dir, study.cfg_dir, odd_indices, workspace / "odd", options.link_mode, per_view)
            workspaces["even"] = (even_ws.raw_dir, even_ws.cfg_dir)
            workspaces["odd"] = (odd_ws.raw_dir, odd_ws.cfg_dir)

        fdk_changed = apply_fdk_para(full_cfg, group.fdk_para)
        filter_replaced = False
        step_fixes: dict[str, list[tuple[str, int, int]]] = {}
        for subset, (_raw, _cfg) in workspaces.items():
            if subset != "full":
                apply_fdk_para(_cfg, group.fdk_para)
            if group.filter_file:
                if apply_filter_file(_cfg, group.filter_file):
                    filter_replaced = True
            if subset != "full":
                fixes = fix_subsample_step(_cfg, options.step_modules)
                if fixes:
                    step_fixes[subset] = fixes

        pwls_info = None
        if options.pwls_scale is not None:
            beta = options.pwls_scale * brightness_mean(study.cfg_dir)
            for _raw, _cfg in workspaces.values():
                patch_pwls_hamming(_cfg, beta)
            pwls_info = {"enabled": True, "scale": options.pwls_scale, "beta": beta,
                         "brightness_mean": beta / options.pwls_scale}

        common = dict(xday_install_dir=options.xday_dir, dtype=options.dtype, show_progress=False, sdk=sdk)
        combos_by_tag = dict(options.combos)
        wanted = {
            (s, t)
            for (s, t) in OUTPUT_NAMES
            if s in subsets_for_mode(study.mode, options.split_modes) and t in combos_by_tag
        }
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
            outputs[name] = f"{rel_prefix}/{name}"

        record = _base_record(study, options, group)
        record.update({
            "status": "success",
            "skipped": False,
            "shape": list(shapes["full", "none"]),
            "dtype": "<f4",
            "spacing_zyx": spacings["full_none"],
            "outputs": outputs,
            "fdk_para": dict(group.fdk_para),
            "fdk_changed": fdk_changed,
            "filter_file": group.filter_file,
            "filter_replaced": filter_replaced,
            "pwls": pwls_info,
            "step_fixes": step_fixes,
            "per_view_sliced": dict(per_view),
            "per_view_warnings": per_view_warnings,
        })
        _write_json(case_out / "metadata.json", record)
        return record
    finally:
        if not options.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def _base_record(study: Study, options: VariantOptions, group: ReconGroup) -> dict[str, Any]:
    split = any(m.casefold() == study.mode.casefold() for m in options.split_modes)
    record = {
        "case_id": study.case_id,
        "study_uid": study.study_uid,
        "scan_id": study.scan_id,
        "mode": study.mode,
        "group": group.name,
        "raw_dir": str(study.raw_dir),
        "cfg_dir": str(study.cfg_dir),
        "kv": study.kv,
        "ma": study.ma,
        "nviews": study.nviews,
        "combos": [{"tag": tag, "post_procs": chain} for tag, chain in options.combos],
        "split": split,
    }
    if split:
        even_idx, odd_idx = split_projection_indices(study.nviews)
        record["even_views"] = len(even_idx)
        record["odd_views"] = len(odd_idx)
    if options.top_dir:
        record["top_dir"] = options.top_dir
    return record


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json_dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
