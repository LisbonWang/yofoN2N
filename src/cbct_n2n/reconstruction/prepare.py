from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .discovery import ScanCase
from .subset import build_subset_workspace, numbered_raw_files, split_projection_indices


@dataclass(frozen=True)
class PrepareOptions:
    xday_dir: Path
    post_procs: str = ""
    dtype: str = "float32"
    link_mode: str = "auto"
    overwrite: bool = False
    keep_workspace: bool = False
    per_view_files: dict[str, int | None] | None = None


@dataclass(frozen=True)
class PairRecord:
    case_id: str
    even_path: Path
    odd_path: Path
    full_path: Path
    metadata_path: Path
    shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]


def _parse_kv_ma(name: str) -> tuple[float, float]:
    match = re.search(r"_(\d+(?:\.\d+)?)kV(\d+(?:\.\d+)?)mA(?:(\d+(?:\.\d+)?)cc)?(?:_|$)", name)
    if not match:
        raise ValueError(f"Cannot parse kV/mA from raw directory name: {name}")
    kv, ma, cc = match.groups()
    return float(kv), float(ma) * (float(cc) if cc else 1.0)


def _default_reconstruct(*args: Any, **kwargs: Any):
    try:
        from xday import invoke_xday
    except ImportError as exc:
        raise RuntimeError("xday-kits is required for reconstruction; install it with pip install -e D:\\workspace2\\xday_kits") from exc
    return invoke_xday(*args, **kwargs)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def _spacing(result: Any) -> tuple[float, float, float]:
    info = result.volume_info
    return float(info.z_spacing), float(info.y_spacing), float(info.x_spacing)


def prepare_case(
    case: ScanCase,
    output_dir: str | Path,
    options: PrepareOptions,
    *,
    reconstruct_fn: Callable[..., Any] = _default_reconstruct,
) -> PairRecord:
    case_output = Path(output_dir) / case.case_id
    even_path, odd_path, full_path = case_output / "even.npy", case_output / "odd.npy", case_output / "full.npy"
    metadata_path = case_output / "pair.json"
    if not options.overwrite and any(p.exists() for p in (even_path, odd_path, full_path, metadata_path)):
        raise FileExistsError(f"Pair output already exists: {case_output}")
    case_output.mkdir(parents=True, exist_ok=True)
    files = numbered_raw_files(case.raw_dir)
    even_indices, odd_indices = split_projection_indices(len(files))
    if not odd_indices or abs(len(even_indices) - len(odd_indices)) > 1:
        raise ValueError(f"Cannot form balanced subsets from {len(files)} projections")
    workspace_parent = case_output / "workspace" if options.keep_workspace else None
    temp_context = None
    if workspace_parent is None:
        temp_context = tempfile.TemporaryDirectory(prefix=f"cbct_n2n_{case.case_id[:12]}_")
        workspace_parent = Path(temp_context.name)
    else:
        if workspace_parent.exists() and options.overwrite:
            shutil.rmtree(workspace_parent)
        workspace_parent.mkdir(parents=True, exist_ok=True)
    try:
        even_workspace = build_subset_workspace(case.raw_dir, case.cfg_dir, even_indices, workspace_parent / "even", options.link_mode, options.per_view_files)
        odd_workspace = build_subset_workspace(case.raw_dir, case.cfg_dir, odd_indices, workspace_parent / "odd", options.link_mode, options.per_view_files)
        kv, ma = _parse_kv_ma(case.raw_dir.name)
        common = dict(xday_install_dir=options.xday_dir, post_procs=options.post_procs, dtype=options.dtype)
        full_result = reconstruct_fn(case.raw_dir, case.cfg_dir, kv, ma, **common)
        even_result = reconstruct_fn(even_workspace.raw_dir, even_workspace.cfg_dir, kv, ma, **common)
        odd_result = reconstruct_fn(odd_workspace.raw_dir, odd_workspace.cfg_dir, kv, ma, **common)
        full = np.asarray(full_result.volume)
        even = np.asarray(even_result.volume)
        odd = np.asarray(odd_result.volume)
        full_spacing, even_spacing, odd_spacing = _spacing(full_result), _spacing(even_result), _spacing(odd_result)
        if full.ndim != 3 or full.shape != even.shape or full.shape != odd.shape:
            raise ValueError(f"Reconstructed shape mismatch: full={full.shape}, even={even.shape}, odd={odd.shape}")
        if full_spacing != even_spacing or full_spacing != odd_spacing:
            raise ValueError(f"Reconstructed spacing mismatch: full={full_spacing}, even={even_spacing}, odd={odd_spacing}")
        _atomic_save_npy(full_path, full)
        _atomic_save_npy(even_path, even)
        _atomic_save_npy(odd_path, odd)
        metadata = {
            "case_id": case.case_id,
            "mode": case.mode,
            "raw_dir": str(case.raw_dir),
            "cfg_dir": str(case.cfg_dir),
            "even_path": str(even_path.resolve()),
            "odd_path": str(odd_path.resolve()),
            "full_path": str(full_path.resolve()),
            "shape": list(full.shape),
            "dtype": str(full.dtype),
            "spacing_zyx": list(full_spacing),
            "full_projections": [file.name for file in files],
            "even_projections": [files[i].name for i in even_indices],
            "odd_projections": [files[i].name for i in odd_indices],
            "reconstruction": {"kv": kv, "ma": ma, "post_procs": options.post_procs},
        }
        temp_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
        temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_metadata, metadata_path)
        return PairRecord(case.case_id, even_path, odd_path, full_path, metadata_path, tuple(full.shape), full_spacing)
    finally:
        if temp_context is not None:
            temp_context.cleanup()
