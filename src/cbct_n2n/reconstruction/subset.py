from __future__ import annotations

import os
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_RAW_NAME = re.compile(r"(\d+)\.raw$", re.IGNORECASE)


@dataclass(frozen=True)
class SubsetWorkspace:
    path: Path
    raw_dir: Path
    cfg_dir: Path
    raw_files: tuple[Path, ...]
    indices: tuple[int, ...]
    view_count: int


def numbered_raw_files(raw_dir: os.PathLike[str] | str) -> list[Path]:
    path = Path(raw_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {path}")
    files = [p for p in path.iterdir() if p.is_file() and _RAW_NAME.fullmatch(p.name)]
    return sorted(files, key=lambda p: int(_RAW_NAME.fullmatch(p.name).group(1)))


def split_projection_indices(count: int) -> tuple[list[int], list[int]]:
    if count < 1:
        raise ValueError("Projection count must be positive")
    return list(range(0, count, 2)), list(range(1, count, 2))


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


def _subset_binary(path: Path, output: Path, indices: Iterable[int], view_count: int, item_width: int) -> None:
    raw = path.read_bytes()
    if item_width <= 0 or len(raw) % item_width or len(raw) // item_width != view_count:
        raise ValueError(f"{path.name} does not contain {view_count} rows of {item_width} bytes")
    rows = [raw[i * item_width : (i + 1) * item_width] for i in indices]
    output.write_bytes(b"".join(rows))


def _update_no_view(path: Path, view_count: int) -> None:
    tree = ET.parse(path)
    node = tree.getroot().find(".//noView")
    if node is None:
        raise ValueError(f"Missing noView in {path}")
    node.text = str(view_count)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_subset_workspace(
    raw_dir: os.PathLike[str] | str,
    cfg_dir: os.PathLike[str] | str,
    indices: Iterable[int],
    destination: os.PathLike[str] | str,
    link_mode: str = "auto",
    per_view_files: dict[str, int | None] | None = None,
) -> SubsetWorkspace:
    raw_files = numbered_raw_files(raw_dir)
    cfg_path = Path(cfg_dir)
    if not cfg_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {cfg_path}")
    selected = tuple(int(i) for i in indices)
    if not selected or any(i < 0 or i >= len(raw_files) for i in selected):
        raise ValueError("indices must be non-empty and within the projection range")
    if len(set(selected)) != len(selected):
        raise ValueError("indices must be unique")
    out = Path(destination)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Subset destination is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    out_raw = out / "raw"
    out_cfg = out / "config"
    out_raw.mkdir()
    out_cfg.mkdir()
    for new_i, source_i in enumerate(selected):
        target = out_raw / f"{new_i:05d}.raw"
        _copy_or_link(raw_files[source_i], target, link_mode)
    for item in cfg_path.iterdir():
        if item.is_file():
            shutil.copy2(item, out_cfg / item.name)
    fdk = out_cfg / "fdkPara.xml"
    if not fdk.exists():
        raise ValueError(f"Missing fdkPara.xml in {cfg_path}")
    _update_no_view(fdk, len(selected))
    # An empty mapping is a legitimate "no per-view slicing" declaration (e.g. Mode1 keeps
    # projectionMatrix.bin as a shared whole file); only fall back when None was passed.
    widths = per_view_files if per_view_files is not None else {"projectionMatrix.bin": 21 * 4}
    for name, width in widths.items():
        if width is None:
            continue
        source = cfg_path / name
        if source.exists():
            _subset_binary(source, out_cfg / name, selected, len(raw_files), width)
    return SubsetWorkspace(out, out_raw, out_cfg, tuple(numbered_raw_files(out_raw)), selected, len(selected))
