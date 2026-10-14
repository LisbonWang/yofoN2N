from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np

PAIRING_METHOD = "angle_range_half_turn_v1"


def half_turn_pair_indices(view_count: int, *, angle_range_turns: float) -> list[tuple[int, int]]:
    """Return in-range frame pairs separated by half a turn of the configured scan."""
    if view_count < 2:
        raise ValueError(f"180-degree pairing requires at least two views, got {view_count}")
    if not np.isfinite(angle_range_turns) or angle_range_turns <= 0:
        raise ValueError(f"angleRange must be finite and positive, got {angle_range_turns}")
    frame_offset = int(np.floor(view_count * 0.5 / angle_range_turns + 0.5))
    if frame_offset < 1 or frame_offset >= view_count:
        raise ValueError(
            f"angleRange={angle_range_turns} turns gives invalid 180-degree frame offset "
            f"{frame_offset} for {view_count} views"
        )
    return [(first, first + frame_offset) for first in range(view_count - frame_offset)]


def read_angle_range_turns(cfg_dir: str | Path) -> float:
    """Read the total scan angle from fdkPara.xml, expressed in full-turn units."""
    path = Path(cfg_dir) / "fdkPara.xml"
    root = ET.parse(path).getroot()
    node = root.find(".//angleRange")
    if node is None or node.text is None:
        raise ValueError(f"Missing <angleRange> in {path}")
    try:
        turns = float(node.text)
    except ValueError as exc:
        raise ValueError(f"Invalid <angleRange> in {path}: {node.text!r}") from exc
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError(f"angleRange in {path} must be finite and positive, got {turns}")
    return turns


def residual_column_scores(paired_projections: Iterable[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return mean row-wise std of absolute residuals for paired projection frames."""
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    for first, opposite in paired_projections:
        first = np.asarray(first, dtype=np.float32)
        opposite = np.asarray(opposite, dtype=np.float32)
        if first.ndim != 2 or first.shape != opposite.shape:
            raise ValueError("paired projections must have matching 2D shapes")
        residual = np.abs(first - opposite).astype(np.float64, copy=False)
        count += 1
        if mean is None:
            mean = np.zeros_like(residual)
            m2 = np.zeros_like(residual)
        delta = residual - mean
        mean += delta / count
        m2 += delta * (residual - mean)
    if count == 0 or m2 is None:
        raise ValueError("at least one paired projection is required")
    return np.sqrt(np.maximum(m2 / count, 0.0)).mean(axis=0)


def projected_origin_column(Ks: np.ndarray, Ts: np.ndarray) -> float:
    """Project (0, 0, 0) through each cbkits view geometry and average detector u."""
    Ks = np.asarray(Ks, dtype=np.float64)
    Ts = np.asarray(Ts, dtype=np.float64)
    if Ks.ndim != 3 or Ks.shape[1:] != (3, 3):
        raise ValueError(f"Ks must have shape (N, 3, 3), got {Ks.shape}")
    if Ts.shape != (len(Ks), 3, 4):
        raise ValueError(f"Ts must have shape ({len(Ks)}, 3, 4), got {Ts.shape}")
    camera_origin = Ts[:, :, 3]
    detector_origin = np.einsum("nij,nj->ni", Ks, camera_origin)
    denominator = detector_origin[:, 2]
    if np.any(np.isclose(denominator, 0.0)):
        raise ValueError("origin projects to infinity in at least one view")
    columns = detector_origin[:, 0] / denominator
    if not np.isfinite(columns).all():
        raise ValueError("origin projection contains non-finite coordinates")
    return float(columns.mean())


def correct_projection_u_offset(
    matrices: np.ndarray,
    *,
    anchor_column: int,
    origin_column: float,
    u_spacing_mm: float,
    max_distance_mm: float = 3.0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Apply one accepted u-principal-point shift to every 21-float projection row."""
    matrices = np.asarray(matrices, dtype=np.float32)
    if matrices.ndim != 2 or matrices.shape[1] != 21:
        raise ValueError(f"projection matrices must have shape (N, 21), got {matrices.shape}")
    if not np.isfinite(origin_column) or not np.isfinite(u_spacing_mm) or u_spacing_mm <= 0:
        raise ValueError("origin column must be finite and u spacing must be positive")
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0:
        raise ValueError("maximum distance must be finite and positive")

    offset_px = float(anchor_column) - float(origin_column)
    distance_mm = abs(offset_px) * float(u_spacing_mm)
    accepted = distance_mm < max_distance_mm
    applied_offset_px = offset_px if accepted else 0.0
    corrected = matrices.copy()
    if accepted:
        intrinsics = corrected[:, 12:].reshape(-1, 3, 3)
        intrinsics[:, 0, 2] += applied_offset_px
    info: dict[str, float | int | bool] = {
        "anchor_column": int(anchor_column),
        "origin_column": float(origin_column),
        "candidate_offset_px": offset_px,
        "candidate_offset_mm": offset_px * float(u_spacing_mm),
        "offset_px": applied_offset_px,
        "offset_mm": applied_offset_px * float(u_spacing_mm),
        "distance_mm": float(distance_mm),
        "accepted": bool(accepted),
    }
    return corrected, info


def estimate_and_apply_projection_u_offset(
    raw_dir: str | Path,
    cfg_dir: str | Path,
    mode: str,
    kv: float,
    ma: float,
    scratch_dir: str | Path,
    *,
    expected_views: int,
    max_distance_mm: float = 3.0,
) -> tuple[dict[str, float | int | bool], np.ndarray]:
    """Load cbkits projections, estimate the residual anchor, and return corrected rows."""
    from cbkits import XDayDualSourceModeData, XDayModeData

    if mode.casefold() == "mode1":
        dataset = XDayDualSourceModeData(cfg_dir, raw_dir, kV=kv, mA=ma)
        geometry = dataset.data_A
        projection_iterator = iter(dataset.iter_projs())
        view_count = geometry.frames_count
    else:
        dataset = XDayModeData(cfg_dir, raw_dir, kV=kv, mA=ma)
        geometry = dataset
        projection_iterator = iter(dataset.iter_projs())
        view_count = dataset.frames_count
    if view_count != expected_views:
        raise ValueError(f"cbkits view count {view_count} != effective fdkPara noView {expected_views}")
    angle_range_turns = read_angle_range_turns(cfg_dir)
    pairs = half_turn_pair_indices(view_count, angle_range_turns=angle_range_turns)
    frame_offset = pairs[0][1]
    pair_count = len(pairs)
    first = next(projection_iterator)
    image_shape = tuple(np.asarray(first).shape)
    if len(image_shape) != 2:
        raise ValueError(f"cbkits projection must be 2D, got {image_shape}")
    scratch_path = Path(scratch_dir) / "projection_first_pair_side.f32"
    first_side = np.memmap(scratch_path, mode="w+", dtype=np.float32, shape=(pair_count, *image_shape))
    try:
        first_side[0] = first

        def paired_frames():
            for view_index in range(1, view_count):
                projection = next(projection_iterator)
                if view_index < pair_count:
                    first_side[view_index] = projection
                if view_index >= frame_offset:
                    yield first_side[view_index - frame_offset], projection

        scores = residual_column_scores(paired_frames())
    finally:
        del first_side
        try:
            scratch_path.unlink()
        except FileNotFoundError:
            pass

    anchor_column = int(np.argmin(scores))
    origin_column = projected_origin_column(geometry.Ks[:view_count], geometry.Ts[:view_count])
    matrices_path = Path(cfg_dir) / "projectionMatrix.bin"
    matrices = np.fromfile(matrices_path, dtype="<f4")
    if matrices.size != expected_views * 21:
        raise ValueError(
            f"projectionMatrix.bin has {matrices.size} floats; expected {expected_views} x 21"
        )
    matrices = matrices.reshape(expected_views, 21)
    corrected, info = correct_projection_u_offset(
        matrices,
        anchor_column=anchor_column,
        origin_column=origin_column,
        u_spacing_mm=float(geometry.u_spacing),
        max_distance_mm=max_distance_mm,
    )
    info["anchor_score"] = float(scores[anchor_column])
    info["u_spacing_mm"] = float(geometry.u_spacing)
    info["pairing_method"] = PAIRING_METHOD
    info["angle_range_turns"] = angle_range_turns
    info["source_view_count"] = int(view_count)
    info["pair_offset_frames"] = int(frame_offset)
    info["paired_view_count"] = int(pair_count)
    info["pair_angle_degrees"] = float(360.0 * angle_range_turns * frame_offset / view_count)
    return info, corrected


def write_projection_matrix(cfg_dir: str | Path, matrices: np.ndarray) -> None:
    """Atomically replace a workspace projection matrix without touching its source."""
    path = Path(cfg_dir) / "projectionMatrix.bin"
    temporary = path.with_name(path.name + ".tmp")
    np.asarray(matrices, dtype="<f4").tofile(temporary)
    os.replace(temporary, path)
