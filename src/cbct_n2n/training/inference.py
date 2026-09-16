from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cbct_n2n.data.dataset import pad_to_multiple, unpad
from .metrics import compute_metrics


def resolve_input_kinds(input_kind: str) -> tuple[str, ...]:
    if input_kind == "both":
        return ("even", "odd")
    if input_kind in {"even", "odd", "full"}:
        return (input_kind,)
    raise ValueError(f"Unsupported input kind: {input_kind}")


@torch.no_grad()
def infer_volume(model, volume: np.ndarray, device: torch.device, batch_size: int = 4) -> np.ndarray:
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    padded_slices, paddings = [], []
    for image in source:
        padded, padding = pad_to_multiple(image.astype(np.float32))
        padded_slices.append(padded)
        paddings.append(padding)
    model.eval()
    outputs = []
    for start in range(0, len(padded_slices), batch_size):
        batch = torch.from_numpy(np.stack(padded_slices[start:start + batch_size])).unsqueeze(1).to(device)
        outputs.append(model(batch).detach().float().cpu().numpy()[:, 0])
    restored = []
    for image, padding in zip(np.concatenate(outputs, axis=0), paddings):
        restored.append(unpad(image, padding))
    return np.stack(restored).astype(np.float32)


@torch.no_grad()
def infer_volume_3d(model, volume: np.ndarray, voxel_depth: int,
                    device: torch.device, batch_size: int = 2,
                    use_fp16: bool = False) -> np.ndarray:
    """Whole-volume 3D inference via a non-overlapping D-slice sliding window.

    The model maps a ``[1, D, H, W]`` slab to a same-D-length denoised slab.
    The volume is split into consecutive ``voxel_depth``-slice windows strided
    by ``voxel_depth`` (matching the non-overlapping training-time placement).
    The final window is zero-padded to a full ``voxel_depth`` when ``n %
    voxel_depth != 0``; its extra (padded) output slices are discarded so the
    returned volume has exactly ``n`` slices -- the same as the input.

    Unlike the 2D :func:`infer_volume`, this shares the model's D-axis
    receptive field, so the architecture must expect a ``[B, C, D, H, W]``
    input (e.g. :class:`cbct_n2n.models.UNet3D`).
    """
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    if voxel_depth < 1:
        raise ValueError("voxel_depth must be positive")

    n = int(source.shape[0])
    model.eval()

    # Pad H/W once; the padding is identical for every slice (H/W uniform).
    padded_slices: list[np.ndarray] = []
    padding = None
    for i in range(n):
        padded, pad = pad_to_multiple(source[i].astype(np.float32))
        padded_slices.append(padded)
        if padding is None:
            padding = pad
    padded = np.stack(padded_slices, axis=0)  # [n, Hp, Wp]
    Hp, Wp = padded.shape[-2], padded.shape[-1]

    # Non-overlapping windows strided by voxel_depth; the final window is
    # zero-padded to a full voxel_depth (its extra output slices are dropped).
    z_starts: list[int] = list(range(0, n, voxel_depth))
    windows: list[np.ndarray] = []
    for z_start in z_starts:
        win = np.zeros((voxel_depth, Hp, Wp), dtype=np.float32)
        z_end = min(z_start + voxel_depth, n)
        win[:z_end - z_start] = padded[z_start:z_end]
        windows.append(win)

    preds: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(np.stack(windows[start:start + batch_size])).unsqueeze(1).to(device)
        if use_fp16:
            batch = batch.half()
        pred = model(batch).detach().float().cpu().numpy()[:, 0]  # [B, D, Hp, Wp]
        preds.append(pred)
    all_pred = np.concatenate(preds, axis=0)  # [num_windows, D, Hp, Wp]

    out = np.zeros_like(padded)  # [n, Hp, Wp]
    for k, z_start in enumerate(z_starts):
        z_end = min(z_start + voxel_depth, n)
        out[z_start:z_end] = all_pred[k][:z_end - z_start]

    restored = [unpad(slice_, padding) for slice_ in out]
    return np.stack(restored).astype(np.float32)


def _trapezoid_window(size: int, ramp: int) -> np.ndarray:
    """1D trapezoid window: ramp up over ``ramp``, flat 1 in the middle, ramp down.

    Used to feather tile edges during overlap-tile fusion so that adjacent
    tiles blend smoothly instead of leaving a hard seam.
    """
    w = np.ones(size, dtype=np.float32)
    r = min(max(int(ramp), 0), size // 2)
    for i in range(r):
        v = (i + 0.5) / r
        w[i] = v
        w[size - 1 - i] = v
    return w


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Start indices covering ``[0, total)`` with tiles of length ``tile``.

    Tiles are placed every ``stride``; the final tile is right-aligned so the
    far end is always covered.  Returns ``[0]`` when ``total <= tile``.
    """
    if total <= tile:
        return [0]
    starts: list[int] = []
    s = 0
    while s + tile < total:
        starts.append(s)
        s += stride
    starts.append(total - tile)
    return starts


@torch.no_grad()
def infer_volume_3d_tiled(model, volume: np.ndarray, voxel_depth: int, patch_hw: int,
                          overlap_d: int, overlap_hw: int, device: torch.device,
                          batch_size: int = 2, use_fp16: bool = False) -> np.ndarray:
    """Whole-volume 3D inference via overlapping H/W + D tiles with ramp fusion.

    The volume is covered by ``voxel_depth x patch_hw x patch_hw`` tiles.  Along
    D the tiles overlap by ``overlap_d`` slices (stride ``voxel_depth -
    overlap_d``); along H and W they overlap by ``overlap_hw`` pixels.  Each
    tile's prediction is weighted by a 3D-separable trapezoid window (flat 1 in
    the interior, linear ramp down at the edges) and accumulated; the final
    volume is the accumulated prediction divided by the accumulated weight,
    which removes the block seams.  Output has exactly the input's ``(z, y, x)``
    shape.
    """
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    if voxel_depth < 1 or patch_hw < 1:
        raise ValueError("voxel_depth and patch_hw must be positive")
    if overlap_d < 0 or overlap_d >= voxel_depth:
        raise ValueError("overlap_d must satisfy 0 <= overlap_d < voxel_depth")
    if overlap_hw < 0 or overlap_hw >= patch_hw:
        raise ValueError("overlap_hw must satisfy 0 <= overlap_hw < patch_hw")

    n, H, W = source.shape
    model.eval()

    # Pad D/H/W so tiles land cleanly: D to at least voxel_depth, H/W to at
    # least patch_hw and to a multiple of 32 (UNet's 2x pooling divisor).
    Np = max(n, voxel_depth)
    Hp = int(np.ceil(max(H, patch_hw) / 32) * 32)
    Wp = int(np.ceil(max(W, patch_hw) / 32) * 32)
    padded = source.astype(np.float32)
    if (Np, Hp, Wp) != (n, H, W):
        padded = np.pad(padded, ((0, Np - n), (0, Hp - H), (0, Wp - W)), mode="constant")

    stride_d = voxel_depth - overlap_d
    stride_hw = patch_hw - overlap_hw
    d_starts = _tile_starts(Np, voxel_depth, stride_d)
    h_starts = _tile_starts(Hp, patch_hw, stride_hw)
    w_starts = _tile_starts(Wp, patch_hw, stride_hw)

    w_d = _trapezoid_window(voxel_depth, overlap_d)
    w_h = _trapezoid_window(patch_hw, overlap_hw)
    w_w = _trapezoid_window(patch_hw, overlap_hw)
    tile_weight = (w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]).astype(np.float32)

    out = np.zeros((Np, Hp, Wp), dtype=np.float32)
    weight = np.zeros((Np, Hp, Wp), dtype=np.float32)

    tiles = [(d0, h0, w0) for d0 in d_starts for h0 in h_starts for w0 in w_starts]
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        slabs = np.stack([
            padded[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw]
            for (d0, h0, w0) in batch_tiles
        ]).astype(np.float32)  # [B, D, Ht, Wt]
        batch = torch.from_numpy(slabs).unsqueeze(1).to(device)
        if use_fp16:
            batch = batch.half()
        pred = model(batch).detach().float().cpu().numpy()[:, 0]  # [B, D, Ht, Wt]
        for (d0, h0, w0), p in zip(batch_tiles, pred):
            out[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw] += p * tile_weight
            weight[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw] += tile_weight

    fused = out / np.maximum(weight, 1e-8)
    return fused[:n, :H, :W].astype(np.float32)


def evaluate_volume(prediction: np.ndarray, target: np.ndarray, metric_data_range: float) -> dict[str, float]:
    return compute_metrics(prediction, target, data_range=metric_data_range)


def write_raw_volume(path: str | Path, volume: np.ndarray) -> None:
    """Write a (z, y, x) volume as headerless little-endian float32 RAW."""
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must have shape (z, y, x)")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        for image in source:
            np.asarray(image, dtype="<f4").tofile(stream)
