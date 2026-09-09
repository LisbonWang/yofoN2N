"""3D-input / 3D-target dataset for the volumetric UNet.

Replaces the earlier ``N2NVoxelDataset`` with a contract that matches how
the 3D UNet is now trained:

* **Input to the model** (``source``): a stack of ``voxel_depth``
  consecutive axial slices starting at a *start index*, shaped
  ``[1, D, H, W]``.  ``D`` is free (any positive integer, default 22);
  there is no odd-depth constraint because the target is no longer a
  single centre slice.
* **Output to the trainer** (``target``): the **same D-slice window** from
  the supervision volume, shaped ``[1, D, H, W]``.  In N2N fashion each
  slice is the half-view partner / full-view reference for the matching
  slice (the same slices the parent :class:`N2NDataset` would have
  returned), so the rest of the supervision logic is unchanged.  The model
  denoises the whole input volume: output D == input D.

Windows are keyed by their *start* slice index: ``start`` runs from
``-identity_layers`` with stride 2 up to ``record_depth - 3``.  A window
head that sticks before slice 0 (``z < 0``) is **zero-filled**; a window
tail that sticks past the last slice (``z >= record_depth``) replicates the
boundary slice.

The first ``identity_layers`` slices of the D window are treated as
**continuity-constraint (identity) layers**: their target is set equal to
the source slice instead of the paired supervision volume, so the model
learns to leave them unchanged (``output == input``) rather than denoising
them.  These layers only provide D-axis context for the remaining slices;
at inference time they are discarded (see ``scripts/test_raw_3d.py``).

We still inherit from :class:`N2NDataset` so the manifest parsing,
mapping / residual logic, and split routing stay in one place.  The
parent's slice-level ``__getitem__`` is bypassed entirely; this class
performs its own per-slice IO using the parent's ``self.volumes`` /
``self.slices`` lists directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .dataset import pad_to_multiple
from .n2n_dataset import N2NDataset, RESIDUAL_VOLUMES


class N2N3DDataset(N2NDataset):
    """Return a 5D ``[1, D, H, W]`` source and a 5D ``[1, D, H, W]`` target.

    Parameters
    ----------
    voxel_depth:
        Number of consecutive axial slices in each returned volume, i.e.
        the D length of both ``source`` and ``target``.  Any positive
        integer is accepted (default 22).  Must be ``>= 2 ** model.depth``
        for the fixed 2x-pooling UNet3D to be able to downsample along D.
    identity_layers:
        Number of leading D slices treated as continuity-constraint
        (identity) layers.  For ``i < identity_layers`` the target slice is
        set equal to the source slice (``target == source``) so the model
        learns an identity mapping there; the remaining slices use the
        normal paired supervision volume.  Default ``0`` disables the
        behaviour (every slice is supervised normally).  Must satisfy
        ``0 <= identity_layers < voxel_depth``.  This also sets the window
        start: ``start`` begins at ``-identity_layers``, so the head
        overflow is either zero or exactly ``identity_layers`` slices.
    All other keyword arguments are forwarded to :class:`N2NDataset`.
    """

    def __init__(self, *args: Any, voxel_depth: int = 22,
                 identity_layers: int = 0, **kwargs: Any) -> None:
        if voxel_depth < 1:
            raise ValueError("voxel_depth must be positive")
        super().__init__(*args, **kwargs)
        self.voxel_depth = int(voxel_depth)
        self.identity_layers = int(identity_layers)
        if self.identity_layers < 0:
            raise ValueError("identity_layers must be non-negative")
        if self.identity_layers >= self.voxel_depth:
            raise ValueError("identity_layers must be < voxel_depth")
        # Windows are keyed by their *start* slice index instead of an anchor:
        # ``start`` runs from ``-identity_layers`` with stride 2 up to
        # ``record_depth - 3`` (inclusive when it aligns with the stride).
        self.slices = self._build_start_slices()

    # ---- helpers ----------------------------------------------------------

    def _build_start_slices(self) -> list[tuple[int, int, str, str, str | None]]:
        """Rebuild ``self.slices`` keyed by the window *start* slice index.

        The parent :class:`N2NDataset` enumerates one sample per anchor slice
        (``z = 0 .. count-1``).  For the volumetric model the window is placed
        by its start index instead: ``start`` runs from ``-identity_layers``
        with stride 2 up to ``record_depth - 3`` inclusive.  Each
        ``(start, mapping, residual)`` combination becomes one sample, reusing
        the parent's already-built ``self.volumes`` / ``self.mappings`` /
        ``self.residual_tags``.
        """
        slices: list[tuple[int, int, str, str, str | None]] = []
        for index, volume in enumerate(self.volumes):
            memmaps = volume["memmaps"]
            count = int(next(iter(memmaps.values())).shape[0])
            available = [tag for tag in self.residual_tags
                         if RESIDUAL_VOLUMES[tag] in memmaps]
            # start from -identity_layers, stride 2, up to record_depth-3.
            for start in range(-self.identity_layers, count - 2, 2):
                for source, target in self.mappings:
                    slices.append((index, start, source, target, None))
                    for tag in available:
                        slices.append((index, start, source, target, tag))
        return slices

    def _gather_slab(self, memmaps: dict[str, np.ndarray], start: int,
                     source_kind: str, target_kind: str, tag: str | None,
                     depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Read ``depth`` consecutive slices starting at ``start``.

        The window head (``z < 0``) is zero-filled; the window tail
        (``z >= record_depth``) replicates the boundary slice.  The first
        ``identity_layers`` in-range slices of the window are continuity-
        constraint (identity) layers: their target is set equal to the source
        slice instead of the paired supervision volume, so the model learns
        ``output == input`` there.

        Returns
        -------
        (source_slab, target_slab)
            Each shaped ``(depth, H, W)``.
        """
        record_depth = int(next(iter(memmaps.values())).shape[0])

        H = int(memmaps[source_kind].shape[1])
        W = int(memmaps[source_kind].shape[2])
        source_buf = np.empty((depth, H, W), dtype=np.float32)
        target_buf = np.empty((depth, H, W), dtype=np.float32)

        for i in range(depth):
            z = start + i
            if z < 0:
                # Head overflow: blank (zero) source and target.
                source_buf[i] = 0.0
                target_buf[i] = 0.0
                continue
            if z >= record_depth:
                z = record_depth - 1  # tail overflow: replicate boundary slice
            z = int(z)
            src = np.asarray(memmaps[source_kind][z], dtype=np.float32)
            if i < self.identity_layers:
                # Identity layer: target mirrors the (possibly residual-
                # augmented) source so the model learns output == input.
                if tag is not None:
                    variant = np.asarray(memmaps[RESIDUAL_VOLUMES[tag]][z], dtype=np.float32)
                    full = np.asarray(memmaps["full"][z], dtype=np.float32)
                    src = src + (variant - full)
                source_buf[i] = src
                target_buf[i] = src
            else:
                tgt = np.asarray(memmaps[target_kind][z], dtype=np.float32)
                if tag is not None:
                    variant = np.asarray(memmaps[RESIDUAL_VOLUMES[tag]][z], dtype=np.float32)
                    full = np.asarray(memmaps["full"][z], dtype=np.float32)
                    src = src + (variant - full)
                    tgt = tgt + (variant - full)
                source_buf[i] = src
                target_buf[i] = tgt
        return source_buf, target_buf

    # ---- Dataset API ------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        volume_index, start, source_kind, target_kind, tag = self.slices[index]
        record = self.volumes[volume_index]
        memmaps = record["memmaps"]

        # Read a slab of D slices (both source and target cover the same
        # window starting at ``start``).
        source_slab, target_slab = self._gather_slab(
            memmaps, int(start), source_kind, target_kind, tag, self.voxel_depth,
        )

        # Pad each slice's H/W to a multiple of 32 (UNet's pool divisor).
        # 640x640 is already aligned; skip the per-slice pad + re-stack when
        # H/W are multiples of 32 (source/target are already [D, H, W]).
        if source_slab.shape[1] % 32 != 0 or source_slab.shape[2] % 32 != 0:
            source_slab = np.stack(
                [pad_to_multiple(slice_)[0] for slice_ in source_slab], axis=0,
            )
            target_slab = np.stack(
                [pad_to_multiple(slice_)[0] for slice_ in target_slab], axis=0,
            )
        if source_slab.shape != target_slab.shape:
            raise ValueError("Source and target slab padding mismatch")

        # 5D source: [1, D, H, W].  from_numpy shares memory (no copy); the
        # augment flips/rotations below are non-in-place, so this is safe.
        source_tensor = torch.from_numpy(source_slab).unsqueeze(0)
        # 5D target: the same D-slice window from the supervision volume.
        target_tensor = torch.from_numpy(target_slab).unsqueeze(0)

        if self.augment:
            # depth-axis flip (only meaningful when D > 1) reverses the
            # anatomical orientation.  Apply to both source and target
            # consistently along the D axis (-3 for [1, D, H, W]).
            if self.voxel_depth > 1 and torch.rand(()) < 0.5:
                source_tensor = torch.flip(source_tensor, dims=(-3,))
                target_tensor = torch.flip(target_tensor, dims=(-3,))
            if torch.rand(()) < 0.5:
                source_tensor = torch.flip(source_tensor, dims=(-1,))
                target_tensor = torch.flip(target_tensor, dims=(-1,))
            if torch.rand(()) < 0.5:
                source_tensor = torch.flip(source_tensor, dims=(-2,))
                target_tensor = torch.flip(target_tensor, dims=(-2,))
            # 90-degree rotations only apply when H == W (same constraint
            # as the 2D dataset).  The D axis is not rotated.
            if source_tensor.shape[-2] == source_tensor.shape[-1]:
                turns = int(torch.randint(0, 4, ()).item())
                source_tensor = torch.rot90(source_tensor, turns, (-2, -1))
                target_tensor = torch.rot90(target_tensor, turns, (-2, -1))

        return {
            "source": source_tensor,
            "target": target_tensor,
            "case_id": record["case_id"],
            "slice_index": int(start),
            "source_kind": source_kind,
            "target_kind": target_kind,
            "mapping": f"{source_kind}->{target_kind}",
            "residual": tag or "",
        }


__all__ = ["N2N3DDataset"]
