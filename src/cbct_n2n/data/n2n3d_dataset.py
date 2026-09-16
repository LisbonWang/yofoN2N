"""3D-input / 3D-target dataset for the volumetric UNet.

Replaces the earlier ``N2NVoxelDataset`` with a contract that matches how
the 3D UNet is now trained:

* **Input to the model** (``source``): a stack of ``voxel_depth``
  consecutive axial slices starting at a *start index*, shaped
  ``[1, D, H, W]``.  ``D`` is free (any positive integer, default 16);
  there is no odd-depth constraint because the target is no longer a
  single centre slice.
* **Output to the trainer** (``target``): the **same D-slice window** from
  the supervision volume, shaped ``[1, D, H, W]``.  In N2N fashion each
  slice is the half-view partner / full-view reference for the matching
  slice (the same slices the parent :class:`N2NDataset` would have
  returned), so the rest of the supervision logic is unchanged.  The model
  denoises the whole input volume: output D == input D.

Windows are keyed by their *start* slice index: ``start`` runs from ``0``
with stride ``stride`` (default 4), and every window's end ``start +
voxel_depth`` stays ``<= record_depth`` (no out-of-range slice, no
zero-fill, no boundary replication).  Consecutive windows overlap by
``voxel_depth - stride`` slices.

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
        integer is accepted (default 16).  Must be ``>= 2 ** model.depth``
        for the fixed 2x-pooling UNet3D to be able to downsample along D.
    stride:
        Step between consecutive window starts along the D axis (default 4).
        Windows overlap by ``voxel_depth - stride`` slices when
        ``stride < voxel_depth``; ``stride == voxel_depth`` reproduces the
        non-overlapping tiling.  Every window end ``start + voxel_depth`` is
        kept ``<= record_depth`` so no slice is ever out of range.
    patch_size:
        Optional square H/W crop size (default ``None`` = use the full slice
        plane).  When set, each slab's H/W is cropped to ``patch_size x
        patch_size``: randomly positioned during training (``augment=True``)
        and centre-cropped otherwise.  This enables patch-based training on
        large slices (e.g. 256x256 from a 640x640 axial plane) while keeping
        the native voxel resolution.  ``patch_size`` must be ``>= 2 ** depth``
        (the fixed 2x pooling) so the model can downsample along H and W.
    All other keyword arguments are forwarded to :class:`N2NDataset`.
    """

    def __init__(self, *args: Any, voxel_depth: int = 16,
                 stride: int = 4, patch_size: int | None = None,
                 **kwargs: Any) -> None:
        if voxel_depth < 1:
            raise ValueError("voxel_depth must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        super().__init__(*args, **kwargs)
        self.voxel_depth = int(voxel_depth)
        self.stride = int(stride)
        self.patch_size = int(patch_size) if patch_size is not None else None
        # Windows are keyed by their *start* slice index.  ``start`` runs
        # from 0 with stride ``self.stride``, and every window end
        # ``start + voxel_depth <= count`` (no out-of-range slice).
        self.slices = self._build_start_slices()
        # Group id per sample == its volume index, so a grouped sampler can
        # shuffle studies first and samples within each study second.
        self.group_ids: list[int] = [s[0] for s in self.slices]

    # ---- helpers ----------------------------------------------------------

    def _build_start_slices(self) -> list[tuple[int, int, str, str, str | None]]:
        """Rebuild ``self.slices`` keyed by the window *start* slice index.

        The parent :class:`N2NDataset` enumerates one sample per anchor slice
        (``z = 0 .. count-1``).  For the volumetric model the window is placed
        by its start index instead: ``start`` runs from ``0`` with stride
        ``self.stride`` up to ``count - voxel_depth`` inclusive (when it
        aligns with the stride), so every window end ``start + voxel_depth``
        stays ``<= count`` -- no out-of-range slice.  Each
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
            # start from 0, stride self.stride, end (start + voxel_depth)
            # never past count.
            for start in range(0, count - self.voxel_depth + 1, self.stride):
                for source, target in self.mappings:
                    slices.append((index, start, source, target, None))
                    for tag in available:
                        slices.append((index, start, source, target, tag))
        return slices

    def _gather_slab(self, memmaps: dict[str, np.ndarray], start: int,
                     source_kind: str, target_kind: str, tag: str | None,
                     depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Read ``depth`` consecutive slices starting at ``start``.

        ``start`` is guaranteed to satisfy ``0 <= start <= record_depth -
        depth`` (see :meth:`_build_start_slices`), so every slice ``z`` is
        in-range and no zero-fill / boundary-replication is needed.

        Returns
        -------
        (source_slab, target_slab)
            Each shaped ``(depth, H, W)``.
        """
        H = int(memmaps[source_kind].shape[1])
        W = int(memmaps[source_kind].shape[2])
        source_buf = np.empty((depth, H, W), dtype=np.float32)
        target_buf = np.empty((depth, H, W), dtype=np.float32)

        for i in range(depth):
            z = int(start + i)
            src = np.asarray(memmaps[source_kind][z], dtype=np.float32)
            tgt = np.asarray(memmaps[target_kind][z], dtype=np.float32)
            if tag is not None:
                variant = np.asarray(memmaps[RESIDUAL_VOLUMES[tag]][z], dtype=np.float32)
                full = np.asarray(memmaps["full"][z], dtype=np.float32)
                src = src + (variant - full)
                tgt = tgt + (variant - full)
            source_buf[i] = src
            target_buf[i] = tgt
        return source_buf, target_buf

    def _crop_patch(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crop a square ``patch_size`` H/W region from both slabs.

        Training (``augment=True``) samples a random offset; evaluation uses a
        centre crop so validation is deterministic.  When the slice plane is
        already ``<= patch_size`` the slabs are returned unchanged.
        """
        if self.patch_size is None:
            return source, target
        H = int(source.shape[1])
        W = int(source.shape[2])
        ps = self.patch_size
        if H <= ps and W <= ps:
            return source, target
        if self.augment:
            h0 = int(np.random.randint(0, H - ps + 1)) if H > ps else 0
            w0 = int(np.random.randint(0, W - ps + 1)) if W > ps else 0
        else:
            h0 = (H - ps) // 2 if H > ps else 0
            w0 = (W - ps) // 2 if W > ps else 0
        return source[:, h0:h0 + ps, w0:w0 + ps], target[:, h0:h0 + ps, w0:w0 + ps]

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

        # Crop the H/W plane to the configured patch size (no-op when unset).
        source_slab, target_slab = self._crop_patch(source_slab, target_slab)

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
