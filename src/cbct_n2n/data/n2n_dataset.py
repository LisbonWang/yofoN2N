from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import pad_to_multiple

#: logical subset names that exist per study (variants only exist for "full")
SUBSETS = ("even", "odd", "full")

#: residual augmentation tag -> variant volume key that carries the augmentation.
#: ``full.bhc`` etc. are the post-processed full-view volumes written by the
#: reconstruction pipeline; the residual is ``variant - full`` per slice.
RESIDUAL_VOLUMES = {
    "bhc": "full.bhc",
    "marfi": "full.marfi",
    "marfibhc": "full.marfi.bhc",
}

#: supervision_mode (legacy) -> explicit mapping list
SUPERVISION_MAPPINGS = {
    "full_view": ("even->full", "odd->full"),
    "half_view_n2n": ("even->odd", "odd->even"),
}

#: supported mapping directions, source subset -> target subset
ALLOWED_SOURCES = ("even", "odd")
ALLOWED_TARGETS = ("even", "odd", "full")


def _manifest_key(item: dict[str, Any]) -> str:
    """Unique key per manifest record (explicit ``key`` or legacy case_id)."""
    return str(item.get("key", item["case_id"]))


def _resolve_output(outputs: dict[str, Any], logical: str) -> str:
    """Resolve a logical volume name to its manifest path.

    Reconstruction index records store file names *with* the extension
    (``even.raw``, ``full.bhc.raw``); hand-written manifests may use plain keys
    (``even``, ``full.bhc``).  Both are accepted.
    """
    for candidate in (logical, f"{logical}.raw"):
        if candidate in outputs:
            return outputs[candidate]
    raise KeyError(f"no output volume matching {logical!r} (have: {sorted(outputs)})")


def _parse_mapping(spec: str) -> tuple[str, str]:
    source, arrow, target = spec.partition("->")
    if not arrow:
        raise ValueError(f"mapping must be '<source>-><target>', got {spec!r}")
    source, target = source.strip(), target.strip()
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"mapping source must be one of {ALLOWED_SOURCES}, got {source!r}")
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"mapping target must be one of {ALLOWED_TARGETS}, got {target!r}")
    if source == target:
        raise ValueError(f"mapping source and target must differ, got {spec!r}")
    return source, target


def _normalize_mappings(mappings: Iterable[str] | None, supervision_mode: str | None) -> tuple[tuple[str, str], ...]:
    """Resolve explicit mappings; fall back to translating a legacy supervision_mode."""
    if mappings is not None:
        specs = [m for m in mappings if m]
        if not specs:
            raise ValueError("mappings is empty")
        return tuple(dict.fromkeys(_parse_mapping(spec) for spec in specs))
    if supervision_mode is None:
        raise ValueError("either mappings or supervision_mode must be provided")
    try:
        legacy = SUPERVISION_MAPPINGS[str(supervision_mode)]
    except KeyError:
        raise ValueError(f"unknown supervision_mode {supervision_mode!r}; "
                         f"use mappings or one of {sorted(SUPERVISION_MAPPINGS)}") from None
    return tuple(dict.fromkeys(_parse_mapping(spec) for spec in legacy))


def _normalize_residuals(residuals: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve enabled residual tags.

    Selecting both ``bhc`` and ``marfi`` automatically adds the combined
    ``marfibhc`` residual (``full.marfi.bhc - full``).
    """
    if residuals is None:
        return ()
    tags = list(dict.fromkeys(str(tag) for tag in residuals if tag))
    unknown = set(tags) - set(RESIDUAL_VOLUMES)
    if unknown:
        raise ValueError(f"unknown residual tags {sorted(unknown)}; "
                         f"supported: {sorted(RESIDUAL_VOLUMES)}")
    if "bhc" in tags and "marfi" in tags and "marfibhc" not in tags:
        tags.append("marfibhc")
    return tuple(tags)


class N2NDataset(Dataset):
    """Configurable Noise2Noise slice dataset over RAW reconstruction volumes.

    Replaces :class:`RawPairedSliceDataset` with an explicit *mapping* design:

    - ``mappings`` selects the input -> target directions, e.g. ``even->odd``,
      ``odd->even``, ``even->full``, ``odd->full``.  Multiple directions may be
      enabled at once and their samples are mixed in the same epoch.  A legacy
      ``supervision_mode`` (``full_view`` / ``half_view_n2n``) is translated when
      no explicit mappings are given.
    - ``residuals`` optionally enables MARFI/BHC data augmentation.  The residual
      image ``R = variant - full`` (computed per slice from the *full-view*
      post-processed volumes ``full.bhc`` / ``full.marfi`` / ``full.marfi.bhc``)
      is added to both the source and the target slice, producing one extra
      training pair per enabled residual.  Selecting both ``bhc`` and ``marfi``
      also enables the combined ``marfibhc`` residual.

    Base (non-augmented) samples are always kept; residual samples are appended.
    """

    def __init__(self, manifest: dict[str, Any], manifest_path: str | Path, split: str, *,
                 mappings: Iterable[str] | None = None,
                 supervision_mode: str | None = None,
                 residuals: Iterable[str] | None = None,
                 augment: bool = False,
                 max_slices_per_case: int | None = None):
        if max_slices_per_case is not None and max_slices_per_case < 1:
            raise ValueError("max_slices_per_case must be positive when provided")
        resolved_mappings = _normalize_mappings(mappings, supervision_mode)
        residual_tags = _normalize_residuals(residuals)
        self.augment = augment
        self.mappings = resolved_mappings
        self.residual_tags = residual_tags
        base = Path(manifest_path).resolve().parent
        records = {_manifest_key(item): item for item in manifest["records"]}
        self.volumes: list[dict[str, Any]] = []
        self.slices: list[tuple[int, int, str, str, str | None]] = []
        for record_key in manifest["splits"][split]:
            item = records[record_key]
            if item.get("reconstruction_status") not in {None, "success"}:
                continue
            shape = tuple(int(value) for value in item["shape"])
            dtype = np.dtype(item.get("dtype", "<f4"))
            outputs = item["outputs"]
            # base volumes required by the active mappings; "full" is also needed
            # whenever residual augmentation is on (residual = variant - full)
            wanted = {subset for mapping in resolved_mappings for subset in mapping}
            if residual_tags:
                wanted.add("full")
            try:
                base_paths = {logical: base / _resolve_output(outputs, logical) for logical in wanted}
            except KeyError:
                # manifest lacks a required subset volume (e.g. Mode1 has no even/odd)
                continue
            memmaps: dict[str, np.memmap] = {}
            try:
                for logical, path in base_paths.items():
                    memmaps[logical] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
            except FileNotFoundError:
                # output file missing on disk -> drop the whole study
                continue
            # variants are optional: a missing post-processed volume only disables
            # its residual tag for this record, base samples are still kept
            for tag in residual_tags:
                variant = RESIDUAL_VOLUMES[tag]
                if variant in memmaps:
                    continue
                try:
                    memmaps[variant] = np.memmap(base / _resolve_output(outputs, variant),
                                                 dtype=dtype, mode="r", shape=shape)
                except (KeyError, FileNotFoundError):
                    continue
            index = len(self.volumes)
            case_id = str(item.get("case_id", record_key))
            self.volumes.append({"key": record_key, "case_id": case_id, "memmaps": memmaps})
            count = min(shape[0], max_slices_per_case or shape[0])
            available = [tag for tag in residual_tags
                         if RESIDUAL_VOLUMES[tag] in memmaps]
            for z in range(count):
                for source, target in resolved_mappings:
                    self.slices.append((index, z, source, target, None))
                    for tag in available:
                        self.slices.append((index, z, source, target, tag))
        if not self.volumes:
            raise ValueError(f"No usable reconstruction records in split {split!r}")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume_index, z, source_kind, target_kind, tag = self.slices[index]
        record = self.volumes[volume_index]
        memmaps = record["memmaps"]
        source = np.asarray(memmaps[source_kind][z], dtype=np.float32)
        target = np.asarray(memmaps[target_kind][z], dtype=np.float32)
        if tag is not None:
            variant = np.asarray(memmaps[RESIDUAL_VOLUMES[tag]][z], dtype=np.float32)
            full = np.asarray(memmaps["full"][z], dtype=np.float32)
            source = source + (variant - full)
            target = target + (variant - full)
        source, source_padding = pad_to_multiple(source)
        target, target_padding = pad_to_multiple(target)
        if source_padding != target_padding:
            raise ValueError("Source and target slice padding mismatch")
        source_tensor = torch.from_numpy(source).clone().unsqueeze(0)
        target_tensor = torch.from_numpy(target).clone().unsqueeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                source_tensor, target_tensor = torch.flip(source_tensor, (-1,)), torch.flip(target_tensor, (-1,))
            if torch.rand(()) < 0.5:
                source_tensor, target_tensor = torch.flip(source_tensor, (-2,)), torch.flip(target_tensor, (-2,))
            if source_tensor.shape[-2] == source_tensor.shape[-1]:
                turns = int(torch.randint(0, 4, ()).item())
                source_tensor, target_tensor = torch.rot90(source_tensor, turns, (-2, -1)), torch.rot90(target_tensor, turns, (-2, -1))
        return {
            "source": source_tensor,
            "target": target_tensor,
            "case_id": record["case_id"],
            "slice_index": z,
            "source_kind": source_kind,
            "target_kind": target_kind,
            "mapping": f"{source_kind}->{target_kind}",
            # base samples carry the empty string sentinel (default_collate
            # cannot mix None and str values inside one batch)
            "residual": tag or "",
        }
