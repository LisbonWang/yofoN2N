"""Create study-level train/val/test manifests from the reconstruction index.

Reads ``reconstruction-index.json`` (written by the batch reconstruction
pipeline) and writes a training manifest consumable by :class:`N2NDataset`.

Semantics (decided 2026-09-02 with the dataset owner):

- ``case_id`` identifies a *device* (e.g. ``YPDL-26A-0026``), NOT a patient.
  Every ``study`` (one ``study_uid``, one acquisition) under a device belongs
  to a different patient, so records do not share anatomy and can be split
  independently at study level.
- The composite ``key = f"{study_uid}::{group}"`` is written into every record
  because ``case_id`` repeats across groups (same device feeds Mode2_1609,
  Mode2_0808, ...) and ``study_uid`` alone repeats when one scan is
  reconstructed into several FOV groups.
- No cross-group/cross-device aggregation is enforced: the same scan rebuilt
  into different FOV groups produces visually different volumes and is not
  co-trained in practice, so each record is an independent unit.
- ``--test-groups`` moves every record of those groups into the test split
  (used for cross-mode evaluation, e.g. ``Mode3_1609``); the remaining records
  are split train/val/test by ``--ratios`` with a fixed ``--seed``.

The manifest must be written into the *same directory* as ``--output-root``:
datasets resolve the relative ``outputs`` paths from the manifest location.

Example (E1, on the training server 192.168.36.102):

    python scripts/create_recon_splits.py \\
      --index /data/raid0/yyc/AIReduceNoise/reconstruction-index.json \\
      --output-root /data/raid0/yyc/AIReduceNoise \\
      --groups Mode2_1609,Mode3_1609 \\
      --test-groups Mode3_1609 \\
      --out /data/raid0/yyc/AIReduceNoise/splits_1609.json \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

#: fields copied verbatim from the reconstruction-index record into the manifest
COPIED_FIELDS = (
    "case_id", "study_uid", "group", "mode", "kv", "ma", "nviews",
    "shape", "dtype", "outputs", "scan_id", "status",
)

#: base reconstruction volumes the N2N mappings rely on (for the presence report)
BASE_SUBSETS = ("even", "odd", "full")


def make_key(study_uid: str, group: str) -> str:
    """Composite key unique across groups (and across records by construction)."""
    return f"{study_uid}::{group}"


def build_manifest(index: dict[str, Any], *, groups: Iterable[str],
                   seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                   test_groups: Iterable[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(manifest, stats)`` for the selected groups.

    ``manifest`` follows the same top-level layout as
    :func:`cbct_n2n.data.manifest.create_case_splits`:
    ``{"seed", "ratios", "records": [...], "splits": {"train": [...], "val": [...], "test": [...]}}``
    plus provenance fields.  ``stats`` carries per-split/per-group counters for
    the console report.
    """
    wanted = set(groups)
    test = set(test_groups)
    unknown_groups = wanted - {item["group"] for item in index.get("records", [])}
    if unknown_groups:
        raise ValueError(f"groups not present in the index: {sorted(unknown_groups)}")
    for group in test:
        if group not in wanted:
            raise ValueError(f"--test-groups {group!r} is not among --groups")

    records: list[dict[str, Any]] = []
    for item in index["records"]:
        if item.get("group") not in wanted:
            continue
        if item.get("status") not in {None, "success"}:
            continue
        record = {field: item[field] for field in COPIED_FIELDS if field in item}
        record["key"] = make_key(record["study_uid"], record["group"])
        records.append(record)
    if not records:
        raise ValueError(f"no records selected for groups {sorted(wanted)}")

    keys = [record["key"] for record in records]
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"composite keys are not unique: {duplicated[:5]} ...")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must be three non-negative values summing to one")

    def split_assign(record: dict[str, Any]) -> str:
        return "test" if record["group"] in test else ""

    pool = [record for record in records if not split_assign(record)]
    forced = [record for record in records if split_assign(record)]

    def raise_if_empty(name: str, count: int, ratio: float) -> None:
        if ratio > 0 and count < 1:
            raise ValueError(f"at least one study is required for the {name} split "
                             f"(pool has {len(pool)} records; adjust --ratios or --test-groups)")

    train_count = int(len(pool) * ratios[0])
    val_count = int(len(pool) * ratios[1])
    raise_if_empty("train", train_count, ratios[0])
    raise_if_empty("val", val_count, ratios[1])
    leftover = len(pool) - train_count - val_count
    raise_if_empty("test", leftover + len(forced), ratios[2])

    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    assignment: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for record in shuffled[:train_count]:
        assignment["train"].append(record["key"])
    for record in shuffled[train_count:train_count + val_count]:
        assignment["val"].append(record["key"])
    for record in shuffled[train_count + val_count:]:
        assignment["test"].append(record["key"])
    for record in forced:
        assignment["test"].append(record["key"])

    manifest: dict[str, Any] = {
        "seed": seed,
        "ratios": list(ratios),
        "groups": sorted(wanted),
        "test_groups": sorted(test),
        "records": records,
        "splits": assignment,
    }

    stats: dict[str, Any] = {
        "total": len(records),
        "pool": len(pool),
        "forced_test": len(forced),
        "by_split": {name: len(keys_) for name, keys_ in assignment.items()},
        "by_split_group": {
            name: dict(sorted(Counter(record["group"] for record in records
                                      if record["key"] in keys_).items()))
            for name, keys_ in assignment.items()
        },
        "missing_base": {
            subset: sum(1 for record in records
                        if subset not in record["outputs"] and f"{subset}.raw" not in record["outputs"])
            for subset in BASE_SUBSETS
        },
    }
    return manifest, stats


def validate_location(output_root: str | Path, out: str | Path) -> None:
    """The manifest must sit directly inside the output root (relative-path base)."""
    root = Path(output_root).resolve()
    destination = Path(out).resolve()
    if destination.parent != root:
        raise ValueError(
            f"--out must be inside --output-root so relative outputs paths resolve; "
            f"got out dir {destination.parent} != output root {root}")


def _report(stats: dict[str, Any]) -> None:
    print(f"selected {stats['total']} studies  "
          f"(random pool {stats['pool']}, forced into test {stats['forced_test']})")
    for name in ("train", "val", "test"):
        detail = ", ".join(f"{group}:{count}" for group, count in stats["by_split_group"][name].items())
        print(f"  {name:<5} {stats['by_split'][name]:>4}  ({detail})")
    missing = {subset: count for subset, count in stats["missing_base"].items() if count}
    if missing:
        print(f"  warning: records missing base volumes for N2N mappings: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a study-level train/val/test manifest from reconstruction-index.json.")
    parser.add_argument("--index", type=Path, required=True, help="reconstruction-index.json")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="directory the relative outputs paths resolve against (== --out dir)")
    parser.add_argument("--groups", required=True, help="comma-separated groups to include, e.g. Mode2_1609,Mode3_1609")
    parser.add_argument("--out", type=Path, required=True, help="manifest output path (same dir as --output-root)")
    parser.add_argument("--test-groups", default="", help="comma-separated groups forced entirely into test")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                        help="train/val/test ratios applied to the non-test-groups pool")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate_location(args.output_root, args.out)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    test_groups = [group.strip() for group in args.test_groups.split(",") if group.strip()]
    manifest, stats = build_manifest(index, groups=groups, seed=args.seed,
                                     ratios=tuple(args.ratios), test_groups=test_groups)
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(stats)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
