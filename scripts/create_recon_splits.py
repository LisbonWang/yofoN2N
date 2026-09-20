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
- ``--exclude-groups`` is an explicit safety filter applied after ``--groups``;
  excluded records never enter ``records`` or any train/val/test split.  The blocked
  groups ``Mode0_1615`` and ``Mode1_1615`` are always excluded even if callers pass
  an empty override.

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

#: Training safety policy: 1615 reconstructions are excluded until the Mode1 dual-source
#: split satisfies XDay's single-source projection requirements and passes image-level QC.
DEFAULT_EXCLUDE_GROUPS = ("Mode0_1615", "Mode1_1615")


def make_key(study_uid: str, group: str) -> str:
    """Composite key unique across groups (and across records by construction)."""
    return f"{study_uid}::{group}"


def build_manifest(index: dict[str, Any], *, groups: Iterable[str],
                   seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                   test_groups: Iterable[str] = (),
                   exclude_groups: Iterable[str] = DEFAULT_EXCLUDE_GROUPS,
                   case_prefix: str = "", per_case: int | None = None,
                   target_total: int | None = None,
                   require_outputs: Iterable[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(manifest, stats)`` for the selected groups.

    ``manifest`` follows the same top-level layout as
    :func:`cbct_n2n.data.manifest.create_case_splits`:
    ``{"seed", "ratios", "records": [...], "splits": {"train": [...], "val": [...], "test": [...]}}``
    plus provenance fields.  ``stats`` carries per-split/per-group counters for
    the console report.
    """
    requested = set(groups)
    excluded = (set(exclude_groups) | set(DEFAULT_EXCLUDE_GROUPS)) & requested
    wanted = requested - excluded
    test = set(test_groups)
    available = {item["group"] for item in index.get("records", [])}
    unknown_groups = requested - available
    if unknown_groups:
        raise ValueError(f"groups not present in the index: {sorted(unknown_groups)}")
    if not wanted:
        raise ValueError("all requested groups were excluded")
    for group in test:
        if group not in wanted:
            raise ValueError(f"--test-groups {group!r} is not among the included groups")

    excluded_counts = Counter(
        item["group"] for item in index.get("records", [])
        if item.get("group") in excluded and item.get("status") in {None, "success"}
    )
    records: list[dict[str, Any]] = []
    for item in index["records"]:
        if item.get("group") not in wanted:
            continue
        if item.get("status") not in {None, "success"}:
            continue
        if case_prefix and not str(item.get("case_id", "")).startswith(case_prefix):
            continue
        outputs = item.get("outputs", {})
        if any(name not in outputs for name in require_outputs):
            continue
        record = {field: item[field] for field in COPIED_FIELDS if field in item}
        record["key"] = make_key(record["study_uid"], record["group"])
        records.append(record)
    if not records:
        raise ValueError(f"no records selected for groups {sorted(wanted)}")

    available_by_case = Counter(record["case_id"] for record in records)
    if per_case is not None or target_total is not None:
        if per_case is None or per_case < 1:
            raise ValueError("per_case must be a positive integer when stratified sampling is enabled")
        if target_total is None or target_total < 1:
            raise ValueError("target_total must be a positive integer when stratified sampling is enabled")
        if target_total > len(records):
            raise ValueError(f"target_total={target_total} exceeds {len(records)} available records")

        by_case: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_case.setdefault(record["case_id"], []).append(record)
        rng = random.Random(seed)
        selected: dict[str, list[dict[str, Any]]] = {}
        remaining: dict[str, list[dict[str, Any]]] = {}
        for case_id in sorted(by_case):
            candidates = sorted(by_case[case_id], key=lambda record: record["key"])
            rng.shuffle(candidates)
            take = min(per_case, len(candidates))
            selected[case_id] = candidates[:take]
            remaining[case_id] = candidates[take:]

        selected_count = sum(len(items) for items in selected.values())
        if selected_count > target_total:
            raise ValueError(
                f"baseline per-case sampling selects {selected_count}, exceeding target_total={target_total}")
        extra_needed = target_total - selected_count
        while extra_needed:
            eligible = [case_id for case_id, items in remaining.items() if items]
            if not eligible:
                raise ValueError(f"cannot fill target_total={target_total}; only {len(records)} available")
            case_id = min(eligible, key=lambda key: (len(selected[key]), key))
            selected[case_id].append(remaining[case_id].pop(0))
            extra_needed -= 1
        records = [record for case_id in sorted(selected) for record in selected[case_id]]

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
        "exclude_groups": sorted(excluded),
        "case_prefix": case_prefix,
        "per_case": per_case,
        "target_total": target_total,
        "require_outputs": list(require_outputs),
        "test_groups": sorted(test),
        "records": records,
        "splits": assignment,
    }

    stats: dict[str, Any] = {
        "total": len(records),
        "pool": len(pool),
        "forced_test": len(forced),
        "excluded_by_group": dict(sorted(excluded_counts.items())),
        "available_by_case": dict(sorted(available_by_case.items())),
        "selected_by_case": dict(sorted(Counter(record["case_id"] for record in records).items())),
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
    if stats["excluded_by_group"]:
        print(f"  excluded groups: {stats['excluded_by_group']}")
    if stats["selected_by_case"]:
        print(f"  selected by case: {stats['selected_by_case']}")
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
    parser.add_argument("--exclude-groups", default=",".join(DEFAULT_EXCLUDE_GROUPS),
                        help="comma-separated groups removed from the manifest even if listed in --groups "
                             "(default: Mode0_1615,Mode1_1615)")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                        help="train/val/test ratios applied to the non-test-groups pool")
    parser.add_argument("--case-prefix", default="", help="only include case_id values with this prefix")
    parser.add_argument("--per-case", type=int, help="baseline random sample count per case/device")
    parser.add_argument("--target-total", type=int,
                        help="final distinct-study count; remaining slots are assigned to the least represented eligible cases")
    parser.add_argument("--require-outputs", default="",
                        help="comma-separated output filenames required before a record is eligible")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate_location(args.output_root, args.out)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    test_groups = [group.strip() for group in args.test_groups.split(",") if group.strip()]
    exclude_groups = [group.strip() for group in args.exclude_groups.split(",") if group.strip()]
    require_outputs = [name.strip() for name in args.require_outputs.split(",") if name.strip()]
    manifest, stats = build_manifest(index, groups=groups, seed=args.seed,
                                     ratios=tuple(args.ratios), test_groups=test_groups,
                                     exclude_groups=exclude_groups, case_prefix=args.case_prefix,
                                     per_case=args.per_case, target_total=args.target_total,
                                     require_outputs=require_outputs)
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(stats)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
