"""Read-only pre-flight for the AIReduceNoise3 r1 blend-at-round-1 round.

Verifies that the Mode2_1609 train/val manifest is ready to host the r1 target:
  1. record count / split sizes (train/val/test),
  2. every record carries the six volumes the four-domain joint training needs,
  3. the source files exist on disk and their size matches shape x 4 bytes,
  4. whether `full.dn_3d_128b_r0.raw` / `full.dn_3d_128b_r1.raw` already exist,
  5. free space on the output filesystem (two new suffixes will be written).

Exit code 0 = ready, 1 = blockers found.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

MANIFEST = Path("/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json")
REQUIRED = ("even", "odd", "full", "full.bhc", "full.marfi", "full.marfi.bhc")
R0_SUFFIX = "dn_3d_128b_r0"
R1_SUFFIX = "dn_3d_128b_r1"


def resolve(outputs: dict, logical: str):
    for key in (logical, f"{logical}.raw"):
        if key in outputs:
            return outputs[key]
    return None


def main() -> int:
    problems: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = MANIFEST.parent
    records = {str(r["key"]): r for r in manifest["records"]}
    splits = {name: [str(k) for k in keys] for name, keys in manifest["splits"].items()}

    print(f"manifest : {MANIFEST}")
    print(f"records  : {len(records)}")
    for name, keys in splits.items():
        print(f"split {name:<6}: {len(keys)}")
    if len(splits.get("test", [])) != 0:
        problems.append(f"train/val manifest must have an empty test split, got {len(splits.get('test', []))}")
    expected = {"train": 180, "val": 20}
    for name, n in expected.items():
        if len(splits.get(name, [])) != n:
            problems.append(f"split {name} has {len(splits.get(name, []))} keys, expected {n}")

    groups = {}
    n_missing_vol = 0
    n_missing_file = 0
    n_bad_size = 0
    n_r0 = 0
    n_r1 = 0
    total_bytes = 0
    for key in splits.get("train", []) + splits.get("val", []):
        record = records[key]
        groups[str(record.get("group"))] = groups.get(str(record.get("group")), 0) + 1
        outputs = record.get("outputs") or {}
        shape = tuple(int(v) for v in record["shape"])
        expected_size = shape[0] * shape[1] * shape[2] * 4
        for logical in REQUIRED:
            rel = resolve(outputs, logical)
            if rel is None:
                n_missing_vol += 1
                problems.append(f"{key}: outputs lacks {logical}")
                continue
            path = base / rel
            if not path.exists():
                n_missing_file += 1
                problems.append(f"{key}: missing on disk {path}")
                continue
            if path.stat().st_size != expected_size:
                n_bad_size += 1
                problems.append(
                    f"{key}: {logical} size {path.stat().st_size} != {expected_size} (shape {shape})"
                )
            total_bytes += expected_size
        if resolve(outputs, f"full.{R0_SUFFIX}") and (base / resolve(outputs, f"full.{R0_SUFFIX}")).exists():
            n_r0 += 1
        if resolve(outputs, f"full.{R1_SUFFIX}") and (base / resolve(outputs, f"full.{R1_SUFFIX}")).exists():
            n_r1 += 1

    print(f"groups   : {len(groups)} devices -> {dict(sorted(groups.items()))}")
    n_studies = len(splits["train"]) + len(splits["val"])
    print(f"existing full.{R0_SUFFIX}.raw : {n_r0}/{n_studies}")
    print(f"existing full.{R1_SUFFIX}.raw : {n_r1}")
    print(f"source volume bytes (6 vols x 200) : {total_bytes / 2**30:.1f} GiB "
          f"({total_bytes / (200 * 6) / 1e6:.1f} MB/volume)")
    print(f"two new suffixes will add ~{2 * 200 * (total_bytes / (200 * 6)) / 2**30:.1f} GiB")
    for logical, count in (("missing outputs key", n_missing_vol),
                           ("missing on disk", n_missing_file),
                           ("size mismatch", n_bad_size)):
        if count:
            problems.append(f"{count} x {logical}")

    usage = shutil.disk_usage(base)
    print(f"disk {base}: total={usage.total / 2**40:.2f} TiB used={usage.used / 2**40:.2f} TiB "
          f"free={usage.free / 2**40:.2f} TiB")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for line in problems[:40]:
            print(f"  - {line}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("\nREADY: manifest + six volumes verified, no blockers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
