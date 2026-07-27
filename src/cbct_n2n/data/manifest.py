from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def create_case_splits(records: Iterable[dict[str, Any]], ratios: tuple[float, float, float], seed: int = 42) -> dict[str, Any]:
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must be three non-negative values summing to one")
    items = list(records)
    counts = [int(len(items) * ratio) for ratio in ratios[:2]]
    for ratio, count, name in zip(ratios, counts, ("train", "validation")):
        if ratio > 0 and count < 1:
            raise ValueError(f"at least one case is required for the positive-ratio {name} split")
    if ratios[2] > 0 and len(items) - sum(counts) < 1:
        raise ValueError("at least one case is required for the positive-ratio test split")
    ids = [str(item["case_id"]) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("case_id values must be unique")
    random.Random(seed).shuffle(ids)
    train_count, val_count = counts
    return {"seed": seed, "ratios": list(ratios), "records": items, "splits": {
        "train": ids[:train_count], "val": ids[train_count:train_count + val_count], "test": ids[train_count + val_count:]
    }}


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
