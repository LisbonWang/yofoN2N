"""Evaluate a trained 3D UNet (UNet3D) on RAW volumes.

3D counterpart to ``scripts/test_raw.py``.  The volumetric model maps a 5D
slab ``[B, 1, D, H, W]`` to a same-D-length volume ``[B, 1, D, H, W]``.
Inference slides a non-overlapping ``voxel_depth``-slice window across z
(strided by ``voxel_depth``, mirroring the training-time window placement)
and keeps every denoised slice; H/W is padded to a multiple of 32.  Metrics
are computed against the full-view volume, identical to ``test_raw.py``.

Example (on the training server 192.168.36.102):
  python scripts/test_raw_3d.py \\
    --config configs/1609_unet3d.yaml \\
    --manifest /data/raid0/yyc/AIReduceNoise/splits_1609.json \\
    --checkpoint runs/1609_n2n_3d/best.pt \\
    --split test \\
    --output-dir outputs/eval_1609_n2n_3d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

import train_voxel  # noqa: E402  (reuse build_model)

from cbct_n2n.config import load_config
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import (
    evaluate_volume,
    infer_volume_3d,
    resolve_input_kinds,
    write_raw_volume,
)


def _resolve_output(record: dict[str, Any], logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (f"{logical}.raw", logical):
        if key in outputs:
            return outputs[key]
    return None


def _load_raw_volume(base: Path, record: dict[str, Any], logical: str) -> np.memmap | None:
    relative = _resolve_output(record, logical)
    if relative is None:
        return None
    path = base / relative
    if not path.exists():
        return None
    shape = tuple(int(dim) for dim in record["shape"])
    dtype = np.dtype(record.get("dtype") or "<f4")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _case_dir_name(record: dict[str, Any]) -> str:
    group = record.get("group") or "unknown_group"
    study_uid = record.get("study_uid") or record.get("key") or "unknown"
    return f"{group}__{study_uid}"


def _aggregate(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for metric in keys:
        values = [float(row[metric]) for row in rows]
        aggregated[metric] = {
            "mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3D UNet on RAW volumes from a study-level manifest."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/1609_unet3d.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--input-kind", choices=["even", "odd", "both", "full"], default="full")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_raw_3d"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-volumes", choices=["none", "raw", "npy"], default="none")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    data_range = float(config["data"]["metric_data_range"])
    voxel_depth = int(config["data"].get("voxel_depth", 16))
    arch = config.get("model", {}).get("arch", "unet3d")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    model = train_voxel.build_model(arch, config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[eval] arch={arch} params={n_params:,} voxel_depth={voxel_depth} "
          f"split={args.split} device={device}", flush=True)

    records = {str(record["key"]): record for record in manifest["records"]}
    split_keys = list(manifest["splits"][args.split])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] split={args.split} studies={len(split_keys)} checkpoint={args.checkpoint} "
          f"device={device} data_range={data_range}", flush=True)

    metric_keys = ("mae", "rmse", "psnr", "ssim")
    per_case: list[dict[str, Any]] = []

    for key in split_keys:
        record = records[key]
        group = record.get("group") or "?"
        full = _load_raw_volume(base, record, "full")
        if full is None:
            print(f"[skip] {key}: full-view reference volume missing", flush=True)
            continue
        for kind in resolve_input_kinds(args.input_kind):
            source = _load_raw_volume(base, record, kind)
            if source is None:
                print(f"[skip] {key} {kind}: source volume missing", flush=True)
                continue
            if source.shape != full.shape:
                print(f"[skip] {key} {kind}: shape mismatch source={source.shape} full={full.shape}", flush=True)
                continue
            prediction = infer_volume_3d(model, source, voxel_depth, device,
                                         args.batch_size)
            metrics = evaluate_volume(prediction, np.asarray(full), data_range)
            entry = {
                "key": key,
                "study_uid": record.get("study_uid"),
                "case_id": record.get("case_id"),
                "group": group,
                "source_kind": kind,
                "reference": "full",
                "n_slices": int(source.shape[0]),
                "metrics": metrics,
            }
            per_case.append(entry)

            case_dir = output_dir / _case_dir_name(record)
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / f"{kind}_metrics.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
            if args.save_volumes != "none":
                if args.save_volumes == "raw":
                    write_raw_volume(case_dir / f"{kind}_denoised.raw", prediction)
                else:
                    np.save(case_dir / f"{kind}_denoised.npy", prediction)
            print(f"[done] {key} ({group}) {kind}: "
                  + " ".join(f"{m}={metrics[m]:.4f}" for m in metric_keys), flush=True)

    if not per_case:
        print("[eval] no cases evaluated — nothing to summarize", flush=True)
        return 1

    by_kind: dict[str, list[dict[str, float]]] = {kind: [] for kind in resolve_input_kinds(args.input_kind)}
    by_kind_group: dict[tuple[str, str], list[dict[str, float]]] = {}
    for entry in per_case:
        kind, group = entry["source_kind"], entry["group"]
        by_kind[kind].append(entry["metrics"])
        by_kind_group.setdefault((kind, group), []).append(entry["metrics"])

    summary: dict[str, Any] = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "voxel_depth": voxel_depth,
        "data_range": data_range,
        "n_studies_evaluated": len({entry["key"] for entry in per_case}),
        "n_cases": len(per_case),
        "by_kind": {kind: _aggregate(rows, metric_keys) for kind, rows in by_kind.items() if rows},
        "by_kind_group": {
            f"{kind}|{group}": _aggregate(rows, metric_keys)
            for (kind, group), rows in sorted(by_kind_group.items())
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def print_table(rows: list[tuple[str, dict[str, dict[str, float]]]]) -> None:
        print(f"{'group':<22} {'n':>3} | " + " | ".join(f"{m:>10}" for m in metric_keys))
        print("-" * (22 + 4 + len(metric_keys) * 13))
        for label, aggregate in rows:
            cells = [f"{aggregate[m]['mean']:.4f}±{aggregate[m]['std']:.4f}" for m in metric_keys]
            print(f"{label:<22} {aggregate['mae']['n']:>3} | " + " | ".join(f"{c:>10}" for c in cells))

    print("\n=== overall (all groups) ===")
    print_table([(kind, summary["by_kind"][kind]) for kind in summary["by_kind"]])
    print("\n=== per group x input kind ===")
    print_table([(label, aggregate) for label, aggregate in summary["by_kind_group"].items()])
    print(f"\nsummary written to: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
