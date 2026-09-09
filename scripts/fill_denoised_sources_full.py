"""Populate a denoised eval output dir with source volumes + full-view denoising.

For every study in the requested split this script:
  1. copies the noisy half-view source volumes (``even.raw`` / ``odd.raw``)
     from the data root into each case dir, next to the denoised ones
  2. runs the model on the *full-view* volume as well (domain-shift
     inference — the E1 model was trained even<->odd) and stores
     ``full_denoised.raw``
  3. writes ``full_metrics.json`` per case (full_denoised vs full, purely
     diagnostic: how much the model changes the already-lower-noise full
     reconstruction)

Consumes the same study-level manifest as ``test_raw.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import evaluate_volume, infer_volume, write_raw_volume


def _resolve_output(record: dict[str, Any], logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (f"{logical}.raw", logical):
        if key in outputs:
            return outputs[key]
    return None


def _case_dir_name(record: dict[str, Any]) -> str:
    group = record.get("group") or "unknown_group"
    study_uid = record.get("study_uid") or record.get("key") or "unknown"
    return f"{group}__{study_uid}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy source half volumes + denoise full-view volumes.")
    parser.add_argument("--config", type=Path, default=Path("configs/1609.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    data_range = float(config["data"]["metric_data_range"])

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    model = UNet2D(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    records = {str(record["key"]): record for record in manifest["records"]}
    split_keys = list(manifest["splits"][args.split])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fill] split={args.split} studies={len(split_keys)} checkpoint={args.checkpoint} "
          f"device={device}", flush=True)

    overview_rows: list[dict[str, Any]] = []
    for key in split_keys:
        record = records[key]
        group = record.get("group") or "?"
        case_dir = output_dir / _case_dir_name(record)
        case_dir.mkdir(parents=True, exist_ok=True)

        # 1) copy source half volumes next to their denoised counterparts
        for kind in ("even", "odd"):
            relative = _resolve_output(record, kind)
            if relative is None:
                print(f"[skip-copy] {key} {kind}: no source path in outputs", flush=True)
                continue
            src = base / relative
            if not src.exists():
                print(f"[skip-copy] {key} {kind}: source missing {src}", flush=True)
                continue
            dst = case_dir / f"{kind}.raw"
            if dst.exists():
                print(f"[exists] {kind}.raw already there, skip copy", flush=True)
                continue
            shutil.copyfile(src, dst)  # streamed, does not load 0.55 GiB into RAM
            print(f"[copied] {kind}.raw <- {relative}", flush=True)

        # 2) denoise the full-view volume
        relative = _resolve_output(record, "full")
        if relative is None:
            print(f"[skip-full] {key}: no full path in outputs", flush=True)
            continue
        full_path = base / relative
        if not full_path.exists():
            print(f"[skip-full] {key}: full missing {full_path}", flush=True)
            continue
        shape = tuple(int(dim) for dim in record["shape"])
        full = np.memmap(full_path, dtype=np.dtype(record.get("dtype") or "<f4"), mode="r", shape=shape)
        denoised = infer_volume(model, full, device, args.batch_size)
        write_raw_volume(case_dir / "full_denoised.raw", denoised)
        metrics = evaluate_volume(denoised, np.asarray(full), data_range)
        entry = {
            "key": key,
            "study_uid": record.get("study_uid"),
            "group": group,
            "source_kind": "full",
            "reference": "full",
            "n_slices": int(shape[0]),
            "metrics": metrics,
        }
        (case_dir / "full_metrics.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        overview_rows.append(entry)
        print(f"[done] {key} ({group}) full_denoised: "
              + " ".join(f"{m}={metrics[m]:.4f}" for m in ("mae", "rmse", "psnr", "ssim")), flush=True)

    # aggregate diagnostics (full_denoised vs full)
    if overview_rows:
        agg: dict[str, dict[str, float]] = {}
        for metric in ("mae", "rmse", "psnr", "ssim"):
            values = [float(row["metrics"][metric]) for row in overview_rows]
            agg[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                           "n": len(values)}
        (output_dir / "full_denoised_overview.json").write_text(
            json.dumps({"split": args.split, "data_range": data_range,
                        "n_studies": len(overview_rows),
                        "note": "full_denoised vs full (domain-shift inference); "
                                "high PSNR = model barely changes full", **agg},
                       indent=2), encoding="utf-8")
        print(f"\noverview written to: {output_dir / 'full_denoised_overview.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
