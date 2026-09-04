"""Prepare denoised-full (``full.dn``) supervision targets for plan C / E4.

For every study in the train (and val) splits this script:
  1. locates ``full.raw`` through the manifest record's outputs,
  2. runs the *teacher* model (the E2 half->full checkpoint) on the full-view
     volume — a further-denoised full reconstruction — and writes
     ``full.dn.raw`` (headerless float32) *next to* the original ``full.raw``,
  3. registers ``full.dn.raw`` in the manifest record's ``outputs`` so that
     N2NDataset can supervise ``even/odd -> full.dn``,
  4. writes a per-study diagnostic ``full.dn vs full`` metric (how much the
     teacher changed the already-low-noise full volume; a near-zero delta means
     the plan degenerates to plain half->full).

Existing ``full.dn.raw`` files are skipped (resume-safe); the manifest patch is
idempotent and applied once after the loop.  Extra ``outputs`` keys do not
affect the E1/E2/E3 datasets (they only request even/odd/full + residuals).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import evaluate_volume, infer_volume, write_raw_volume

DN_TARGET_SUFFIX = "dn"  # volume file: full.<suffix>.raw ; manifest key: full.<suffix>.raw


def _resolve_output(record: dict, logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (logical, f"{logical}.raw"):
        if key in outputs:
            return outputs[key]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full.dn teacher-denoised targets for train/val.")
    parser.add_argument("--config", type=Path, default=Path("configs/1609_fullview_dn.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.resolve().parent
    data_range = float(config["data"]["metric_data_range"])

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cuda":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    model = UNet2D(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    records = {str(record["key"]): record for record in manifest["records"]}
    splits = ["train", "val"]
    keys = [key for split in splits for key in manifest["splits"][split]]
    print(f"[prep] teacher={args.checkpoint} splits={splits} studies={len(keys)} "
          f"device={device} target_suffix={DN_TARGET_SUFFIX}", flush=True)

    overview_rows: list[dict] = []
    n_existing = n_new = n_skip_missing = 0
    for key in keys:
        record = records[key]
        relative = _resolve_output(record, "full")
        if relative is None:
            n_skip_missing += 1
            print(f"[skip] {key}: no full.raw in outputs", flush=True)
            continue
        full_path = base / relative
        if not full_path.exists():
            n_skip_missing += 1
            print(f"[skip] {key}: full missing {full_path}", flush=True)
            continue

        dst = full_path.with_name(f"{full_path.stem}.{DN_TARGET_SUFFIX}.raw")
        if dst.exists():
            n_existing += 1
            print(f"[exists] {key}: {dst.name} already there", flush=True)
            continue
        shape = tuple(int(dim) for dim in record["shape"])
        dtype = np.dtype(record.get("dtype") or "<f4")
        full = np.memmap(full_path, dtype=dtype, mode="r", shape=shape)
        denoised = infer_volume(model, full, device, args.batch_size)
        write_raw_volume(dst, denoised)
        metrics = evaluate_volume(denoised, np.asarray(full), data_range)
        entry = {
            "key": key,
            "study_uid": record.get("study_uid"),
            "group": record.get("group"),
            "case_id": record.get("case_id"),
            "n_slices": int(shape[0]),
            "relpath": str(dst.relative_to(base).as_posix()),
            "metrics": metrics,
        }
        overview_rows.append(entry)
        n_new += 1
        print(f"[done] {key} ({record.get('group')}): "
              + " ".join(f"{m}={metrics[m]:.4f}" for m in ("mae", "rmse", "psnr", "ssim")), flush=True)

    # register the new target volume in the manifest (idempotent; once)
    outputs_key = f"full.{DN_TARGET_SUFFIX}.raw"
    patched = 0
    for entry in overview_rows:
        record = records[entry["key"]]
        outputs = record.setdefault("outputs", {})
        if outputs.get(outputs_key) != entry["relpath"]:
            outputs[outputs_key] = entry["relpath"]
            patched += 1
    if patched:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"[manifest] patched {patched} records with {outputs_key}", flush=True)
    else:
        print(f"[manifest] no change needed for {outputs_key}", flush=True)

    # aggregate diagnostics: how much did the teacher change full?
    if overview_rows:
        agg: dict[str, dict] = {}
        for metric in ("mae", "rmse", "psnr", "ssim"):
            values = [float(row["metrics"][metric]) for row in overview_rows]
            agg[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                           "n": len(values)}
        overview_path = base / "prepare_full_dn_overview.json"
        overview_path.write_text(json.dumps({
            "splits": splits,
            "n_new": n_new, "n_existing": n_existing, "n_skip_missing": n_skip_missing,
            "teacher_checkpoint": str(args.checkpoint),
            "data_range": data_range,
            "note": "full.dn vs full: high PSNR / tiny delta => targets ~= full "
                    "(plan degenerates toward plain half->full); "
                    "larger delta => teacher meaningfully cleans full",
            **agg}, indent=2), encoding="utf-8")
        print(f"\noverview written: {overview_path}")
        print(f"summary: new={n_new} existing={n_existing} skipped(missing full)={n_skip_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
