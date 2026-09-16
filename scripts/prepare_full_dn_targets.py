"""Prepare teacher-denoised full-view (``full.<suffix>``) supervision targets.

Step 1 of the **recursive training scheme** (half -> full_denoised): use an
already-converged *teacher* model (the round ``k-1`` half->full checkpoint) to
denoise the ``full`` reconstruction of every train/val study, then supervise
the next round on ``even/odd -> full.<suffix>``.

For every study in the requested splits this script:
  1. locates ``full.raw`` through the manifest record's outputs,
  2. runs the teacher model on the full-view volume (further-denoised full
     reconstruction) and writes ``full.<suffix>.raw`` (headerless float32)
     *next to* the original ``full.raw``,
  3. registers ``full.<suffix>.raw`` in the manifest record's ``outputs`` so
     that :class:`cbct_n2n.data.n2n_dataset.N2NDataset` can supervise
     ``even/odd -> full.<suffix>`` (the target whitelist already accepts any
     ``full.*`` variant),
  4. writes a per-study diagnostic of the *noise level* of ``full.<suffix>``
     vs ``full``.  This -- NOT PSNR -- is the correct signal for whether the
     recursive target is actually cleaner: because ``full`` is itself noisy,
     ``half -> full`` already contains a Noise2Noise component, so the
     recursive objective is a lower-noise image, not a higher PSNR against
     the noisy ``full``.  A noise-std drop (``full`` > ``full.<suffix>``)
     means the teacher produced a meaningfully cleaner target and the round
     is worth training; a ~0 drop means the teacher did not clean ``full``
     and the round degenerates.

Existing ``full.<suffix>.raw`` files are skipped (resume-safe); the manifest
patch is idempotent and applied once after the loop.  Extra ``outputs`` keys
do not affect plain half->full / half<->half datasets (they only request
even/odd/full + residuals).

The teacher architecture is independent of the student: ``--arch`` only
selects how the *teacher* runs inference (unet2d / unet2d_ds / unet3d).  The
student is whatever config you then train against these targets.

Rounds use distinct suffixes (``dn`` / ``dn1`` / ``dn2`` ...) so consecutive
recursive rounds never collide on the same ``full.<suffix>.raw`` file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

import train_voxel  # noqa: E402  (reuse build_model: unet3d / unet2d / unet2d_ds)

from cbct_n2n.config import load_config
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import (
    infer_volume,
    infer_volume_3d,
    infer_volume_3d_tiled,
    write_raw_volume,
)
from cbct_n2n.training.metrics import estimate_noise_std_immerkaer

DEFAULT_TARGET_SUFFIX = "dn"  # volume file: full.<suffix>.raw ; manifest key: full.<suffix>.raw


def _resolve_output(record: dict, logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (logical, f"{logical}.raw"):
        if key in outputs:
            return outputs[key]
    return None


def _denoise_volume(model, arch: str, volume: np.ndarray, device: torch.device,
                    batch_size: int, voxel_depth: int, tiled: bool = False,
                    patch_hw: int = 256, overlap_d: int = 16,
                    overlap_hw: int = 16) -> np.ndarray:
    """Run the teacher on one (z, y, x) volume using the arch-appropriate path."""
    if arch == "unet3d":
        if tiled:
            return infer_volume_3d_tiled(model, volume, voxel_depth, patch_hw,
                                         overlap_d, overlap_hw, device, batch_size)
        return infer_volume_3d(model, volume, voxel_depth, device, batch_size)
    return infer_volume(model, volume, device, batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate full.<suffix> teacher-denoised targets for the recursive scheme."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/1609v2_unet2d_ds_h2f.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Teacher checkpoint (the converged half->full model).")
    parser.add_argument("--arch", choices=("unet2d", "unet2d_ds", "unet3d"), default=None,
                        help="Teacher arch for inference (default: read config.model.arch).")
    parser.add_argument("--target-suffix", default=DEFAULT_TARGET_SUFFIX,
                        help="Suffix for the denoised full volume (default 'dn'); "
                             "use 'dn1'/'dn2'/... to index recursive rounds.")
    parser.add_argument("--splits", default="train,val",
                        help="Comma-separated splits to denoise (default 'train,val'). "
                             "Never include 'test' -- it must stay leakage-free.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiled", action="store_true",
                        help="Use overlapping-tile 3D inference for the teacher "
                             "(for 256x256x128-window models).")
    parser.add_argument("--patch-hw", type=int, default=256,
                        help="H/W tile size for --tiled (default 256).")
    parser.add_argument("--overlap-d", type=int, default=16,
                        help="D-axis tile overlap for --tiled (default 16 slices).")
    parser.add_argument("--overlap-hw", type=int, default=16,
                        help="H/W tile overlap for --tiled (default 16 pixels).")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.resolve().parent
    voxel_depth = int(config["data"].get("voxel_depth", 16))
    arch = args.arch or str(config.get("model", {}).get("arch", "unet2d"))

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    model = train_voxel.build_model(arch, config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[prep] teacher={args.checkpoint} arch={arch} params={n_params:,} "
          f"device={device} target_suffix={args.target_suffix}", flush=True)

    records = {str(record["key"]): record for record in manifest["records"]}
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits:
        print("[warn] 'test' split should NOT be teacher-denoised (leakage). Removing it.", flush=True)
        splits = [s for s in splits if s != "test"]
    keys = [key for split in splits for key in manifest["splits"][split]]
    print(f"[prep] splits={splits} studies={len(keys)}", flush=True)

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

        dst = full_path.with_name(f"{full_path.stem}.{args.target_suffix}.raw")
        if dst.exists():
            n_existing += 1
            print(f"[exists] {key}: {dst.name} already there", flush=True)
            continue
        shape = tuple(int(dim) for dim in record["shape"])
        dtype = np.dtype(record.get("dtype") or "<f4")
        full = np.memmap(full_path, dtype=dtype, mode="r", shape=shape)
        denoised = _denoise_volume(model, arch, full, device, args.batch_size, voxel_depth,
                                   tiled=args.tiled, patch_hw=args.patch_hw,
                                   overlap_d=args.overlap_d, overlap_hw=args.overlap_hw)
        write_raw_volume(dst, denoised)
        full_arr = np.asarray(full)
        # Fast noise-level gate only (Immerkaer Laplacian kernel, ~1s per volume).
        # The median-filter estimator (estimate_noise_std) is ~20s/volume and must
        # NOT run inside the target-generation loop; PSNR/SSIM vs full are also
        # meaningless here (full is noisy) and are omitted.
        noise = {
            "full_immerkaer": estimate_noise_std_immerkaer(full_arr),
            "dn_immerkaer": estimate_noise_std_immerkaer(denoised),
        }
        entry = {
            "key": key,
            "study_uid": record.get("study_uid"),
            "group": record.get("group"),
            "case_id": record.get("case_id"),
            "n_slices": int(shape[0]),
            "relpath": str(dst.relative_to(base).as_posix()),
            "noise": noise,
        }
        overview_rows.append(entry)
        n_new += 1
        print(f"[done] {key} ({record.get('group')}): "
              f"immerkaer {noise['full_immerkaer']:.5f}->{noise['dn_immerkaer']:.5f}", flush=True)

    # register the new target volume in the manifest (idempotent; once)
    outputs_key = f"full.{args.target_suffix}.raw"
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

    # aggregate diagnostics
    if overview_rows:
        agg: dict[str, dict] = {}

        # noise-level comparison -- the actual gate (Immerkaer only; the median-filter
        # estimator is ~20s/volume and is deliberately NOT run in the training loop).
        noise_rows = [row for row in overview_rows if row.get("noise")]
        for est in ("immerkaer",):
            full_vals = np.asarray([row["noise"][f"full_{est}"] for row in noise_rows], dtype=np.float64)
            dn_vals = np.asarray([row["noise"][f"dn_{est}"] for row in noise_rows], dtype=np.float64)
            delta = full_vals - dn_vals
            denom = np.where(full_vals > 0, full_vals, np.nan)
            rel = delta / denom  # fractional reduction
            agg[f"noise_{est}"] = {
                "full_mean": float(np.nanmean(full_vals)),
                "dn_mean": float(np.nanmean(dn_vals)),
                "delta_mean": float(np.nanmean(delta)),       # > 0 => cleaner target
                "delta_median": float(np.nanmedian(delta)),
                "relative_reduction_mean": float(np.nanmean(rel)),
                "n": int(len(noise_rows)),
            }

        overview_path = base / f"prepare_full_{args.target_suffix}_overview.json"
        overview_path.write_text(json.dumps({
            "splits": splits,
            "n_new": n_new, "n_existing": n_existing, "n_skip_missing": n_skip_missing,
            "teacher_checkpoint": str(args.checkpoint),
            "target_suffix": args.target_suffix,
            "note": (
                "Gate = NOISE LEVEL (Immerkaer Laplacian-kernel std), not PSNR: full is itself "
                "noisy, so half->full carries a Noise2Noise component and the recursive goal is "
                "a lower-noise image. noise_immerkaer 'delta' = noise(full) - noise(full.<suffix>); "
                "delta > 0 (relative_reduction_mean > 0) means the teacher produced a "
                "meaningfully cleaner target and the round is worth training. A ~0 delta means "
                "the teacher did not clean full (degenerate round). The median-filter estimator "
                "is too slow for the training loop and is intentionally not computed."
            ),
            **agg}, indent=2), encoding="utf-8")
        print(f"\noverview written: {overview_path}")
        for est in ("immerkaer",):
            a = agg[f"noise_{est}"]
            print(f"  noise_{est}: full={a['full_mean']:.5f} -> dn={a['dn_mean']:.5f} "
                  f"delta={a['delta_mean']:.5f} ({a['relative_reduction_mean']*100:.2f}% reduction)")
        print(f"summary: new={n_new} existing={n_existing} skipped(missing full)={n_skip_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
