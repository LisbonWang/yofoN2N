"""Prepare teacher-inferred full-view (``full.<suffix>``) supervision targets.

Use a converged teacher checkpoint to infer from a full-view volume or a
previously generated ``full.*`` volume. This supports chained distillation
such as ``N2N(R0(full))``.

For every study in the requested splits this script:
  1. locates the configured ``full`` / ``full.<input-suffix>`` volume through
     the manifest record's outputs,
  2. runs the teacher model on that volume (further-denoised full
     reconstruction) and writes ``full.<target-suffix>.raw`` (headerless
     float32) in the study output directory,
  3. registers ``full.<target-suffix>.raw`` in the manifest record's
     ``outputs`` so :class:`cbct_n2n.data.n2n_dataset.N2NDataset` can use it
     as a target,
  4. writes a per-study diagnostic of the *noise level* of output vs input.
     This -- NOT PSNR -- is a useful signal for whether the teacher target is
     actually lower-noise: because the input is noisy,
     ``half -> full`` already contains a Noise2Noise component, so the
     teacher objective is the transformed volume, not a higher PSNR against
     the noisy input. A noise-std drop is only a high-frequency signal and
     must be considered alongside visual texture checks.

Existing ``full.<suffix>.raw`` files are skipped (resume-safe); the manifest
patch is idempotent and applied once after the loop.  The overview JSON is
merged across runs as well (one row per study), so a resumed generation still
reports full coverage to the round gate.  Extra ``outputs`` keys
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


def _full_logical_name(suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return "full"
    if "/" in suffix or "\\" in suffix or suffix in {".", ".."}:
        raise ValueError(f"invalid volume suffix {suffix!r}")
    return f"full.{suffix}"


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


def merge_overview_rows(previous: object, new_rows: list[dict]) -> list[dict]:
    """Merge diagnostic rows across runs, keyed by study key (newest wins).

    A resumed run only *creates* the missing targets, so aggregating over that
    run alone would shrink the coverage the round gate sees (200 -> however
    many were left) and cause a false gate failure.  Rows are therefore merged
    with the previous overview of the same suffix.
    """
    merged: dict[str, dict] = {}
    if isinstance(previous, dict):
        for row in previous.get("rows", []) or []:
            if isinstance(row, dict) and row.get("key"):
                merged[str(row["key"])] = row
    for row in new_rows:
        if isinstance(row, dict) and row.get("key"):
            merged[str(row["key"])] = row
    return sorted(merged.values(), key=lambda row: str(row["key"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate full.<suffix> targets from a registered full volume using a teacher model."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Task YAML config matching the teacher checkpoint.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Teacher checkpoint used to infer the requested input volume.")
    parser.add_argument("--input-suffix", default="",
                        help="Input suffix; empty reads full.raw, while 'by_r0' reads full.by_r0.raw.")
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
    if not args.target_suffix.strip():
        parser.error("--target-suffix must not be empty")
    input_logical = _full_logical_name(args.input_suffix)
    target_logical = _full_logical_name(args.target_suffix)
    if input_logical == target_logical:
        parser.error("input and target suffixes must differ")
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
          f"device={device} input={input_logical} target={target_logical}", flush=True)

    records = {str(record["key"]): record for record in manifest["records"]}
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits:
        print("[warn] 'test' split should NOT be teacher-denoised (leakage). Removing it.", flush=True)
        splits = [s for s in splits if s != "test"]
    keys = [key for split in splits for key in manifest["splits"][split]]
    print(f"[prep] splits={splits} studies={len(keys)}", flush=True)

    overview_rows: list[dict] = []
    output_paths: dict[str, str] = {}
    n_existing = n_new = n_skip_missing = 0
    for key in keys:
        record = records[key]
        relative = _resolve_output(record, input_logical)
        if relative is None:
            n_skip_missing += 1
            print(f"[skip] {key}: no {input_logical}.raw in outputs", flush=True)
            continue
        input_path = base / relative
        if not input_path.exists():
            n_skip_missing += 1
            print(f"[skip] {key}: input missing {input_path}", flush=True)
            continue

        dst = input_path.parent / f"{target_logical}.raw"
        output_paths[key] = str(dst.relative_to(base).as_posix())
        if dst.exists():
            n_existing += 1
            print(f"[exists] {key}: {dst.name} already there", flush=True)
            continue
        shape = tuple(int(dim) for dim in record["shape"])
        dtype = np.dtype(record.get("dtype") or "<f4")
        source_volume = np.memmap(input_path, dtype=dtype, mode="r", shape=shape)
        denoised = _denoise_volume(model, arch, source_volume, device, args.batch_size, voxel_depth,
                                   tiled=args.tiled, patch_hw=args.patch_hw,
                                   overlap_d=args.overlap_d, overlap_hw=args.overlap_hw)
        write_raw_volume(dst, denoised)
        source_arr = np.asarray(source_volume)
        # Fast noise-level diagnostic only (Immerkaer Laplacian kernel, ~1s per volume).
        # The median-filter estimator (estimate_noise_std) is ~20s/volume and must
        # NOT run inside the target-generation loop; PSNR/SSIM vs full are also
        # meaningless here (full is noisy) and are omitted.
        noise = {
            "input_immerkaer": estimate_noise_std_immerkaer(source_arr),
            "target_immerkaer": estimate_noise_std_immerkaer(denoised),
        }
        if input_logical == "full":
            noise["full_immerkaer"] = noise["input_immerkaer"]
            noise["dn_immerkaer"] = noise["target_immerkaer"]
        entry = {
            "key": key,
            "study_uid": record.get("study_uid"),
            "group": record.get("group"),
            "case_id": record.get("case_id"),
            "n_slices": int(shape[0]),
            "relpath": str(dst.relative_to(base).as_posix()),
            "input_logical": input_logical,
            "target_logical": target_logical,
            "noise": noise,
        }
        overview_rows.append(entry)
        n_new += 1
        print(f"[done] {key} ({record.get('group')}): "
              f"immerkaer {noise['input_immerkaer']:.5f}->{noise['target_immerkaer']:.5f}", flush=True)

    # register the new target volume in the manifest (idempotent; once)
    outputs_key = f"{target_logical}.raw"
    patched = 0
    for key, relative in output_paths.items():
        record = records[key]
        outputs = record.setdefault("outputs", {})
        if outputs.get(outputs_key) != relative:
            outputs[outputs_key] = relative
            patched += 1
    if patched:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"[manifest] patched {patched} records with {outputs_key}", flush=True)
    else:
        print(f"[manifest] no change needed for {outputs_key}", flush=True)

    # aggregate diagnostics
    #
    # Merge with a previous overview of the same suffix: a resumed run only
    # *creates* the missing targets, so aggregating over this run alone would
    # shrink the coverage the gate sees (200 -> however many were left) and
    # produce a false gate failure.  Rows are keyed by study, newest wins.
    overview_path = base / f"prepare_full_{args.target_suffix}_overview.json"
    previous_overview: object = None
    if overview_path.exists():
        try:
            previous_overview = json.loads(overview_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_overview = None  # unreadable/legacy overview -> rebuild from this run alone
    all_rows = merge_overview_rows(previous_overview, overview_rows)

    if all_rows:
        agg: dict[str, dict] = {}

        # Noise-level comparison only (Immerkaer; the median-filter estimator
        # is ~20s/volume and is deliberately not run in the generation loop).
        noise_rows = [row for row in all_rows if row.get("noise")]
        for est in ("immerkaer",):
            input_vals = np.asarray([
                row["noise"].get(f"input_{est}", row["noise"].get(f"full_{est}"))
                for row in noise_rows
            ], dtype=np.float64)
            target_vals = np.asarray([
                row["noise"].get(f"target_{est}", row["noise"].get(f"dn_{est}"))
                for row in noise_rows
            ], dtype=np.float64)
            delta = input_vals - target_vals
            denom = np.where(input_vals > 0, input_vals, np.nan)
            rel = delta / denom  # fractional reduction
            agg[f"noise_{est}"] = {
                "input_mean": float(np.nanmean(input_vals)),
                "target_mean": float(np.nanmean(target_vals)),
                "delta_mean": float(np.nanmean(delta)),       # > 0 => cleaner target
                "delta_median": float(np.nanmedian(delta)),
                "relative_reduction_mean": float(np.nanmean(rel)),
                "n": int(len(noise_rows)),
            }
            if input_logical == "full":
                agg[f"noise_{est}"]["full_mean"] = agg[f"noise_{est}"]["input_mean"]
                agg[f"noise_{est}"]["dn_mean"] = agg[f"noise_{est}"]["target_mean"]

        overview_path.write_text(json.dumps({
            "splits": splits,
            "n_new": n_new, "n_existing": n_existing, "n_skip_missing": n_skip_missing,
            "n_rows": len(all_rows),
            "teacher_checkpoint": str(args.checkpoint),
            "input_logical": input_logical,
            "target_logical": target_logical,
            "target_suffix": args.target_suffix,
            "note": (
                "Diagnostic = Immerkaer Laplacian-kernel noise level, not PSNR. "
                "noise_immerkaer delta = input_mean - target_mean; positive values indicate "
                "lower high-frequency energy in the teacher output, but do not distinguish "
                "noise removal from texture loss. The median-filter estimator "
                "is too slow for the training loop and is intentionally not computed. "
                "'rows' keeps one diagnostic row per study and is merged across runs, so the "
                "gate's coverage stays complete even after a resumed/partial generation."
            ),
            **agg,
            "rows": all_rows}, indent=2), encoding="utf-8")
        print(f"\noverview written: {overview_path}")
        for est in ("immerkaer",):
            a = agg[f"noise_{est}"]
            print(f"  noise_{est}: input={a['input_mean']:.5f} -> target={a['target_mean']:.5f} "
                  f"delta={a['delta_mean']:.5f} ({a['relative_reduction_mean']*100:.2f}% reduction)")
        print(f"summary: new={n_new} existing={n_existing} skipped(missing full)={n_skip_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
