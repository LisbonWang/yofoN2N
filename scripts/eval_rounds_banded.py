"""Output-layer banded validation for the recursive rounds (the real r3 gate).

The r3 density-blend plan (``docs/plans/r3-density-blend.md``) has two gates:

1. **target layer** -- ``blend_dn_targets.py`` checks the blended *supervision
   target* is distributed correctly (low band ~ r1, high band ~ r2).  That is a
   sanity check only.
2. **output layer** -- *this script*.  Under half->full training the student
   output is one recursion step cleaner than its supervision target, so the
   blended target's banded distribution says nothing about the final effect.
   The real judgement is: run each round's model on the same ``full`` volume,
   key the density bands on the (denoised) r2 intensity, and measure the
   banded Immerkaer sigma of every model's output:

   - soft-tissue band: ``M3 ~= M2`` (the blend's cap on low-density recursion
     depth works -> soft tissue does NOT get pushed past r2);
   - bone band: ``M3 < M2`` (bone keeps recursing one step past r2);
   - every band: ``M3 < full`` (denoising direction is right).

Nothing is written to the dataset here -- each checkpoint denoises ``full``
in memory (overlapping-tile 3D inference, the same fusion used at deployment),
and only the banded/global noise numbers are emitted.  ``--save-dir`` optionally
dumps the per-case denoised volumes as ``full_denoised_<name>.raw``.

Example (server, 128-window chain, test split)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/anaconda3/envs/pytorch201/bin/python \
      scripts/eval_rounds_banded.py \
      --config configs/1609v2_unet3d_h2f_128.yaml \
      --manifest /data/raid0/yyc/AIReduceNoise2/splits_1609v2.json \
      --split test \
      --checkpoint M1=runs/1609v2_n2n_3d_h2f_128_dn_3d_128_r1/best.pt \
      --checkpoint M2=runs/1609v2_n2n_3d_h2f_128_dn_3d_128_r2/best.pt \
      --checkpoint M3=runs/1609v2_n2n_3d_h2f_128_dn_3d_128_r3/best.pt \
      --band-key M2 --lo 0.2 --hi 0.4 \
      --out outputs/eval_1609v2_h2f_128_banded_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.models import UNet2D, UNet2D_DS, UNet3D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import infer_volume, infer_volume_3d, infer_volume_3d_tiled
from cbct_n2n.training.metrics import band_noise_std_immerkaer, estimate_noise_std_immerkaer

TISSUE_FLOOR = 0.01  # below this mu value a voxel is air (excluded from every band)


def build_model(arch: str, model_cfg: dict) -> "torch.nn.Module":
    cfg = {k: v for k, v in model_cfg.items() if k != "arch"}
    if arch == "unet2d":
        return UNet2D(**cfg)
    if arch == "unet2d_ds":
        return UNet2D_DS(**cfg)
    if arch == "unet3d":
        return UNet3D(**cfg)
    raise ValueError(f"unknown model.arch {arch!r}; expected unet2d | unet2d_ds | unet3d")


def resolve_output(record: dict[str, Any], logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (f"{logical}.raw", logical):
        if key in outputs:
            return outputs[key]
    return None


def parse_checkpoint(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--checkpoint expects NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def band_differentiation(bands: dict[str, dict[str, float]], names: list[str]) -> dict[str, float]:
    """Density selectivity scalar: ``sigma(high) / sigma(low)`` per checkpoint.

    ``1.0`` means bone and soft tissue are smoothed equally hard (no density
    adaptation); below ``1.0`` means the model is deliberately gentler on soft
    tissue.  Plan B's r1 blend should produce a clearly sub-1.0 value for the
    first recursion round; later naive rounds erode it back toward 1.0.
    """
    out: dict[str, float] = {}
    for name in list(names) + ["full"]:
        lo_v = bands.get("low", {}).get(name)
        hi_v = bands.get("high", {}).get(name)
        out[name] = float(hi_v / lo_v) if lo_v else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Banded output-layer validation of the recursive rounds.")
    p.add_argument("--config", type=Path, required=True, help="Config used to build the model (voxel_depth, model block).")
    p.add_argument("--manifest", type=Path, required=True, help="Study-level manifest JSON.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True,
                   metavar="NAME=PATH", help="Repeatable, ordered (M1, M2, M3, ...).")
    p.add_argument("--band-key", default=None,
                   help="Checkpoint whose output keys the density bands (default: 2nd checkpoint, i.e. M2/r2).")
    p.add_argument("--lo", type=float, default=0.2, help="Low/mid band edge in mu (default 0.2).")
    p.add_argument("--hi", type=float, default=0.4, help="Mid/high band edge in mu (default 0.4).")
    p.add_argument("--arch", default=None, choices=("unet2d", "unet2d_ds", "unet3d"),
                   help="Override model.arch from the config.")
    p.add_argument("--tiled", action="store_true", default=True,
                   help="Overlapping-tile 3D inference (default; pass --no-tiled for the plain D-window path).")
    p.add_argument("--no-tiled", dest="tiled", action="store_false")
    p.add_argument("--patch-hw", type=int, default=256)
    p.add_argument("--overlap-d", type=int, default=16)
    p.add_argument("--overlap-hw", type=int, default=16)
    p.add_argument("--max-slices", type=int, default=24, help="Slices sampled per banded/global sigma.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="Only the first N studies (0 = all).")
    p.add_argument("--save-dir", type=Path, default=None, help="Optional per-case denoised volumes.")
    p.add_argument("--out", type=Path, required=True, help="Output overview JSON.")
    args = p.parse_args()

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    records = {str(r["key"]): r for r in manifest["records"]}
    keys = list(manifest["splits"][args.split])
    if args.limit:
        keys = keys[: args.limit]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    arch = args.arch or str(config.get("model", {}).get("arch", "unet3d"))
    voxel_depth = int(config["data"].get("voxel_depth", 16))

    models: dict[str, "torch.nn.Module"] = {}
    for name, path in args.checkpoint:
        model = build_model(arch, config["model"]).to(device)
        load_checkpoint(str(path), model, map_location=device)
        model.eval()
        models[name] = model
        print(f"[load] {name} <- {path}", flush=True)

    names = [n for n, _ in args.checkpoint]
    band_key = args.band_key or (names[1] if len(names) > 1 else names[0])
    if band_key not in models:
        raise SystemExit(f"--band-key {band_key!r} is not one of {names}")

    def denoise(vol: np.ndarray, model: "torch.nn.Module") -> np.ndarray:
        if arch == "unet3d":
            if args.tiled:
                return infer_volume_3d_tiled(model, vol, voxel_depth, args.patch_hw,
                                             args.overlap_d, args.overlap_hw, device, 1)
            return infer_volume_3d(model, vol, voxel_depth, device, 1)
        return infer_volume(model, vol, device, 1)

    print(f"[eval] split={args.split} studies={len(keys)} arch={arch} voxel_depth={voxel_depth} "
          f"bands lo={args.lo} hi={args.hi} band_key={band_key} device={device}", flush=True)

    per_case: list[dict[str, Any]] = []
    band_acc: dict[str, dict[str, list[float]]] = {
        band: {n: [] for n in names + ["full"]} for band in ("low", "mid", "high")
    }
    global_acc: dict[str, list[float]] = {n: [] for n in names + ["full"]}
    band_vox: dict[str, list[int]] = {"low": [], "mid": [], "high": []}

    for idx, key in enumerate(keys, 1):
        record = records[key]
        full_rel = resolve_output(record, "full")
        if not full_rel:
            print(f"[skip] {key}: no full source", flush=True)
            continue
        full_path = base / full_rel
        if not full_path.exists():
            print(f"[skip] {key}: {full_path} missing", flush=True)
            continue
        shape = tuple(int(d) for d in record["shape"])
        dtype = np.dtype(record.get("dtype") or "<f4")
        full = np.asarray(np.memmap(full_path, dtype=dtype, mode="r", shape=shape))

        t0 = time.time()
        outs: dict[str, np.ndarray] = {n: denoise(full, models[n]) for n in names}
        if args.save_dir:
            case_dir = args.save_dir / f"{record.get('group') or 'unknown'}__{record.get('study_uid') or key}"
            case_dir.mkdir(parents=True, exist_ok=True)
            from cbct_n2n.training.inference import write_raw_volume
            for n, vol in outs.items():
                write_raw_volume(case_dir / f"full_denoised_{n}.raw", vol)

        # density bands keyed on the (denoised) band-key output
        kvol = outs[band_key]
        masks = {
            "low": (kvol > TISSUE_FLOOR) & (kvol <= args.lo),
            "mid": (kvol > args.lo) & (kvol < args.hi),
            "high": kvol >= args.hi,
        }
        row: dict[str, Any] = {"key": key, "group": record.get("group"), "bands": {}, "global": {}}
        for band, mask in masks.items():
            n_vox = int(np.count_nonzero(mask[1:-1, 1:-1]))
            band_vox[band].append(n_vox)
            entry = {"full": band_noise_std_immerkaer(full, mask, max_slices=args.max_slices)}
            for n in names:
                entry[n] = band_noise_std_immerkaer(outs[n], mask, max_slices=args.max_slices)
                band_acc[band][n].append(entry[n])
            band_acc[band]["full"].append(entry["full"])
            entry["n_voxels"] = n_vox
            row["bands"][band] = entry
        row["global"] = {"full": estimate_noise_std_immerkaer(full, max_slices=args.max_slices)}
        for n in names:
            row["global"][n] = estimate_noise_std_immerkaer(outs[n], max_slices=args.max_slices)
            global_acc[n].append(row["global"][n])
        global_acc["full"].append(row["global"]["full"])
        per_case.append(row)

        hi_row = row["bands"]["high"]
        lo_row = row["bands"]["low"]
        print(f"[{idx}/{len(keys)}] {key} ({record.get('group')}) {time.time() - t0:.1f}s | "
              f"low {' '.join(f'{n}={lo_row[n]:.5f}' for n in names + ['full'])} | "
              f"high {' '.join(f'{n}={hi_row[n]:.5f}' for n in names + ['full'])}",
              flush=True)
        del full, outs, kvol, masks

    def agg(acc: dict[str, list[float]]) -> dict[str, float]:
        out = {}
        for name, vals in acc.items():
            arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
            out[name] = float(arr.mean()) if arr.size else float("nan")
        return out

    bands = {band: agg(band_acc[band]) for band in ("low", "mid", "high")}
    glob = agg(global_acc)
    for band in bands:
        bands[band]["mean_n_voxels"] = float(np.mean(band_vox[band])) if band_vox[band] else float("nan")

    # ---- verdict: the r3 semantics, band by band -------------------------
    verdict: dict[str, Any] = {}
    if len(names) >= 2:
        ref, new = names[-2], names[-1]  # r2 (reference) vs r3 (last round)
        tol = 0.02  # "cap" tolerance: 2% above r2 still counts as "not pushed past"
        lr = bands["low"][new] / bands["low"][ref] if bands["low"][ref] else float("nan")
        hr = bands["high"][new] / bands["high"][ref] if bands["high"][ref] else float("nan")
        mr = bands["mid"][new] / bands["mid"][ref] if bands["mid"][ref] else float("nan")
        verdict = {
            "reference": ref, "new": new,
            "low_ratio_new_over_ref": lr, "mid_ratio_new_over_ref": mr, "high_ratio_new_over_ref": hr,
            "soft_tissue_capped": bool(lr <= 1.0 + tol),
            "bone_pushed_past_ref": bool(hr < 1.0),
            "all_bands_below_full": bool(all(bands[b][new] < bands[b]["full"] for b in bands)),
            "global_below_ref": bool(glob[new] < glob[ref]),
            "global_reduction_vs_full_pct": float(100.0 * (1.0 - glob[new] / glob["full"])) if glob["full"] else float("nan"),
            "gate_pass": bool(lr <= 1.0 + tol and hr < 1.0 and all(bands[b][new] < bands[b]["full"] for b in bands)),
        }

    # ---- density selectivity: sigma(high) / sigma(low) per checkpoint -----
    # This is the plan-B judgement.  Plan A blended *inside* r3, so its gate is
    # "does the last round stop pushing soft tissue past the reference round"
    # (``verdict`` above, which is vacuous on a chain that decreases
    # monotonically everywhere).  Plan B instead blends at r1 and then recurses
    # naively, so the question is whether the band *differentiation* introduced
    # at r1 survives the later rounds.  We measure it directly: a ratio near 1.0
    # means the model denoises bone and soft tissue equally hard, i.e. no
    # density adaptation at all.
    diff = band_differentiation(bands, names)
    first_round = names[0] if names else None
    diff_vs_first = {
        n: float(diff[n] / diff[first_round])
        for n in names
        if first_round and diff[first_round] and np.isfinite(diff[n])
    } if first_round else {}

    overview = {
        "config": str(args.config), "manifest": str(args.manifest), "split": args.split,
        "n_studies": len(per_case), "arch": arch, "voxel_depth": voxel_depth,
        "tiled": bool(args.tiled), "patch_hw": args.patch_hw,
        "overlap_d": args.overlap_d, "overlap_hw": args.overlap_hw,
        "checkpoints": {n: str(pth) for n, pth in args.checkpoint},
        "band_key": band_key, "lo": args.lo, "hi": args.hi, "tissue_floor": TISSUE_FLOOR,
        "band_differentiation_high_over_low": diff,
        "band_differentiation_retained_vs_first_round": diff_vs_first,
        "note": (
            "Output-layer gate for the recursive rounds (the real r3 judgement). Each checkpoint "
            "denoises the same 'full' volume with overlapping-tile inference; density bands are keyed "
            "on the band_key (r2) output above the tissue floor. Expectation: soft tissue ~ r2 "
            "(capped) and bone < r2 (one recursion deeper), every band < full."
        ),
        "bands": bands, "global": glob, "verdict": verdict, "per_case": per_case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\noverview written: {args.out}")
    for band in ("low", "mid", "high"):
        b = bands[band]
        print(f"  [{band:>4}] " + " ".join(f"{n}={b[n]:.5f}" for n in names + ["full"]))
    print("  [glob] " + " ".join(f"{n}={glob[n]:.5f}" for n in names + ["full"]))
    if verdict:
        print(f"  verdict: soft_tissue_capped={verdict['soft_tissue_capped']} "
              f"(low {verdict['low_ratio_new_over_ref']:.3f}x), "
              f"bone_pushed_past_ref={verdict['bone_pushed_past_ref']} "
              f"(high {verdict['high_ratio_new_over_ref']:.3f}x), "
              f"gate_pass={verdict['gate_pass']}")
        print("  NOTE: 'verdict' is the plan-A (blend-inside-r3) semantics and is vacuous on a chain "
              "that decreases everywhere; for plan B read band_differentiation below.")
    print("  [diff] sigma(high)/sigma(low) -- 1.00 = no density adaptation:")
    print("         " + "  ".join(f"{n}={diff[n]:.3f}" for n in names + ["full"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
