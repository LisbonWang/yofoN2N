"""Blend r1 & r2 denoised full volumes into the r3 (density-adaptive) target.

Step 1.5 of the recursive scheme.  Unlike rounds 1/2 -- where a single teacher
denoises ``full`` to produce the next, globally-cleaner target -- round 3 mixes
two already-written *model outputs* voxel-wise, keyed on the r2 intensity (the
denoised CT value in the mu domain):

    r1 = M1(full) = r1-round model output = full.dn_3d_128_r2.raw  (conservative)
    r2 = M2(full) = r2-round model output = full.dn_3d_128_r3.raw  (aggressive)

NOTE the off-by-one: the *file* suffix ``dn_rN`` is the target of round N, i.e.
the output of model M_{N-1}.  So "r1's result" = M1(full) = file dn_r2, and
"r2's result" = M2(full) = file dn_r3.  Reading dn_r1/dn_r2 here would mix
M0/M1 and be one round too conservative.

    w(v) = 0                             v <= lo          -> r1 (conservative)
         = smoothstep((v-lo)/(hi-lo))    lo < v < hi      -> smooth crossfade
         = 1                             v >= hi          -> r2 (aggressive)

    full.<suffix>.raw = r1 + w(r2) * (r2 - r1)

This is a *spatially adaptive* target, not a globally-cleaner one: its global
noise level sits between r1 and r2 (bone-volume-fraction weighted), so the
global immerkaer "delta > 0" gate still holds but the reduction no longer
monotonically exceeds r2.  We therefore also gate per density band (low / mid /
high, keyed on r2 above a tissue floor) so the blend never exceeds the noisier
parent in any band.

No model is loaded -- pure memmap I/O + numpy, a few minutes for the whole
train+val set.  Both source volumes must already exist (written by rounds 1/2
target generation) and be registered in the manifest.
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

from cbct_n2n.config import load_config
from cbct_n2n.training.blend import blend_volumes
from cbct_n2n.training.inference import write_raw_volume
from cbct_n2n.training.metrics import band_noise_std_immerkaer

_TISSUE_FLOOR = 0.01  # mu threshold below which voxels are air (excluded from sigma)


def _resolve_output(record: dict, logical: str) -> str | None:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in (logical, f"{logical}.raw"):
        if key in outputs:
            return outputs[key]
    return None


def _band_immerkaer(vol: np.ndarray, mask: np.ndarray, max_slices: int = 24) -> float:
    """Banded Immerkaer sigma -- thin alias of the shared estimator.

    Kept as a local name because the band gate below reads better with it;
    the implementation (and therefore the numbers) is identical to
    :func:`cbct_n2n.training.metrics.band_noise_std_immerkaer`, which the
    output-layer gate ``scripts/eval_rounds_banded.py`` also uses -- so target
    layer and output layer report the same quantity.
    """
    return band_noise_std_immerkaer(vol, mask, max_slices=max_slices)


def main() -> int:
    p = argparse.ArgumentParser(description="Blend r1 & r2 targets into the r3 density-adaptive target.")
    p.add_argument("--config", type=Path, default=None, help="Config to read data.manifest from.")
    p.add_argument("--manifest", type=Path, help="Manifest JSON (default: config data.manifest).")
    p.add_argument("--r1-suffix", default="dn_3d_128_r2",
                   help="Conservative (low-density) source. Either a target suffix "
                        "(= M_{N-1}(full) model output) or the literal 'full' to blend "
                        "from the raw full reconstruction itself (blend-at-round-1 scheme).")
    p.add_argument("--r2-suffix", default="dn_3d_128_r3", help="Aggressive (high-density) source = M2(full) = r2-round output (also the blend key).")
    p.add_argument("--target-suffix", default="dn_3d_128_r3_blend", help="Blended target suffix.")
    p.add_argument("--lo", type=float, default=0.2, help="Below this mu value use r1 (default 0.2).")
    p.add_argument("--hi", type=float, default=0.4, help="Above this mu value use r2 (default 0.4).")
    p.add_argument("--kind", default="smoothstep", choices=("smoothstep", "cosine", "linear"))
    p.add_argument("--mask-source", default="high", choices=("low", "high"),
                   help="Key the blend on 'high' (=r2, denoised) intensity (default) or 'low'.")
    p.add_argument("--splits", default="train,val", help="Comma-separated splits (never include test).")
    p.add_argument("--max-slices", type=int, default=24, help="Slices sampled for per-band sigma (default 24).")
    args = p.parse_args()

    if args.manifest:
        manifest_path = args.manifest
    elif args.config:
        manifest_path = Path(load_config(args.config)["data"]["manifest"])
    else:
        raise SystemExit("need --manifest or --config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.resolve().parent
    lo, hi = args.lo, args.hi
    if hi <= lo:
        raise SystemExit(f"hi ({hi}) must be > lo ({lo})")

    records = {str(r["key"]): r for r in manifest["records"]}
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits:
        print("[warn] 'test' must stay leakage-free; removing it.", flush=True)
        splits = [s for s in splits if s != "test"]
    keys = [k for s in splits for k in manifest["splits"][s]]
    print(f"[blend] lo={lo} hi={hi} kind={args.kind} mask_source={args.mask_source} "
          f"splits={splits} studies={len(keys)}", flush=True)

    band_rows: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    n_new = n_existing = n_skip = 0
    for key in keys:
        record = records[key]
        full_rel = _resolve_output(record, "full")
        if args.r1_suffix in ("", "full"):
            # blend-at-round-1: the low side IS the raw full reconstruction, so the
            # soft-tissue band trains from `full` (recursion starts one round later
            # there) while the bone band starts from M0(full) = full.<r2-suffix>.
            r1_rel = full_rel
        else:
            r1_rel = _resolve_output(record, f"full.{args.r1_suffix}")
        r2_rel = _resolve_output(record, f"full.{args.r2_suffix}")
        if not (r1_rel and r2_rel and full_rel):
            n_skip += 1
            print(f"[skip] {key}: missing r1/r2/full source in outputs", flush=True)
            continue
        r1_path, r2_path, full_path = base / r1_rel, base / r2_rel, base / full_rel
        if not (r1_path.exists() and r2_path.exists() and full_path.exists()):
            n_skip += 1
            print(f"[skip] {key}: source file missing on disk", flush=True)
            continue

        dst = full_path.with_name(f"{full_path.stem}.{args.target_suffix}.raw")
        if dst.exists():
            n_existing += 1
            print(f"[exists] {key}: {dst.name}", flush=True)
            continue

        shape = tuple(int(d) for d in record["shape"])
        dtype = np.dtype(record.get("dtype") or "<f4")
        r1 = np.asarray(np.memmap(r1_path, dtype=dtype, mode="r", shape=shape))
        r2 = np.asarray(np.memmap(r2_path, dtype=dtype, mode="r", shape=shape))
        target = blend_volumes(r1, r2, lo, hi, args.kind, args.mask_source)
        write_raw_volume(dst, target)

        # per-density-band gate, keyed on the (denoised) r2 intensity
        band_mask = {
            "low": (r2 > _TISSUE_FLOOR) & (r2 <= lo),
            "mid": (r2 > lo) & (r2 < hi),
            "high": r2 >= hi,
        }
        full = np.asarray(np.memmap(full_path, dtype=dtype, mode="r", shape=shape))
        for band, mask in band_mask.items():
            if not np.any(mask):
                band_rows[band].append({k: float("nan") for k in ("full", "r1", "r2", "target")})
                continue
            band_rows[band].append({
                "full": _band_immerkaer(full, mask, args.max_slices),
                "r1": _band_immerkaer(r1, mask, args.max_slices),
                "r2": _band_immerkaer(r2, mask, args.max_slices),
                "target": _band_immerkaer(target, mask, args.max_slices),
            })
        del full, r1, r2, target, band_mask
        n_new += 1
        hi_row = band_rows["high"][-1]
        print(f"[done] {key} ({record.get('group')}): high-band immerkaer "
              f"r1={hi_row['r1']:.5f} r2={hi_row['r2']:.5f} target={hi_row['target']:.5f}", flush=True)

    # register the blended target in the manifest (idempotent)
    outputs_key = f"full.{args.target_suffix}.raw"
    patched = 0
    for key in keys:
        record = records[key]
        full_rel = _resolve_output(record, "full")
        if not full_rel:
            continue
        relpath = Path(full_rel).with_name(f"{Path(full_rel).stem}.{args.target_suffix}.raw").as_posix()
        outputs = record.setdefault("outputs", {})
        if outputs.get(outputs_key) != relpath:
            outputs[outputs_key] = relpath
            patched += 1
    if patched:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[manifest] patched {patched} records with {outputs_key}", flush=True)

    # aggregate per-band sigma (mean across studies)
    agg_bands: dict[str, dict] = {}
    for band, rows in band_rows.items():
        agg_bands[band] = {
            k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
            for k in ("full", "r1", "r2", "target")
        }
        agg_bands[band]["n"] = len(rows)
    overview = {
        "splits": splits, "lo": lo, "hi": hi, "kind": args.kind,
        "mask_source": args.mask_source, "target_suffix": args.target_suffix,
        "n_new": n_new, "n_existing": n_existing, "n_skip_missing": n_skip,
        "note": (
            "r3 TARGET = r1 + w(r2)*(r2-r1), w=smoothstep over [lo,hi] keyed on r2 intensity. "
            "This is the TARGET layer only: per-band sigma here checks target correctness "
            "(low ~ r1, high ~ r2, target <= max(r1,r2)). Under half->full training the "
            "student output is one step cleaner than its target, so the FINAL effect is "
            "soft tissue ~ r2 (NOT r1) and bone > r2 -- measure the student output after "
            "training for the real gate."
            if args.r1_suffix not in ("", "full") else
            "BLEND-AT-ROUND-1 target: low band = raw full (soft-tissue recursion starts one "
            "round later), high band = M0(full) = full.<r2-suffix>; w=smoothstep over [lo,hi] "
            "keyed on the M0 output intensity. Target-layer gate here: low ~ full, high ~ M0 "
            "output, target within [full, M0] everywhere. The rounds after this one recurse "
            "plainly (half -> previous teacher output), so the density-dependent depth set "
            "here propagates automatically."
        ),
        "bands": agg_bands,
    }
    overview_path = base / f"blend_full_{args.target_suffix}_overview.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\noverview written: {overview_path}")
    for band, a in agg_bands.items():
        print(f"  [{band}] full={a['full']:.5f} r1={a['r1']:.5f} r2={a['r2']:.5f} "
              f"target={a['target']:.5f} (n={a['n']})")
    print(f"summary: new={n_new} existing={n_existing} skipped={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
