"""Render axial slices of denoised eval outputs for visual comparison.

For selected case dirs of shape (Z, H, W) float32 raw volumes, build one
comparison figure per case: rows = volume type (noisy half source, per-model
even_denoised, full reference, full_denoised), columns = axial slice indices.
A center-zoom strip on the mid slice is appended for fine-detail inspection.

Example (replace paths with locations available to your machine):
  python scripts/render_denoised_slices.py \
    --root /path/to/dataset-root \
    --outdir outputs/slice_cmp \
    --case 'Mode2_1609__<uid>' --case 'Mode3_1609__<uid>'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# volume naming in a case dir.  E2/E3/E4 filled dirs share this layout.
# ---------------------------------------------------------------------------
VOLS = [
    ("even",            "noisy half (even)",            "even.raw"),
    ("e2",              "E2 half→full",                 "even_denoised.raw"),
    ("e3",              "E3 joint",                     "even_denoised.raw"),
    ("e4",              "E4 half→full.dn",              "even_denoised.raw"),
    ("full",            "full reference",               "full.raw"),
    ("e2f",             "E2 full→full (out-of-domain)", "full_denoised.raw"),
    ("e3f",             "E3 full→full (out-of-domain)", "full_denoised.raw"),
    ("e4f",             "E4 full→full (out-of-domain)", "full_denoised.raw"),
]


def _load(root: Path, exp: str, case: str, fname: str, shape) -> np.ndarray:
    p = root / exp / case / fname
    if not p.exists():
        return None
    return np.memmap(p, dtype="<f4", mode="r", shape=shape)


def _window(v: np.ndarray | None) -> tuple[float, float]:
    """Global display window from the reference volume (robust percentiles)."""
    if v is None:
        return 0.0, 4.0
    lo, hi = np.percentile(v[::8, ::8, ::8], (0.5, 99.5))
    if hi - lo < 1e-6:
        return float(lo - 1.0), float(hi + 1.0)
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Root directory containing rendered experiment outputs.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/slice_cmp"))
    ap.add_argument("--case", action="append", required=True, help="case dir name")
    ap.add_argument("--zs", type=int, nargs="+", default=[90, 180, 270])
    ap.add_argument("--zoom", type=int, default=180, help="center crop half-size (px) for zoom strip")
    ap.add_argument("--manifests", type=Path, required=True,
                    help="Manifest describing the source records and shapes.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifests).read_text(encoding="utf-8"))
    records = {str(r["key"]): r for r in manifest["records"]}
    root = args.root
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for case in args.case:
        # derive record key -> shape from manifest (case dir name group__uid)
        group, uid = case.split("__", 1)
        rec = records.get(f"{uid}::{group}")
        if rec is None:
            print(f"[warn] {case}: not in manifest, skip", flush=True)
            continue
        shape = tuple(int(d) for d in rec["shape"])
        zmid = shape[0] // 2
        zs = [z if 0 <= z < shape[0] else zmid for z in args.zs]

        full = _load(root, "denoised_1609_fdn", case, "full.raw", shape)
        vmin, vmax = _window(full)
        print(f"[render] {case} shape={shape} window=[{vmin:.3f}, {vmax:.3f}]", flush=True)

        # metric labels per model (even_denoised vs full)
        labels: dict[str, str] = {}
        for exp, label, _fn in VOLS:
            mfile = root / exp / case / "even_metrics.json"
            if (root / exp / case / "even_denoised.raw").exists() and mfile.exists():
                m = json.loads(mfile.read_text(encoding="utf-8"))["metrics"]
                labels[label] = f"PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}"

        rows: list[tuple[str, str, np.ndarray | None]] = []
        exp_map = {"e2": "denoised_1609_fullview", "e3": "denoised_1609_joint",
                   "e4": "denoised_1609_fdn", "e2f": "denoised_1609_fullview",
                   "e3f": "denoised_1609_joint", "e4f": "denoised_1609_fdn",
                   "even": "denoised_1609_fdn", "full": "denoised_1609_fdn"}
        for tag, label, fn in VOLS:
            exp = exp_map[tag]
            arr = _load(root, exp, case, fn, shape)
            if arr is None and tag in ("even", "full"):
                # fall back to a filled dir that already has sources
                for alt in ("denoised_1609_fullview",):
                    arr = _load(root, alt, case, fn, shape)
                    if arr is not None:
                        break
            rows.append((tag, label, arr))

        nrow, ncol = len(rows), len(zs)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow))
        axes = np.atleast_2d(axes)
        for i, (tag, label, arr) in enumerate(rows):
            for j, z in enumerate(zs):
                ax = axes[i, j]
                if arr is None:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                    ax.axis("off")
                    continue
                ax.imshow(arr[z], cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f"{label}\n{labels.get(label, '')}", fontsize=9)
                if i == 0:
                    ax.set_title(f"z={z}", fontsize=10)
        fig.tight_layout()

        # zoom strip on the mid slice: noisy source vs each model vs full
        h, w = shape[1], shape[2]
        cy, cx = h // 2, w // 2
        zs_ = args.zoom
        y0, y1 = max(cy - zs_, 0), min(cy + zs_, h)
        x0, x1 = max(cx - zs_, 0), min(cx + zs_, w)
        zoom_rows = [r for r in rows if r[0] in ("even", "e2", "e3", "e4", "full")]
        nzoom = len(zoom_rows)
        fig2, axs2 = plt.subplots(1, nzoom, figsize=(3.0 * nzoom, 3.4))
        if nzoom == 1:
            axs2 = [axs2]
        for k, (tag, label, arr) in enumerate(zoom_rows):
            ax = axs2[k]
            if arr is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                ax.axis("off")
                continue
            ax.imshow(arr[zmid, y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label.split("\n")[0], fontsize=9)
        fig2.suptitle(f"{case}  z={zmid}  zoom {2 * zs_}×{2 * zs_}px center", fontsize=10)
        fig2.tight_layout()

        out_main = outdir / f"cmp_{case}.png"
        out_zoom = outdir / f"cmp_{case}_zoom.png"
        fig.savefig(out_main, dpi=150)
        fig2.savefig(out_zoom, dpi=150)
        plt.close(fig)
        plt.close(fig2)
        print(f"[saved] {out_main}\n[saved] {out_zoom}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
