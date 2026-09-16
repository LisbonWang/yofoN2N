"""QC for a single-volume denoise run (``scripts/denoise_single_raw.py`` output).

Given the input RAW and the denoised RAW (identical headerless float32 shape),
reports:

* tissue-masked Immerkaer noise std before/after (the project's gating metric);
* air / tissue mean bias -- N2N output should not shift the background or the
  tissue mean; a non-trivial tissue shift means domain shift or H/W padding;
* a coarse input->output transfer curve (value-binned mean delta);
* a D-axis seam check: per-slice tissue residual at ``z % voxel_depth == 0``
  boundaries vs interior, which catches artefacts from the non-overlapping
  sliding window.

With ``--figure`` it additionally renders an input-vs-denoised comparison PNG
(three axials + coronal + sagittal) and a 2x zoom texture pair.

Example:
  python scripts/qc_single_denoise.py \\
    --input D:/vol-origin.raw --output D:/vol-origin-denoised.raw \\
    --shape 520 680 680 --voxel-depth 16 \\
    --figure outputs/qc_vol.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cbct_n2n.training.metrics import estimate_noise_std_immerkaer


def _load(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path.name}: {path.stat().st_size} bytes != {expected} for shape {shape} float32."
        )
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def _as_stack(volume: np.memmap, step: int) -> np.ndarray:
    return np.asarray(volume[::step], dtype=np.float32)


def report(src: np.memmap, dst: np.memmap, voxel_depth: int, step: int) -> dict:
    n = int(src.shape[0])
    a = _as_stack(src, step)
    b = _as_stack(dst, step)

    sigma_in = estimate_noise_std_immerkaer(np.asarray(src[:: max(1, n // 32)], dtype=np.float32))
    sigma_out = estimate_noise_std_immerkaer(np.asarray(dst[:: max(1, n // 32)], dtype=np.float32))

    from scipy.ndimage import uniform_filter

    local = np.stack([uniform_filter(s, size=9, mode="nearest") for s in a])
    air, tissue = local < 0.005, local > 0.05

    print("== volume ==")
    print(f"  shape={tuple(src.shape)} dtype=float32  sampled {a.shape[0]} slices (step {step})")
    print("== noise (Immerkaer, tissue-masked) ==")
    print(f"  in  sigma = {sigma_in:.5f}")
    print(f"  out sigma = {sigma_out:.5f}")
    print(f"  delta     = {100 * (sigma_in - sigma_out) / sigma_in:+.1f}%")
    print("== bias ==")
    print(f"  air    in={a[air].mean():+.5f} out={b[air].mean():+.5f} d={(b - a)[air].mean():+.5f}")
    rel = 100 * (b - a)[tissue].mean() / a[tissue].mean()
    print(f"  tissue in={a[tissue].mean():.5f} out={b[tissue].mean():.5f} "
          f"d={(b - a)[tissue].mean():+.5f} (rel {rel:+.2f}%)")
    print("== transfer curve (binned by input local mean) ==")
    edges = [0.005, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.3]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (local >= lo) & (local < hi)
        if m.sum() < 500:
            continue
        print(f"  [{lo:5.3f},{hi:5.3f}) n={int(m.sum()):>9d} "
              f"in={a[m].mean():.4f} out={b[m].mean():.4f} d={(b - a)[m].mean():+.5f}")

    # D-axis seam check on the mid-volume slab, tissue-masked per slice.
    # Sample on a grid aligned to the window grid, otherwise no sample ever
    # lands on a ``z % voxel_depth == 0`` boundary and the check is vacuous.
    half = max(1, voxel_depth // 2)
    lo_z, hi_z = n // 4, 3 * n // 4
    lo_z = -(-lo_z // half) * half
    prof = []
    for z in range(lo_z, hi_z, half):
        s = np.asarray(src[z], dtype=np.float32)
        d = np.asarray(dst[z], dtype=np.float32)
        m = s > 0.05
        prof.append((z, float((d[m] - s[m]).mean()) if m.sum() > 1000 else np.nan))
    zs = np.array([p[0] for p in prof], dtype=float)
    vals = np.array([p[1] for p in prof])
    on_boundary = np.isin(zs % voxel_depth, [0])
    if np.isfinite(vals).all() and on_boundary.any() and (~on_boundary).any():
        edge, inner = float(np.nanmean(vals[on_boundary])), float(np.nanmean(vals[~on_boundary]))
        print("== D-axis seam check ==")
        print(f"  slice-mean tissue residual: {np.nanmean(vals):+.5f} +- {np.nanstd(vals):.5f} "
              f"(min {np.nanmin(vals):+.5f} max {np.nanmax(vals):+.5f})")
        print(f"  z%{voxel_depth}==0: {edge:+.5f}   interior: {inner:+.5f}   "
              f"delta {abs(edge - inner):.5f} vs profile std {np.nanstd(vals):.5f} -> "
              f"{'SEAM SUSPECT' if abs(edge - inner) > 2 * np.nanstd(vals) else 'no seam'}")

    return {"sigma_in": float(sigma_in), "sigma_out": float(sigma_out),
            "tissue_rel_pct": float(rel), "air_d": float((b - a)[air].mean())}


def figure(src: np.memmap, dst: np.memmap, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, h, _ = src.shape
    lo, hi = float(min(src.min(), dst.min())), float(np.percentile(np.asarray(src[::25]), 99.5))
    zs = [n // 4, n // 2, 3 * n // 4]
    cy, cx = h // 2, h // 2

    fig, axes = plt.subplots(2, 5, figsize=(23, 10))
    for i, (vol, name) in enumerate(((src, "origin"), (dst, "denoised (3D r1)"))):
        a = np.asarray(vol)
        views = [("axial z=%d" % z, a[z]) for z in zs]
        views.append(("coronal y=%d (z-x)" % cy, a[:, cy, :]))
        views.append(("sagittal x=%d (z-y)" % cx, a[:, :, cx]))
        for j, (title, img) in enumerate(views):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title("%s  |  %s" % (title, name), fontsize=11)
            ax.axis("off")
    fig.suptitle("%s -> denoised by 1609v2_n2n_3d_h2f_v2_dn_3d_r1" % src.filename, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("figure ->", out)

    # pick the most textured 160x160 patch on the mid-volume slab so the
    # zoom actually shows structure instead of flat soft tissue
    half = 80
    mid = np.asarray(src[n // 2], dtype=np.float32)
    best, bz, by, bx = -1.0, n // 2, 0, 0
    for z in (n // 4, n // 2, 3 * n // 4):
        s = np.asarray(src[z], dtype=np.float32)
        for y in range(0, s.shape[0] - 2 * half, 40):
            for x in range(0, s.shape[1] - 2 * half, 40):
                v = float(s[y:y + 2 * half, x:x + 2 * half].std())
                if v > best:
                    best, bz, by, bx = v, z, y, x
    patch = (slice(by, by + 2 * half), slice(bx, bx + 2 * half))
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (vol, name) in zip(ax2, ((src, "origin"), (dst, "denoised (3D r1)"))):
        ax.imshow(np.asarray(vol)[bz][patch], cmap="gray", vmin=lo, vmax=hi,
                  interpolation="nearest")
        ax.set_title("z=%d zoom 160x160 (most textured patch) | %s" % (bz, name), fontsize=10)
        ax.axis("off")
    fig2.tight_layout()
    zoom_out = out.with_name(out.stem + "_zoom" + out.suffix)
    fig2.savefig(zoom_out, dpi=150)
    print("figure ->", zoom_out)


def main() -> int:
    p = argparse.ArgumentParser(description="QC a single-volume denoise run.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True)
    p.add_argument("--voxel-depth", type=int, default=16)
    p.add_argument("--step", type=int, default=10, help="z sub-sampling step for the bias scan")
    p.add_argument("--figure", type=Path)
    args = p.parse_args()

    shape = tuple(args.shape)
    src, dst = _load(args.input, shape), _load(args.output, shape)
    report(src, dst, args.voxel_depth, args.step)
    if args.figure:
        figure(src, dst, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
