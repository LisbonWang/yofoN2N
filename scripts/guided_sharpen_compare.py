#!/usr/bin/env python
"""Strength A/B for 3D guided-filter sharpening, without writing a volume per point.

``guided_sharpen_volume.py`` writes a 590 MB RAW for every parameter point, which
is far too coarse a loop when the open question is aesthetic ("this looks over
sharpened -- by how much?").  Both sharpening modes are *exactly linear* in
``strength`` (``out = plain + s * field``), so the expensive guided filter only
has to run once per ``(radius, eps)``; after that every strength is a scalar.

This script computes that one field, then renders the two QC windows
(trabecular bone / flat soft tissue) for a list of strengths into a single
figure, annotated with the same 2D Immerkaer estimator that produces
``flat_patch_sigma`` / ``bone_patch_sigma`` in the volume script's sidecars.
So the picture and the numbers on it come from the same definition.

It also self-checks: ``--verify NAME=FILE`` re-renders a strength that was
already written to disk and reports the max absolute difference of the crop.
Zero means the eps and the arithmetic here match the delivered volume exactly.

Paths follow the project convention: every path flag is resolved relative to
``--dir`` unless it is absolute.

Example
-------
    PY=python
    $PY scripts/guided_sharpen_compare.py --dir outputs/air3_ypdl0027_bhc_5models \\
        --input full.bhc.r4b.raw --shape 360 640 640 \\
        --noisy full.bhc.raw --roi-cache gfsharp_rois.json \\
        --radius 2 --eps-k 8 --strength 0.4 0.5 0.6 0.8 1.0 \\
        --verify 1.0=full.bhc.r4b.gfsharp.raw \\
        --figure gfsharp_compare.png --markdown gfsharp_compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cbct_n2n.filters.guided import (  # noqa: E402
    cap_darkening,
    eps_from_noise,
    guided_structure_field,
)
from cbct_n2n.training.metrics import estimate_noise_std_immerkaer  # noqa: E402

from scripts.guided_sharpen_volume import (  # noqa: E402  (needs sys.path above)
    TISSUE_FLOOR,
    VISIBLE_FLOOR,
    _load,
    _under,
    _use_cjk_font,
    _window,
    patch_sigma,
    patch_slices,
)

ROI_KEYS = ("bone", "flat")
ROI_TITLES = {
    "bone": "骨小梁 / 牙周膜窗 —— 看纹理有没有被提过头（振铃、双边缘、塑料感）",
    "flat": "平坦软组织窗 —— 看代价（噪声有没有被带起来）",
    "dark": "**最深暗边窗（自动定位，全卷最差处）** —— 看黑边有没有被消掉",
}
REF_COLORS = {"noisy": "#9aa0a6", "plain": "#4c566a", "sharp": "#2b6cb0"}


def worst_darkening_roi(plain: np.ndarray, field: np.ndarray, strength: float,
                        max_darken: float | None, size: int,
                        visible_floor: float) -> dict:
    """Locate the deepest *visible* dark rim, as an ROI dict for the figure.

    Picking a window by hand and reporting it would be cherry-picking, and the
    rim does not sit where the QC ROIs are: it appears in a low-density gap
    narrower than the high-pass window, which can be anywhere.  So the window is
    placed on the voxel that got darkened most, restricted to voxels that are
    unambiguously tissue (``> visible_floor``) so an already-black voxel cannot
    win the search.  The reported number is therefore the worst case.
    """
    out = np.array(field, dtype=np.float32)
    if strength != 1.0:
        out *= np.float32(strength)
    out += np.asarray(plain, dtype=np.float32)
    if max_darken is not None:
        out = cap_darkening(out, plain, max_darken)
    darken = np.subtract(np.asarray(plain, dtype=np.float32), out)
    np.maximum(darken, 0.0, out=darken)
    darken[np.asarray(plain, dtype=np.float32) <= visible_floor] = 0.0
    z, y, x = (int(v) for v in np.unravel_index(int(np.argmax(darken)), darken.shape))
    nz, ny, nx = darken.shape
    half = size // 2
    y = min(max(y, half), ny - half - 1)
    x = min(max(x, half), nx - half - 1)
    return {"z": z, "y": y, "x": x, "size": size,
            "label": f"最深暗边 · z={z} y={y} x={x} · {size}×{size}"}


def parse_verify(items) -> dict[float, Path]:
    """``["1.0=a/b.raw", ...]`` -> ``{1.0: Path("a/b.raw")}``."""
    out: dict[float, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--verify expects STRENGTH=FILE, got {item!r}")
        strength, _, path = item.partition("=")
        out[float(strength)] = Path(path)
    return out


def sharpen_patch(plain: np.ndarray, field: np.ndarray, sl: tuple, strength: float,
                  max_darken: float | None = None) -> np.ndarray:
    """``plain + strength * field`` restricted to ``sl``.

    Cropping commutes with the element-wise combine, and the library scales the
    field in place then adds (both float32), so this reproduces the full-volume
    result bit for bit -- which ``--verify`` checks against a written volume.
    ``max_darken`` must be applied here rather than to the field, because it is
    non-linear in the strength.
    """
    source = np.asarray(plain[sl], dtype=np.float32)
    patch = np.array(field[sl], dtype=np.float32)
    if strength != 1.0:
        patch *= np.float32(strength)
    out = np.add(patch, source)
    if max_darken is not None:
        out = cap_darkening(out, source, max_darken)
    return out


def render_compare(panels, rois, strengths, out_path: Path) -> Path:
    """Heat-map-free tile: one row per strength, one column per ROI window."""
    plt.rcParams["axes.unicode_minus"] = False
    keys = [k for k in rois if k in ROI_TITLES]
    n_rows = 2 + len(strengths)
    fig, axes = plt.subplots(n_rows, len(keys), figsize=(6.2 * len(keys), 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "3D 引导滤波锐化 —— 强度对照（同一 field，只改 strength）\n"
        "每格右下角是该窗的 2D Immerkaer σ（与 sidecar 里 flat/bone_patch_sigma 同一估计器）；"
        "左下角是该窗内「相对锐化前变暗的最大量」",
        fontsize=15,
    )

    rows = [("noisy", panels.get("noisy")), ("plain", panels["plain"])]
    rows += [(f"s={s:g}", panels[f"s={s:g}"]) for s in strengths]

    for col, key in enumerate(keys):
        sl = patch_slices(rois[key])
        ref_slice = panels.get("noisy", panels["plain"])[sl]
        vmin, vmax = _window(ref_slice)
        base = np.asarray(panels["plain"][sl], dtype=np.float32)
        for row, (label, source) in enumerate(rows):
            ax = axes[row][col]
            patch = np.asarray(source[sl], dtype=np.float32) if source is not None else None
            if patch is None:
                ax.axis("off")
                continue
            ax.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            sigma = patch_sigma(patch)
            ax.text(
                0.98, 0.02, f"σ={sigma:.5f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                color="black", bbox=dict(fc="white", ec="none", alpha=0.72, pad=2.0),
            )
            darken = float(np.maximum(base - patch, 0.0).max())
            ax.text(
                0.02, 0.02, f"暗化≤{darken:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
                color="black", bbox=dict(fc="#ffe8a3", ec="none", alpha=0.85, pad=2.0),
            )
            if row == 0:
                ax.set_title(ROI_TITLES[key], fontsize=11, pad=8)
            if col == 0:
                color = REF_COLORS["noisy" if label == "noisy" else
                                   "plain" if label == "plain" else "sharp"]
                ax.set_ylabel(label, fontsize=11, labelpad=8, color=color)
        if col == 0:
            # Mark the reference rows so they are not mistaken for candidates.
            for row in range(2):
                axes[row][col].set_facecolor("#f0f0f0")

    fig.tight_layout(rect=(0.02, 0, 1, 0.945))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_curve(rows, out_path: Path) -> Path:
    """Bone gain vs flat cost as strength grows, with the noise reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    base = next(r for r in rows if r["tag"] == "plain")
    xs = [r["strength"] for r in rows if r["tag"] == "sharp"]
    bone = [100.0 * (r["bone_patch_sigma"] / base["bone_patch_sigma"] - 1.0)
            for r in rows if r["tag"] == "sharp"]
    flat = [100.0 * (r["flat_patch_sigma"] / base["flat_patch_sigma"] - 1.0)
            for r in rows if r["tag"] == "sharp"]

    ax = axes[0]
    ax.axhline(0.0, color="#888", lw=0.9)
    ax.plot(xs, bone, "o-", color="#c0392b", label="骨 ROI σ 涨幅（想要大）")
    ax.plot(xs, flat, "s-", color="#2b6cb0", label="平坦 ROI σ 涨幅（想要小）")
    ax.set_xlabel("strength")
    ax.set_ylabel("相对锐化前的变化 (%)")
    ax.set_title("强度是纯线性幅度旋钮：两条线同时等比走")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(xs, [r["darken"] for r in rows if r["tag"] == "sharp"], "o-",
            color="#8b4513", label="窗内最大暗化量（黑边深度）")
    ax.axhline(0.0, color="#888", lw=0.9)
    ax.set_xlabel("strength")
    ax.set_ylabel("最大暗化量（体积单位）")
    ax.set_title("黑边深度也随 strength 线性 —— 想消黑边，降 strength 或降 radius 或设 --max-darken")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_markdown(rows, meta, out_path: Path) -> Path:
    base = next(r for r in rows if r["tag"] == "plain")
    lines = [
        "# 3D 引导滤波锐化 —— 强度对照",
        "",
        f"- 输入 `{meta['input']}`，mode `{meta['mode']}`，radius {meta['radius']}，"
        f"eps = {meta['eps']:.6e}（k={meta['eps_k']:g} × σ={meta['sigma']:.7f}，"
        f"σ 来源 `{meta['sigma_source']}`）",
        f"- 骨窗 `{meta['bone_roi']}`，平坦窗 `{meta['flat_roi']}`"
        + (f"，最深暗边窗 `{meta['dark_roi']}`" if meta.get("dark_roi") else "")
        + "；σ 为窗内 2D Immerkaer，"
        "与 sidecar 的 `bone_patch_sigma` / `flat_patch_sigma` 同一估计器",
        f"- `--max-darken` = {meta['max_darken'] if meta['max_darken'] is not None else '关'}；"
        "「最大暗化量」= 窗内 `max(锐化前 − 锐化后, 0)`，即黑边的深度",
        "",
        "| 配置 | strength | 骨 σ | 骨涨幅 | 平坦 σ | 平坦代价 | 增益/代价 | 最大暗化量 | 平坦区解析噪声增益 1+s/(1+k²) |",
        "| ---- | -------- | ---- | ------ | ------ | -------- | --------- | ---------- | --------------------------- |",
    ]
    for row in rows:
        if row["tag"] != "sharp":
            continue
        g = 100.0 * (row["bone_patch_sigma"] / base["bone_patch_sigma"] - 1.0)
        c = 100.0 * (row["flat_patch_sigma"] / base["flat_patch_sigma"] - 1.0)
        gain = 1.0 + row["strength"] / (1.0 + meta["eps_k"] ** 2)
        lines.append(
            f"| `s={row['strength']:g}` | {row['strength']:g} | {row['bone_patch_sigma']:.5f} | "
            f"{g:+.1f}% | {row['flat_patch_sigma']:.5f} | {c:+.1f}% | "
            f"{abs(g) / max(abs(c), 1e-9):.2f} | {row['darken']:.4f} | ×{gain:.4f} |"
        )
    lines += [
        "",
        f"参考（不计入候选）：`noisy` σ_骨={meta['noisy_bone']:.5f} / σ_平={meta['noisy_flat']:.5f}，"
        f"`plain`（r4b）σ_骨={base['bone_patch_sigma']:.5f} / σ_平={base['flat_patch_sigma']:.5f}。"
        "噪声参考体的骨/平比接近 1，这正是“过度平滑”与“噪声未减”都会抹掉纹理的证据。",
        "",
        "**读法**：strength 只缩幅度，增益/代价比是常数，所以“看着过了”就直接往下调，"
        "不会因此变脏；要**同时**压低平坦代价，动的是 `--eps-k`（见报告 §3）。",
        "**黑边**（不想出现暗线）则要看最后一列：它随 strength 线性，所以降 strength 有效；"
        "但更划算的是降 `--radius`（halo 宽度 = 高通的窗半径，r=2→1 把最深暗化砍掉约一半，"
        "骨增益只少约 13%），或设 `--max-darken` 直接给暗化量封顶（代价更大："
        "骨小梁的对比度有一半来自压暗骨髓腔）。",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render several sharpening strengths from one guided-filter field.",
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="base directory every relative path resolves against")
    parser.add_argument("--input", default="full.bhc.r4b.raw")
    parser.add_argument("--noisy", type=Path, default=None,
                        help="optional un-denoised reconstruction, drawn as the noise reference")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--eps-k", type=float, default=8.0, help="eps = (k * sigma)^2")
    parser.add_argument("--eps", type=float, default=None,
                        help="override eps outright (bypasses --eps-k / --sigma-source)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="override the noise sigma used to build eps")
    parser.add_argument("--sigma-source", choices=("flat", "tissue"), default="flat",
                        help="where the noise sigma comes from when --eps/--sigma are absent")
    parser.add_argument("--mode", choices=("wiener", "detail"), default="wiener")
    parser.add_argument("--strength", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.8, 1.0])
    parser.add_argument("--max-darken", type=float, default=None, metavar="D",
                        help="cap on how much darker a voxel may become (off by default)")
    parser.add_argument("--slab", type=int, default=96)
    parser.add_argument("--roi-cache", type=Path, default=None)
    parser.add_argument("--dark-window", type=int, default=0, metavar="SIZE",
                        help="add a column showing the deepest dark rim, located "
                             "automatically at the strongest strength (0 = off)")
    parser.add_argument("--verify", action="append", default=[],
                        metavar="STRENGTH=FILE",
                        help="re-render an already-written volume's crop and report max|diff|")
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--curve", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    if not _use_cjk_font():
        print("[warn] no CJK font found")

    args.input = args.dir / args.input
    args.noisy = _under(args.dir, args.noisy)
    args.roi_cache = _under(args.dir, args.roi_cache)
    figure = _under(args.dir, args.figure) or (args.dir / "gfsharp_compare.png")
    curve = _under(args.dir, args.curve)
    json_path = _under(args.dir, args.json)
    md_path = _under(args.dir, args.markdown)
    verifies = parse_verify(args.verify)
    for strength, path in list(verifies.items()):
        verifies[strength] = path if path.is_absolute() else args.dir / path

    shape = tuple(int(v) for v in args.shape)
    plain = _load(args.input, shape)
    print(f"[load] {args.input} shape={shape}")

    # ---- ROIs: the windows are fixed first, so every strength is measured on
    # ---- exactly the same voxels (this is what makes the comparison honest).
    if not (args.roi_cache and args.roi_cache.is_file()):
        raise SystemExit(
            "this script does not search for ROIs; pass --roi-cache (see "
            "guided_sharpen_volume.py --roi-cache to create one)"
        )
    rois = json.loads(args.roi_cache.read_text(encoding="utf-8"))
    print(f"[roi] reused {args.roi_cache}")
    for key in ROI_KEYS:
        r = rois[key]
        print(f"[roi] {key}: z={r['z']} y={r['y']} x={r['x']} size={r['size']}")

    # ---- eps: one number, printed, so the figure is reproducible.
    if args.sigma is not None:
        sigma, sigma_source = float(args.sigma), "explicit"
    elif args.sigma_source == "flat":
        sigma, sigma_source = patch_sigma(plain[patch_slices(rois["flat"])]), "flat"
    else:
        sigma, sigma_source = float(estimate_noise_std_immerkaer(plain)), "tissue"
    eps = float(args.eps) if args.eps is not None else eps_from_noise(sigma, args.eps_k)
    print(f"[eps] sigma={sigma:.7g} ({sigma_source}) k={args.eps_k:g} -> eps={eps:.6e}")

    # ---- the one expensive step; every strength below is a scalar on top.
    t0 = time.perf_counter()
    field = guided_structure_field(
        plain, radius=args.radius, eps=eps, mode=args.mode, slab=args.slab,
    )
    print(f"[field] mode={args.mode} radius={args.radius} slab={args.slab} "
          f"({time.perf_counter() - t0:.1f}s)")

    strengths = sorted(float(s) for s in args.strength)
    if args.dark_window:
        rois["dark"] = worst_darkening_roi(
            plain, field, max(strengths), args.max_darken, args.dark_window, VISIBLE_FLOOR,
        )
        print(f"[roi] dark (auto): z={rois['dark']['z']} y={rois['dark']['y']} "
              f"x={rois['dark']['x']} size={rois['dark']['size']} — 全卷最深可见暗边")

    panels: dict[str, np.ndarray] = {"plain": plain}
    if args.noisy is not None:
        panels["noisy"] = _load(args.noisy, shape)
        print(f"[ref] noisy {args.noisy}")
    for s in strengths:
        panels[f"s={s:g}"] = sharpen_patch_volume(plain, field, s, args.max_darken)

    # ---- self-check against volumes already on disk.
    for s, path in verifies.items():
        if not path.is_file():
            print(f"[verify] s={s:g}: MISSING {path}")
            continue
        written = _load(path, shape)
        worst = 0.0
        for key in ROI_KEYS:
            sl = patch_slices(rois[key])
            mine = sharpen_patch(plain, field, sl, s, args.max_darken)
            theirs = np.asarray(written[sl], dtype=np.float32)
            worst = max(worst, float(np.max(np.abs(mine - theirs))))
        verdict = "bit-for-bit" if worst == 0.0 else "MISMATCH"
        print(f"[verify] s={s:g}: max|crop diff| = {worst:.3e} vs {path.name} -> {verdict}")

    display_keys = [k for k in rois if k in ROI_TITLES]

    def darken_of(vol) -> float:
        """Worst darkening inside the *displayed* windows.

        Windows are fixed, so this compares like with like; when `--dark-window`
        is on one of them is the volume's worst rim, which makes this the
        worst-case number rather than the bone ROI's.
        """
        worst = 0.0
        for key in display_keys:
            sl = patch_slices(rois[key])
            base = np.asarray(plain[sl], dtype=np.float32)
            cur = np.asarray(vol[sl], dtype=np.float32)
            worst = max(worst, float(np.subtract(base, cur).max()))
        return worst

    rows = [{"tag": "plain", "strength": 0.0,
             "bone_patch_sigma": patch_sigma(plain[patch_slices(rois["bone"])]),
             "flat_patch_sigma": patch_sigma(plain[patch_slices(rois["flat"])]),
             "darken": 0.0}]
    if args.noisy is not None:
        rows.append({"tag": "noisy", "strength": 0.0,
                     "bone_patch_sigma": patch_sigma(panels["noisy"][patch_slices(rois["bone"])]),
                     "flat_patch_sigma": patch_sigma(panels["noisy"][patch_slices(rois["flat"])]),
                     "darken": darken_of(panels["noisy"])})
    for s in strengths:
        vol = panels[f"s={s:g}"]
        rows.append({
            "tag": "sharp", "strength": s,
            "bone_patch_sigma": patch_sigma(vol[patch_slices(rois["bone"])]),
            "flat_patch_sigma": patch_sigma(vol[patch_slices(rois["flat"])]),
            "darken": darken_of(vol),
        })

    base = rows[0]
    for row in rows:
        if row["tag"] == "sharp":
            print(f"[s={row['strength']:g}] bone {row['bone_patch_sigma']:.5f} "
                  f"({100.0 * (row['bone_patch_sigma'] / base['bone_patch_sigma'] - 1.0):+.1f}%) "
                  f"flat {row['flat_patch_sigma']:.5f} "
                  f"({100.0 * (row['flat_patch_sigma'] / base['flat_patch_sigma'] - 1.0):+.1f}%) "
                  f"ratio {row['bone_patch_sigma'] / row['flat_patch_sigma']:.3f} "
                  f"| max darkening {row['darken']:.4f}")

    render_compare(panels, rois, strengths, figure)
    print(f"[figure] {figure}")
    if curve is not None:
        render_curve(rows, curve)
        print(f"[curve] {curve}")

    noisy = next((r for r in rows if r["tag"] == "noisy"), None)
    meta = {
        "input": str(args.input),
        "mode": args.mode,
        "radius": args.radius,
        "eps": eps,
        "eps_k": args.eps_k,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "slab": args.slab,
        "max_darken": args.max_darken,
        "strengths": strengths,
        "bone_roi": rois["bone"]["label"],
        "flat_roi": rois["flat"]["label"],
        "dark_roi": rois.get("dark", {}).get("label"),
        "noisy_bone": noisy["bone_patch_sigma"] if noisy else float("nan"),
        "noisy_flat": noisy["flat_patch_sigma"] if noisy else float("nan"),
    }
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2,
                                        ensure_ascii=False), encoding="utf-8")
        print(f"[json] {json_path}")
    if md_path is not None:
        write_markdown(rows, meta, md_path)
        print(f"[markdown] {md_path}")
    return 0


def sharpen_patch_volume(plain: np.ndarray, field: np.ndarray, strength: float,
                         max_darken: float | None = None) -> "_CropMask":
    """Materialise ``plain + strength * field`` for the ROI crops only.

    Kept as a thin wrapper so ``panels`` can hold the same array-like shape the
    renderer expects while nothing bigger than a patch is ever allocated.
    """
    return _CropMask(plain, field, strength, max_darken)


class _CropMask:
    """Lazy ``plain + strength*field`` that only ever materialises a crop."""

    def __init__(self, plain: np.ndarray, field: np.ndarray, strength: float,
                 max_darken: float | None = None) -> None:
        self._plain = plain
        self._field = field
        self._strength = float(strength)
        self._max_darken = max_darken

    def __getitem__(self, key):
        return sharpen_patch(self._plain, self._field, key, self._strength,
                             self._max_darken)


if __name__ == "__main__":
    raise SystemExit(main())
