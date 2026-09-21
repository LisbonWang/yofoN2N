"""Strength / radius / eps sweep for the 3D guided-filter sharpening.

The sharpened volume is ``volume + strength * field`` and ``field`` does not
depend on ``strength`` (both modes are linear in it), so the guided filter -- the
only expensive part -- is computed once per ``(radius, eps)`` and every strength
is then just a different scalar on the same field.  Metrics are evaluated on an
evenly spaced subset of slices, exactly the slices the project's Immerkaer
estimator uses anyway.

The point of the sweep is the trade-off: how much high-frequency energy comes
back in trabecular bone versus how much is added in a patch of tissue that is
provably flat.  A sharpener is only useful if the second number stays small.

Example::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/guided_sharpen_sweep.py \
      --dir outputs/air3_ypdl0027_bhc_5models --input full.bhc.r4b.raw \
      --shape 360 640 640 --noisy full.bhc.raw \
      --radius 2 4 6 --eps-k 1 2 4 8 --strength 0.25 0.5 0.75 1 1.5 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from cbct_n2n.filters.guided import eps_from_noise, guided_structure_field
from cbct_n2n.training.metrics import _subsample_slices, estimate_noise_std_immerkaer

from scripts.guided_sharpen_volume import (  # noqa: E402  (needs sys.path above)
    TISSUE_FLOOR,
    _load,
    _under,
    _use_cjk_font,
    band_masks,
    band_noise_std_immerkaer,
    octave_bands,
    patch_sigma,
    patch_slices,
    pick_rois,
)


def _slice_metrics(volume: np.ndarray, masks: dict[str, np.ndarray], tissue: np.ndarray,
                   flat_patch: tuple, bone_patch: tuple) -> dict:
    """Metrics on the z-subsampled stack (``masks``/``tissue`` are already sliced)."""
    bands = {b: band_noise_std_immerkaer(volume, m, max_slices=volume.shape[0])
             for b, m in masks.items()}
    flat = np.asarray(volume[flat_patch], dtype=np.float32)
    bone = np.asarray(volume[bone_patch], dtype=np.float32)
    tissue_values = volume[tissue]
    return {
        "bands": bands,
        "global_sigma": estimate_noise_std_immerkaer(volume, max_slices=volume.shape[0]),
        "band_differentiation": (float(bands["high"] / bands["low"]) if bands["low"]
                                 else float("nan")),
        "flat_sigma": patch_sigma(flat),
        "bone_sigma": patch_sigma(bone),
        "flat_octaves": octave_bands(flat),
        "bone_octaves": octave_bands(bone),
        "tissue_mean": float(np.mean(tissue_values, dtype=np.float64)) if tissue_values.size
        else float("nan"),
        "vmin": float(volume.min()),
        "vmax": float(volume.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--input", default="full.bhc.r4b.raw")
    parser.add_argument("--noisy", default=None,
                        help="Undenoised volume: gives the 'how much HF was ever there' bound.")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--mode", default="wiener", choices=("wiener", "detail"))
    parser.add_argument("--radius", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--eps-k", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--strength", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--sigma", type=float, default=None,
                        help="Noise sigma for eps (default: measured, see --sigma-source).")
    parser.add_argument("--sigma-source", choices=("flat", "tissue"), default="flat",
                        help="See guided_sharpen_volume.py: 'flat' makes k read as "
                             "'k x the residual noise floor'.")
    parser.add_argument("--max-slices", type=int, default=24,
                        help="Slices used for every metric.")
    parser.add_argument("--slab", type=int, default=96)
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.4)
    parser.add_argument("--flat-roi-size", type=int, default=128)
    parser.add_argument("--bone-roi-size", type=int, default=80)
    parser.add_argument("--roi-cache", type=Path, default=None,
                        help="JSON with flat/bone ROI picks; written on first run and reused, "
                             "so every sweep point measures exactly the same voxels.")
    parser.add_argument("--max-flat-cost", type=float, default=5.0,
                        help="Allowed relative rise of the flat-patch sigma, in percent. "
                             "The recommended row is the strongest bone gain inside it.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    if not _use_cjk_font():
        print("[warn] no CJK font found")
    shape = tuple(int(v) for v in args.shape)
    # Path flags are relative to --dir unless absolute; see the twin helper in
    # scripts/guided_sharpen_volume.py.  Without this, a bare --json/--figure
    # landed in the process CWD instead of the data folder.
    args.roi_cache = _under(args.dir, args.roi_cache)
    args.json = _under(args.dir, args.json)
    args.figure = _under(args.dir, args.figure)
    args.markdown = _under(args.dir, args.markdown)
    volume = _load(args.dir / args.input, shape)
    sigma_tissue = estimate_noise_std_immerkaer(volume, max_slices=args.max_slices)
    print(f"[cfg] mode={args.mode} radii={args.radius} eps_k={args.eps_k} "
          f"strengths={args.strength} slices={args.max_slices}")

    # ------------------------------------------------------------------ reference
    idx = _subsample_slices(np.arange(shape[0]), args.max_slices)
    base = np.asarray(volume[idx], dtype=np.float32)
    masks = band_masks(np.asarray(volume), args.lo, args.hi)
    masks = {b: m[idx] for b, m in masks.items()}
    tissue = np.asarray(volume) > TISSUE_FLOOR
    tissue = tissue[idx]

    roi_cache = args.roi_cache or (args.dir / "gfsharp_rois.json")
    if roi_cache.is_file():
        rois = json.loads(roi_cache.read_text(encoding="utf-8"))
        print(f"[roi] reused {roi_cache}")
    else:
        rois = pick_rois(np.asarray(volume), args.lo, args.hi,
                         flat_size=args.flat_roi_size, bone_size=args.bone_roi_size,
                         pool=[int(z) for z in idx])
        roi_cache.write_text(json.dumps(rois, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[roi] picked and cached to {roi_cache}")
    # The ROIs were picked from the very slices the metrics use, so the index is exact.
    patch = {}
    for key, roi in rois.items():
        half = roi["size"] // 2
        z_index = int(np.argmin(np.abs(idx - roi["z"])))
        if int(idx[z_index]) != int(roi["z"]):
            raise SystemExit(f"ROI {key} sits on z={roi['z']}, which is not a measured slice; "
                             f"delete {roi_cache} and re-run")
        patch[key] = (z_index, slice(roi["y"] - half, roi["y"] - half + roi["size"]),
                      slice(roi["x"] - half, roi["x"] - half + roi["size"]))
        print(f"[roi] {key}: z={roi['z']} (slice #{z_index}) y={roi['y']} x={roi['x']} "
              f"size={roi['size']}")

    half = rois["flat"]["size"] // 2
    sigma_flat = patch_sigma(np.asarray(volume[patch_slices(rois["flat"])], dtype=np.float32))
    sigma = args.sigma if args.sigma is not None else (
        sigma_flat if args.sigma_source == "flat" else sigma_tissue)
    print(f"[noise] sigma_tissue={sigma_tissue:.6f} sigma_flat={sigma_flat:.6f} "
          f"-> using sigma={sigma:.6f} (--sigma-source {args.sigma_source})")

    rows: list[dict] = []

    def record(tag: str, radius, eps, eps_k, strength, stack) -> None:
        row = _slice_metrics(stack, masks, tissue, patch["flat"], patch["bone"])
        row.update({"tag": tag, "mode": args.mode, "radius": radius, "eps": eps,
                    "eps_k": eps_k, "strength": strength})
        rows.append(row)

    record("plain", None, None, None, 0.0, base)
    if args.noisy:
        noisy = np.asarray(_load(args.dir / args.noisy, shape)[idx], dtype=np.float32)
        record("noisy", None, None, None, 0.0, noisy)
        del noisy
    print(f"[ref] plain  global_sigma={rows[0]['global_sigma']:.5f} "
          f"flat={rows[0]['flat_sigma']:.5f} bone={rows[0]['bone_sigma']:.5f}")

    # ------------------------------------------------------------------ sweep
    combinations = [(r, k) for r in args.radius for k in args.eps_k]
    for radius, eps_k in combinations:
        eps = eps_from_noise(sigma, eps_k)
        started = time.perf_counter()
        field = guided_structure_field(volume, radius=radius, eps=eps, mode=args.mode,
                                       slab=args.slab)
        field_sub = np.asarray(field[idx], dtype=np.float32)
        del field
        for strength in args.strength:
            stack = np.add(base, np.float32(strength) * field_sub).astype(np.float32)
            record(f"r{radius}_k{eps_k}_s{strength}", radius, eps, eps_k, strength, stack)
        elapsed = time.perf_counter() - started
        print(f"[sweep] radius={radius} eps_k={eps_k} eps={eps:.3e} "
              f"({elapsed:.1f}s for {len(args.strength)} strengths)")

    # ------------------------------------------------------------------ outputs
    def find(tag: str) -> dict:
        return next(r for r in rows if r["tag"] == tag)

    plain = find("plain")
    best = None
    for row in rows:
        if row["tag"] in ("plain", "noisy"):
            continue
        # The operating point is chosen by a budget, not by a ratio: allow the
        # soft-tissue noise floor to rise by at most --max-flat-cost percent, then
        # take the strongest bone gain that stays inside that budget.
        gain = row["bone_sigma"] / plain["bone_sigma"] - 1.0
        cost = 100 * (row["flat_sigma"] / plain["flat_sigma"] - 1.0)
        if cost <= args.max_flat_cost and (best is None or gain > best[0]):
            best = (gain, row)

    json_path = args.json or (args.dir / "gfsharp_sweep.json")
    json_path.write_text(json.dumps({
        "input": str(args.dir / args.input),
        "noisy": args.noisy,
        "mode": args.mode,
        "sigma_for_eps": sigma,
        "sigma_source": args.sigma_source if args.sigma is None else "explicit",
        "sigma_tissue": sigma_tissue,
        "sigma_flat_patch": sigma_flat,
        "slices": [int(z) for z in idx],
        "rois": rois,
        "patches_on_subsampled_stack": {
            key: {"slice_index": int(p[0]), "z": int(idx[p[0]]),
                  "y": [int(p[1].start), int(p[1].stop)],
                  "x": [int(p[2].start), int(p[2].stop)]}
            for key, p in patch.items()
        },
        "rows": rows,
        "recommended": best[1]["tag"] if best else None,
        "max_flat_cost_pct": args.max_flat_cost,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[json] {json_path}")

    figure_path = args.figure or (args.dir / "figures_gf" / "gfsharp_sweep.png")
    render_sweep(rows, plain, figure_path, "noisy" in {r["tag"] for r in rows})
    print(f"[figure] {figure_path}")

    if best is not None:
        print(f"[recommend] {best[1]['tag']}: bone {plain['bone_sigma']:.5f} -> "
              f"{best[1]['bone_sigma']:.5f} ({100 * best[0]:+.0f}%), flat "
              f"{plain['flat_sigma']:.5f} -> {best[1]['flat_sigma']:.5f} "
              f"({100 * (best[1]['flat_sigma'] / plain['flat_sigma'] - 1):+.0f}%)")
    md_path = args.markdown or (args.dir / "gfsharp_sweep.md")
    write_markdown(rows, plain, best, args, md_path,
                   {"tissue": sigma_tissue, "flat": sigma_flat, "used": sigma})
    print(f"[markdown] {md_path}")
    return 0


def render_sweep(rows: list[dict], plain: dict, out_path: Path, has_noisy: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    points = [r for r in rows if r["tag"] != "plain" and r["tag"] != "noisy"]
    for row in points:
        d_flat = 100 * (row["flat_sigma"] / plain["flat_sigma"] - 1)
        d_bone = 100 * (row["bone_sigma"] / plain["bone_sigma"] - 1)
        color = plt.cm.viridis((row["radius"] - 1) / 8.0)
        axes[0].plot(d_flat, d_bone, "o", color=color, markersize=6)
        axes[0].annotate(f"{row['radius']}/{row['eps_k']:g}/{row['strength']:g}",
                         (d_flat, d_bone), fontsize=6, alpha=0.75,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].axvline(0, color="gray", lw=0.8)
    axis_max = max(5.0, max(100 * (r["bone_sigma"] / plain["bone_sigma"] - 1) for r in points))
    axes[0].plot([0, axis_max / 4], [0, axis_max], "--", color="#888888", lw=1.0,
                 label="代价 = 收益/4 参考线")
    axes[0].set_xlabel("平坦软组织 ROI 的 σ 增加 [%]（噪声/软组织代价）")
    axes[0].set_ylabel("骨小梁 ROI 的 σ 增加 [%]（纹理回收）")
    axes[0].set_title("权衡：点越靠左上越好\n（标注 = radius/eps_k/strength，颜色 = radius）")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for axis, band in zip(axes[1:], ("mid", "fine")):
        for radius in sorted({r["radius"] for r in points}):
            subset = [r for r in points if r["radius"] == radius and r["eps_k"] == 2.0]
            if not subset:
                continue
            strengths = [r["strength"] for r in subset]
            base_bone = plain["bone_octaves"][band]
            axis.plot(strengths, [100 * (r["bone_octaves"][band] / base_bone - 1) for r in subset],
                      "-o", label=f"r={radius} 骨 ROI", lw=1.4, markersize=4)
            base_flat = plain["flat_octaves"][band]
            axis.plot(strengths, [100 * (r["flat_octaves"][band] / base_flat - 1) for r in subset],
                      "--s", label=f"r={radius} 平坦 ROI", lw=1.2, markersize=3.5, alpha=0.7)
        axis.set_xlabel("strength")
        axis.set_ylabel(f"{'mid（骨小梁尺度）' if band == 'mid' else 'fine（1 体素尺度）'} σ 变化 [%]")
        axis.set_title(f"{'mid' if band == 'mid' else 'fine'} 分频段："
                       f"{'想抬的是这条' if band == 'mid' else '最不想抬的是这条'}")
        axis.axhline(0, color="gray", lw=0.8)
        axis.grid(alpha=0.3)
        if band == "mid":
            axis.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_markdown(rows, plain, best, args, out_path: Path, sigmas: dict) -> None:
    def line(row) -> str:
        gain = 100 * (row["bone_sigma"] / plain["bone_sigma"] - 1)
        cost = 100 * (row["flat_sigma"] / plain["flat_sigma"] - 1)
        ratio = row["bone_sigma"] / row["flat_sigma"]
        return (f"| {row['tag']} | {row['radius'] or '-'} | {row['eps_k'] or '-'} "
                f"| {row['strength']:.2f} | {row['global_sigma']:.5f} | {row['flat_sigma']:.5f} "
                f"| {row['bone_sigma']:.5f} | {gain:+.1f}% | {cost:+.1f}% "
                f"| {gain / cost if cost else float('inf'):.2f} | {ratio:.3f} |")

    lines = [
        "# 3D 引导滤波锐化 —— 参数扫描",
        "",
        f"- 输入 `{args.input}`，mode `{args.mode}`，eps = (k·σ_noise)²",
        f"- σ_noise 来源 `--sigma-source {args.sigma_source}`"
        + ("（显式给定）" if args.sigma is not None else "")
        + f"：全组织实测 {sigmas['tissue']:.6f}，平坦 ROI 实测 {sigmas['flat']:.6f}，"
        f"本次用 **{sigmas['used']:.6f}**",
        f"- 扫描：radius {args.radius} × eps_k {args.eps_k} × strength {args.strength}",
        f"- 每点只在 {args.max_slices} 个等距层面上评估指标（与 Immerkaer 估计器同一批层面）",
        "",
        "| 配置 | radius | eps_k | strength | 全局 σ | 平坦 σ | 骨 σ | 骨增益 | 平坦代价 "
        "| 增益/代价 | 骨/平 |",
        "| ---- | ------ | ----- | -------- | ------ | ------ | ---- | ------ | -------- "
        "| --------- | ----- |",
    ]
    lines += [line(r) for r in rows]
    lines += [
        "",
        "「骨增益」/「平坦代价」= 相对锐化前的 σ 变化百分比；增益/代价 = 两者的比值，越大越好。",
        "",
        f"**推荐**：{best[1]['tag'] if best else 'n/a'}"
        + (f"（骨增益 {100 * (best[1]['bone_sigma'] / plain['bone_sigma'] - 1):+.1f}%，"
           f"平坦代价 {100 * (best[1]['flat_sigma'] / plain['flat_sigma'] - 1):+.1f}%）"
           if best else ""),
        "",
        "选择规则：在平坦 ROI σ 涨幅 ≤ "
        f"{args.max_flat_cost:g}% 的配置里取骨 ROI 涨幅最大者"
        "（`--max-flat-cost` 可调）。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
