"""Conventional (non-AI) denoising vs the recursive AI rounds, on one volume.

Reads the volumes produced by ``scripts/recon_conventional_denoise.py`` and
``scripts/denoise_multi_models.py`` from a single folder and renders three figures plus a
metrics table:

* ``conv_vs_ai_orthogonal.png`` -- every method on the same axial / coronal / sagittal slice
* ``conv_vs_ai_roi.png`` -- the same methods on three auto-selected ROI crops, where
  texture loss is actually visible
* ``conv_vs_ai_metrics.png`` -- banded Immerkaer sigma, noise cut, and bone/soft ratio

Every panel shares a greyscale window taken from the *input* volume, so differences in the
picture are differences in the data, not in the display scaling.  The banded metrics reuse
the exact definition of ``denoise_multi_models.py`` (bands keyed on one reference volume,
tissue floor 0.01) so the numbers are directly comparable with an existing run summary.

Example::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/compare_conv_denoise.py \
      --dir outputs/air3_ypdl0027_bhc_5models \
      --input full.bhc.raw --conv full.bhc.conv_rn_ani.raw \
      --model r0=full.bhc.r0.raw --model r1b=full.bhc.r1b.raw \
      --model r2b=full.bhc.r2b.raw --model r3b=full.bhc.r3b.raw \
      --model r4b=full.bhc.r4b.raw --shape 360 640 640 --band-key r2b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.training.metrics import band_noise_std_immerkaer, estimate_noise_std_immerkaer

DTYPE = np.dtype("<f4")
TISSUE_FLOOR = 0.01  # below this a voxel is air and is excluded from every band

# Greyscale, then a distinct colour per method for the bar charts (input = neutral grey,
# conventional = orange, AI rounds = blue -> red ramp).
METHOD_COLORS = {
    "input": "#7f7f7f",
    "conv": "#ff7f0e",
    "r0": "#1f77b4",
    "r1b": "#2ca02c",
    "r2b": "#bcbd22",
    "r3b": "#d62728",
    "r4b": "#9467bd",
}

METHOD_LABELS = {
    "input": "输入 full.bhc（无降噪）",
    "conv": "常规降噪 ReduceNoise+Anisotropic",
    "r0": "r0 · AI 第 1 代",
    "r1b": "r1b · 方案B 第 1 轮",
    "r2b": "r2b · 朴素递归 第 2 轮",
    "r3b": "r3b · 朴素递归 第 3 轮",
    "r4b": "r4b · 朴素递归 第 4 轮",
}


def _use_cjk_font() -> bool:
    """Switch matplotlib to a font with CJK coverage when one is installed."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * DTYPE.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"{path}: {actual} B != {expected} B for shape {shape} float32 RAW")
    return np.asarray(np.memmap(path, dtype=DTYPE, mode="r", shape=shape))


def band_masks(key_volume: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    return {
        "low": (key_volume > TISSUE_FLOOR) & (key_volume <= lo),
        "mid": (key_volume > lo) & (key_volume < hi),
        "high": key_volume >= hi,
    }


def _metrics(
    volume: np.ndarray,
    masks: dict[str, np.ndarray],
    full: np.ndarray,
    full_global: float,
    tissue: np.ndarray,
    tissue_mean: float,
    max_slices: int,
) -> dict[str, float]:
    """Same metric definitions as ``denoise_multi_models.py`` (identical voxel sets)."""
    banded = {
        band: band_noise_std_immerkaer(volume, mask, max_slices=max_slices)
        for band, mask in masks.items()
    }
    global_sigma = estimate_noise_std_immerkaer(volume, max_slices=max_slices)
    tissue_n = int(np.count_nonzero(tissue))
    row = {
        "global_sigma": global_sigma,
        "bands": banded,
        "band_differentiation_high_over_low": (
            float(banded["high"] / banded["low"]) if banded["low"] else float("nan")
        ),
        "sigma_reduction_vs_input_pct": (
            float(100.0 * (1.0 - global_sigma / full_global)) if full_global else float("nan")
        ),
        "mean_offset_on_tissue": (
            float(np.sum(np.asarray(volume)[tissue], dtype=np.float64) / tissue_n - tissue_mean)
            if tissue_n else float("nan")
        ),
        "volume_min": float(volume.min()),
        "volume_max": float(volume.max()),
        "volume_mean": float(volume.mean()),
        "volume_std": float(volume.std()),
    }
    return row


# --------------------------------------------------------------------------- ROI selection


def _best_box(mask2d: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Place a ``size x size`` window on the 2D plane where ``mask2d`` has the most hits.

    A centroid is the wrong target for a U-shaped bone ring: the mean of the mandible
    cortex lands in the *air inside* the ring.  Scanning every window position with an
    integral image instead centres the crop on the structure itself.
    """
    height, width = mask2d.shape
    size = int(min(size, height, width))
    cumulative = np.zeros((height + 1, width + 1), dtype=np.int64)
    cumulative[1:, 1:] = mask2d.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    sums = (cumulative[size:, size:] - cumulative[:-size, size:]
            - cumulative[size:, :-size] + cumulative[:-size, :-size])
    row, column = np.unravel_index(int(np.argmax(sums)), sums.shape)
    row, column = int(row), int(column)
    return row, row + size, column, column + size


def select_rois(
    input_volume: np.ndarray,
    key_volume: np.ndarray,
    lo: float,
    hi: float,
    roi_size: int,
    bone_z: int | None = None,
    soft_z: int | None = None,
) -> list[dict]:
    """Pick three crops: two axial (bone-bearing / soft tissue) and one coronal.

    The bone slice maximises the high-density voxel count **inside the central 70 % box**
    of the y-x plane.  Counting over the whole plane picks the skull base instead, which
    sits right at the FOV rim; restricting to the centre lands on the dental arch /
    mandible body, i.e. the trabecular bone and tooth roots the figure is meant to judge.
    The soft slice maximises low-density tissue inside the middle of the z axis, so the
    crop is soft tissue rather than the neck at the bottom of the volume.
    """
    nz, ny, nx = input_volume.shape
    high = key_volume >= hi
    low = (input_volume > TISSUE_FLOOR) & (key_volume <= lo)

    if bone_z is None:
        margin_y, margin_x = int(ny * 0.15), int(nx * 0.15)
        central = np.zeros_like(high)
        central[:, margin_y:ny - margin_y, margin_x:nx - margin_x] = (
            high[:, margin_y:ny - margin_y, margin_x:nx - margin_x]
        )
        bone_z = int(np.argmax(central.sum(axis=(1, 2))))
    if soft_z is None:
        lo_z, hi_z = int(0.25 * nz), int(0.7 * nz)
        counts = low[lo_z:hi_z].sum(axis=(1, 2))
        soft_z = lo_z + int(np.argmax(counts)) if counts.size else nz // 2

    bone_y0, bone_y1, bone_x0, bone_x1 = _best_box(high[bone_z], roi_size)
    soft_y0, soft_y1, soft_x0, soft_x1 = _best_box(low[soft_z], roi_size)
    bone_cy = (bone_y0 + bone_y1) // 2
    cor_z0, cor_z1, cor_x0, cor_x1 = _best_box(high[:, bone_cy, :], roi_size)

    return [
        {"key": "axial-bone", "axis": 0, "index": bone_z, "label": f"轴位 骨/牙 ROI · z={bone_z}",
         "box": (bone_y0, bone_y1, bone_x0, bone_x1)},
        {"key": "axial-soft", "axis": 0, "index": soft_z, "label": f"轴位 软组织 ROI · z={soft_z}",
         "box": (soft_y0, soft_y1, soft_x0, soft_x1)},
        {"key": "coronal-bone", "axis": 1, "index": bone_cy,
         "label": f"冠状位 骨/牙 ROI · y={bone_cy}",
         "box": (cor_z0, cor_z1, cor_x0, cor_x1)},
    ]


def _roi_crop(volume: np.ndarray, roi: dict) -> np.ndarray:
    z0, z1, x0, x1 = roi["box"]
    if roi["axis"] == 0:
        return volume[roi["index"], z0:z1, x0:x1]
    return volume[z0:z1, roi["index"], x0:x1]


def _window(reference: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    finite = reference[np.isfinite(reference)]
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


# --------------------------------------------------------------------------- figures


def render_orthogonal(volumes, metrics, shape, out_path: Path, lo: float, hi: float, order) -> Path:
    nz, ny, nx = shape
    views = [
        (f"轴位 z={nz // 2}", lambda v: v[nz // 2]),
        (f"冠状位 y={ny // 2}", lambda v: v[:, ny // 2, :]),
        (f"矢状位 x={nx // 2}", lambda v: v[:, :, nx // 2]),
    ]
    vmin, vmax = _window(np.asarray(volumes["input"]))
    fig, axes = plt.subplots(len(order), len(views), figsize=(13.5, 3.2 * len(order)))
    fig.suptitle(
        f"常规降噪（ReduceNoise + Anisotropic）与 AI 递归去噪对照 —— {out_path.parent.parent.name}\n"
        f"同一层面、同一灰度窗（{vmin:.3f} ~ {vmax:.3f}），窗口取自输入体数据",
        fontsize=14,
    )
    for row, name in enumerate(order):
        row_metrics = metrics[name]
        label = METHOD_LABELS.get(name, name)
        if name != "input":
            label += (f"\nσ {row_metrics['global_sigma']:.5f}  "
                      f"({row_metrics['sigma_reduction_vs_input_pct']:+.1f}%)  "
                      f"diff {row_metrics['band_differentiation_high_over_low']:.3f}")
        else:
            label += f"\nσ {row_metrics['global_sigma']:.5f}"
        for col, (view_label, pick) in enumerate(views):
            ax = axes[row][col]
            ax.imshow(pick(np.asarray(volumes[name])), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(view_label, fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, labelpad=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def render_rois(volumes, rois, out_path: Path, order, roi_size: int) -> Path:
    input_volume = np.asarray(volumes["input"])
    fig, axes = plt.subplots(len(order), len(rois), figsize=(4.4 * len(rois), 3.3 * len(order)))
    fig.suptitle(
        f"ROI 放大对照（{roi_size}×{roi_size} 体素 = {roi_size * 0.25:.0f} mm，双侧同一窗宽）\n"
        "骨小梁 / 牙周膜纹理是否保留，看这一张",
        fontsize=14,
    )
    windows = [_window(_roi_crop(input_volume, roi)) for roi in rois]
    for row, name in enumerate(order):
        volume = np.asarray(volumes[name])
        for col, roi in enumerate(rois):
            vmin, vmax = windows[col]
            ax = axes[row][col]
            ax.imshow(_roi_crop(volume, roi), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(roi["label"], fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS.get(name, name), fontsize=10, labelpad=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return out_path


def render_metrics(metrics, out_path: Path, order) -> Path:
    names = [n for n in order]
    colors = [METHOD_COLORS.get(n, "#333333") for n in names]
    short = [METHOD_LABELS.get(n, n).split("（")[0].split(" · ")[0] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    bands = ["low", "mid", "high", "global"]
    band_labels = ["软组织带\n(μ≤0.2)", "过渡带", "骨带\n(μ≥0.4)", "全局"]
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = [metrics[name]["bands"][b] for b in ("low", "mid", "high")] + [
            metrics[name]["global_sigma"]
        ]
        positions = np.arange(len(bands)) + index * width - 0.4 + width / 2
        axes[0].bar(positions, values, width=width, color=colors[index],
                    label=short[index], edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(range(len(bands)))
    axes[0].set_xticklabels(band_labels)
    axes[0].set_ylabel("Immerkaer σ（高频能量）")
    axes[0].set_title("按密度带的高频残差：越低越平滑")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    cut = [metrics[n]["sigma_reduction_vs_input_pct"] for n in names]
    colors_cut = ["#bbbbbb" if n == "input" else ("#ff7f0e" if n == "conv" else "#1f77b4")
                  for n in names]
    bars = axes[1].bar(range(len(names)), cut, color=colors_cut)
    for rect, value in zip(bars, cut):
        axes[1].text(rect.get_x() + rect.get_width() / 2, value + 1.2, f"{value:.1f}",
                     ha="center", fontsize=9)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("全局 σ 削减 [%]")
    axes[1].set_title("相对输入的高频削减")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.3)

    diff = [metrics[n]["band_differentiation_high_over_low"] for n in names]
    bars = axes[2].bar(range(len(names)), diff, color=colors)
    for rect, value in zip(bars, diff):
        axes[2].text(rect.get_x() + rect.get_width() / 2, value + 0.012, f"{value:.3f}",
                     ha="center", fontsize=9)
    axes[2].set_ylim(0, max(diff) * 1.24)
    axes[2].axhline(1.0, color="#444444", linestyle="--", linewidth=1.1)
    axes[2].annotate("=1.0 骨与软组织同等对待（全局平滑）",
                     xy=(0.985, 1.035), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=8, color="#444444")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("diff = σ(骨带) / σ(软组织带)")
    axes[2].set_title("密度选择性：<1 表示对骨更狠、对软组织更温柔")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _cut(row: dict, band: str | None = None) -> float:
    """Percent reduction of one band (or of the global sigma) against the input.

    ``row['_input_bands']`` is injected by :func:`render_markdown`; the global value comes
    straight from the metrics so the two are always computed the same way.
    """
    if band is None:
        return float(row["sigma_reduction_vs_input_pct"])
    reference = row["_input_bands"][band]
    return float(100.0 * (1.0 - row["bands"][band] / reference)) if reference else float("nan")


def render_markdown(metrics, order, rois, out_path: Path, meta: dict) -> Path:
    for name in order:
        metrics[name]["_input_bands"] = metrics["input"]["bands"]
    conv = metrics[meta["conv_label"]]
    ranking = [n for n in order if n != "input"]
    most_adaptive = min(ranking, key=lambda n: metrics[n]["band_differentiation_high_over_low"])
    hardest = max(ranking, key=lambda n: metrics[n]["sigma_reduction_vs_input_pct"])
    diff_ordering = sorted(ranking, key=lambda n: metrics[n]["band_differentiation_high_over_low"])
    diff_summary = ", ".join(
        "{0} {1:.3f}".format(n, metrics[n]["band_differentiation_high_over_low"])
        for n in diff_ordering
    )

    lines = [
        "# 常规降噪（ReduceNoise + Anisotropic）与 AI 递归去噪对照",
        "",
        meta["note"],
        "",
        f"- 输入体数据：`{meta['input']}`（md5 `{meta.get('input_md5', 'n/a')}`）",
        f"- 常规降噪体积：`{meta['conv']}`（XDay 后处理链 `{meta['post_procs']}`，"
        f"pwls_scale={meta.get('pwls_scale')}）",
        f"- 密度带按 `{meta['band_key']}` 输出分带，`lo={meta['lo']}` / `hi={meta['hi']}`，"
        f"组织阈值 `>{TISSUE_FLOOR}`",
        "",
        "## 指标",
        "",
        "| 方法 | 软组织带 | 过渡带 | 骨带 | 全局 σ | 全局削减 | 软组织削减 | 骨带削减 | diff | 组织均值偏移 |",
        "| ---- | -------- | ------ | ---- | ------ | -------- | ---------- | -------- | ---- | ------------ |",
    ]
    for name in order:
        row = metrics[name]
        lines.append(
            f"| {METHOD_LABELS.get(name, name)} "
            f"| {row['bands']['low']:.5f} | {row['bands']['mid']:.5f} | {row['bands']['high']:.5f} "
            f"| {row['global_sigma']:.5f} | {_cut(row):+.1f}% "
            f"| {_cut(row, 'low'):+.1f}% | {_cut(row, 'high'):+.1f}% "
            f"| {row['band_differentiation_high_over_low']:.4f} "
            f"| {row['mean_offset_on_tissue']:+.2e} |"
        )
    lines += [
        "",
        "「削减」= 相对输入同一体素集合的 σ 下降百分比；正值越大压得越狠。",
        "「diff」= σ(骨带) / σ(软组织带)，读法见下。",
        "",
        "## ROI 位置（自动选取）",
        "",
    ]
    for roi in rois:
        lines.append(f"- `{roi['key']}`：{roi['label']}，裁剪框 `{roi['box']}`")
    lines += [
        "",
        "## 看图要点",
        "",
        "1. **三张图都是同一批体素**：灰度窗取自输入体数据（正交图）或输入体的同一 ROI（放大图），",
        "   所以更黑/更糊就是真的更黑/更糊，不是显示差异。",
        f"2. **常规降噪是“不做 AI 时的天花板”**：全局 σ {conv['global_sigma']:.5f}"
        f"（{_cut(conv):+.1f}%），量级上已经追平了递归链中段的方法。",
        f"   但它的削减是**反向分布**的——软组织带 {_cut(conv, 'low'):+.1f}%，骨带只有 "
        f"{_cut(conv, 'high'):+.1f}%，diff = {conv['band_differentiation_high_over_low']:.3f}。",
        "   ReduceNoise 是带梯度阈值的中值类滤波、Anisotropic 是各向异性平滑，两者都不按密度分流；",
        "   软组织里没有强梯度可保，于是被抹得更狠。",
        f"3. **diff 的读法**：<1 = 对软组织更温柔、对骨更狠（密度自适应）；=1 = 骨与软组织同等对待；",
        f"   >1 = 软组织被压得比骨还狠。本卷最“看密度”的是 **{most_adaptive}**"
        f"（{metrics[most_adaptive]['band_differentiation_high_over_low']:.3f}），"
        f"排序：{diff_summary}。",
        "4. **σ 不能区分噪声与纹理，骨带 σ 尤其不可信**。高密度带里的高频能量由高对比度界面主导",
        "   （金属牙、皮质骨边缘），中值类滤波会保住这些强边缘，于是常规降噪的骨带 σ 反而偏高——",
        "   但这**不代表**骨小梁纹理被保住了。判读必须落到 `conv_vs_ai_roi.png` 的骨小梁/牙周膜上。",
        f"5. **最激进的是 {hardest}**（全局削减 {metrics[hardest]['sigma_reduction_vs_input_pct']:+.1f}%）：",
        "   一幅 CBCT 的高频内容只剩两成左右，是否磨平必须肉眼确认。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- cli


def _parse_named(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Folder holding all volumes.")
    parser.add_argument("--input", default="full.bhc.raw", help="Undenoised reference volume.")
    parser.add_argument("--conv", default="full.bhc.conv_rn_ani.raw",
                        help="Conventional-denoising volume.")
    parser.add_argument("--model", type=_parse_named, action="append", default=None, metavar="NAME=PATH",
                        help="Repeatable; order defines the report order.")
    parser.add_argument("--conv-label", default="conv", help="Method key used for the conv volume.")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--band-key", default=None,
                        help="Method whose output keys the density bands (default: 2nd AI model).")
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.4)
    parser.add_argument("--max-slices", type=int, default=24)
    parser.add_argument("--roi-size", type=int, default=192)
    parser.add_argument("--bone-z", type=int, default=None, help="Override the axial bone ROI slice.")
    parser.add_argument("--soft-z", type=int, default=None, help="Override the axial soft ROI slice.")
    parser.add_argument("--figures", type=Path, default=None, help="defaults to <dir>/figures")
    parser.add_argument("--json", type=Path, default=None, help="defaults to <dir>/conv_vs_ai.json")
    parser.add_argument("--markdown", type=Path, default=None, help="defaults to <dir>/conv_vs_ai_summary.md")
    parser.add_argument("--post-procs", default="BHC ReduceNoise Anisotropic",
                        help="Chain recorded in the report for the conventional volume.")
    args = parser.parse_args()

    shape = tuple(int(v) for v in args.shape)
    models = args.model or [
        ("r0", Path("full.bhc.r0.raw")), ("r1b", Path("full.bhc.r1b.raw")),
        ("r2b", Path("full.bhc.r2b.raw")), ("r3b", Path("full.bhc.r3b.raw")),
        ("r4b", Path("full.bhc.r4b.raw")),
    ]

    order = ["input", args.conv_label] + [name for name, _ in models]
    paths = {"input": args.dir / args.input, args.conv_label: args.dir / args.conv}
    for name, path in models:
        paths[name] = path if path.is_absolute() else args.dir / path
    for name in order:
        if not paths[name].exists():
            raise SystemExit(f"missing volume for {name!r}: {paths[name]}")

    band_key = args.band_key or (models[1][0] if len(models) > 1 else order[-1])
    if band_key not in order:
        raise SystemExit(f"--band-key {band_key!r} is not one of {order}")

    use_cjk = _use_cjk_font()
    if not use_cjk:
        print("[warn] no CJK font found; falling back to the default font")

    volumes = {name: np.asarray(np.memmap(paths[name], dtype=DTYPE, mode="r", shape=shape))
               for name in order}
    key_volume = volumes[band_key]
    masks = band_masks(key_volume, args.lo, args.hi)
    band_n_voxels = {b: int(np.count_nonzero(m[1:-1, 1:-1])) for b, m in masks.items()}

    full = volumes["input"]
    full_global = estimate_noise_std_immerkaer(full, max_slices=args.max_slices)
    tissue = np.asarray(full) > TISSUE_FLOOR
    tissue_n = int(np.count_nonzero(tissue))
    tissue_mean = float(np.sum(np.asarray(full)[tissue], dtype=np.float64) / tissue_n)

    metrics = {}
    for name in order:
        metrics[name] = _metrics(volumes[name], masks, full, full_global, tissue,
                                 tissue_mean, args.max_slices)

    print(f"band key={band_key} lo={args.lo} hi={args.hi} tissue>{TISSUE_FLOOR}")
    header = "method".ljust(10) + "".join(b.rjust(11) for b in ("low", "mid", "high", "global", "cut%", "diff"))
    print(header)
    for name in order:
        row = metrics[name]
        print(f"{name:10s}"
              + f"{row['bands']['low']:11.5f}{row['bands']['mid']:11.5f}"
              + f"{row['bands']['high']:11.5f}{row['global_sigma']:11.5f}"
              + f"{row['sigma_reduction_vs_input_pct']:11.1f}"
              + f"{row['band_differentiation_high_over_low']:11.4f}")

    rois = select_rois(np.asarray(full), key_volume, args.lo, args.hi, args.roi_size,
                       bone_z=args.bone_z, soft_z=args.soft_z)
    for roi in rois:
        print(f"[roi] {roi['key']}: {roi['label']} box={roi['box']}")

    figures = args.figures or (args.dir / "figures")
    made = [
        render_orthogonal(volumes, metrics, shape, figures / "conv_vs_ai_orthogonal.png",
                          args.lo, args.hi, order),
        render_rois(volumes, rois, figures / "conv_vs_ai_roi.png", order, args.roi_size),
        render_metrics(metrics, figures / "conv_vs_ai_metrics.png", order),
    ]
    for path in made:
        print(f"[figure] {path}")

    payload = {
        "input": str(paths["input"]),
        "input_md5": _md5(paths["input"]),
        "conv": str(paths[args.conv_label]),
        "conv_md5": _md5(paths[args.conv_label]),
        "post_procs": args.post_procs,
        "band_key": band_key,
        "lo": args.lo,
        "hi": args.hi,
        "tissue_floor": TISSUE_FLOOR,
        "band_n_voxels": band_n_voxels,
        "rois": rois,
        "order": order,
        "metrics": metrics,
        "figures": [str(p) for p in made],
        "note": (
            "Immerkaer sigma measures high-frequency energy; it cannot tell noise from "
            "texture. 'diff' = sigma(high band)/sigma(low band): <1 means the method is "
            "gentler on soft tissue than on bone (density-adaptive), 1.0 means bone and "
            "soft tissue are smoothed equally hard."
        ),
    }
    json_path = args.json or (args.dir / "conv_vs_ai.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[json] {json_path}")

    markdown_path = args.markdown or (args.dir / "conv_vs_ai_summary.md")
    conv_sidecar = Path(str(paths[args.conv_label]) + ".json")
    conv_meta = {}
    if conv_sidecar.is_file():
        conv_meta = json.loads(conv_sidecar.read_text(encoding="utf-8"))
    render_markdown(metrics, order, rois, markdown_path, {
        "note": payload["note"],
        "input": paths["input"].name,
        "input_md5": payload["input_md5"],
        "conv": paths[args.conv_label].name,
        "conv_label": args.conv_label,
        "post_procs": args.post_procs,
        "pwls_scale": (conv_meta.get("pwls") or {}).get("scale"),
        "band_key": band_key,
        "lo": args.lo,
        "hi": args.hi,
    })
    print(f"[markdown] {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
