"""3D guided-filter sharpening of one CBCT volume, with figures and metrics.

Reads a raw float32 volume (default: the over-smoothed ``r4b`` of the recursive
chain), sharpens it with the noise-aware guided filter of
``cbct_n2n.filters.guided``, and writes:

* the sharpened RAW volume plus a ``.json`` sidecar,
* ``gfsharp_orthogonal.png`` -- input / unsharpened / sharpened on the same
  axial, coronal and sagittal slice, one greyscale window taken from the input;
* ``gfsharp_roi.png`` -- the same volumes on a flat soft-tissue patch and on a
  trabecular-bone patch, so noise addition and texture recovery can be told
  apart by eye;
* ``gfsharp_metrics.png`` -- banded Immerkaer sigma, per-octave energy in the
  two patches, and the (noise added) vs (structure restored) trade-off;
* ``gfsharp_summary.md`` + ``gfsharp.json`` -- the numbers behind the figures.

The interesting quantity is not the global sigma (it cannot tell noise from
texture) but the *ratio* between what the sharpening does to a patch that is
provably flat and to a patch that provably carries bone texture.

Example::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/guided_sharpen_volume.py \
      --dir outputs/air3_ypdl0027_bhc_5models --input full.bhc.r4b.raw \
      --shape 360 640 640 --radius 4 --eps-k 2 --strength 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from scipy.signal import convolve2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.filters.guided import (
    MODES,
    cap_darkening,
    eps_from_noise,
    guided_structure_field,
)
from cbct_n2n.training.metrics import band_noise_std_immerkaer, estimate_noise_std_immerkaer

DTYPE = np.dtype("<f4")
TISSUE_FLOOR = 0.01
#: A voxel above this is unambiguously tissue in any reasonable display window
#: (flat soft tissue measures ~0.22 here), so a darkening measured on this set is
#: a visible dark rim rather than an already-black voxel being nudged.
VISIBLE_FLOOR = 0.15
_LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

METHOD_COLORS = {
    "noisy": "#7f7f7f",
    "r3b": "#d62728",
    "plain": "#1f77b4",
    "sharp": "#e6550d",
}
METHOD_LABELS = {
    "noisy": "未去噪重建 full.bhc（高频上限参考）",
    "plain": "锐化前 · 过平滑输入",
    "r3b": "r3b · 链中段（参考）",
    "sharp": "引导滤波锐化后",
}


# --------------------------------------------------------------------------- io


def _use_cjk_font() -> bool:
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
    """Memory-mapped view of a raw float32 volume, size-checked against ``shape``."""
    expected = int(np.prod(shape)) * DTYPE.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"{path}: {actual} B != {expected} B for shape {shape} float32 RAW")
    return np.asarray(np.memmap(path, dtype=DTYPE, mode="r", shape=shape))


def _under(base: Path, value: Path | None) -> Path | None:
    """Resolve a path flag against ``base`` (``--dir``) unless it is absolute.

    ``None`` (the flag was not given) is passed through so the per-artifact
    defaults still apply.
    """
    if value is None:
        return None
    return value if value.is_absolute() else base / value


def _write_raw(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(volume, dtype=DTYPE).tofile(path)


# --------------------------------------------------------------------------- metrics


def band_masks(key_volume: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    """Density bands keyed on ``key_volume`` (identical to the project's definition)."""
    return {
        "low": (key_volume > TISSUE_FLOOR) & (key_volume <= lo),
        "mid": (key_volume > lo) & (key_volume < hi),
        "high": key_volume >= hi,
    }


def patch_sigma(patch: np.ndarray) -> float:
    """Immerkaer sigma of a 2D patch (same kernel and normalisation as the 3D version)."""
    patch = np.asarray(patch, dtype=np.float32)
    lap = convolve2d(patch, _LAPLACIAN, mode="valid")
    interior = patch[1:-1, 1:-1]
    keep = lap[interior > TISSUE_FLOOR]
    if keep.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(keep**2) / (36.0 * keep.size)))


def octave_bands(patch: np.ndarray, radii: tuple[int, ...] = (2, 6, 16)) -> dict[str, float]:
    """Sigma of the difference-of-box octave bands of a 2D patch.

    ``fine`` ~ voxel-scale (and therefore mostly noise), ``mid`` ~ trabecular
    scale, ``coarse`` ~ the slowly varying background.  Which band gains energy
    is the whole question for a sharpener: restoring mid-frequency structure is
    the point, pumping up ``fine`` is not.
    """
    means = {0: np.asarray(patch, dtype=np.float32)}
    for r in radii:
        means[r] = uniform_filter(means[0], size=2 * r + 1, mode="nearest")
    chain = [0, *radii]
    out: dict[str, float] = {}
    names = ["fine", "mid", "coarse"]
    for index, name in enumerate(names):
        out[name] = float((means[chain[index]] - means[chain[index + 1]]).std())
    out["dc"] = float(means[radii[-1]].std())
    return out


def hf_correlation(a: np.ndarray, b: np.ndarray, radius: int = 2) -> float:
    """Pearson correlation of the high-pass content of two patches.

    Kept as a diagnostic, **not** as evidence of restored structure: it was tried
    for that and does not work.  ``b`` is a reconstruction the sharpened volume
    is itself derived from, so their noise is not independent, and every round of
    the chain turns out to be an almost linear high-frequency attenuator (in flat
    tissue the correlation with the input's high pass stays at 0.99 from r0 to
    r4b).  A high value therefore means "scaled copy of the input", not "signal
    rather than noise".
    """
    ha = np.asarray(a, dtype=np.float32) - uniform_filter(a, size=2 * radius + 1, mode="nearest")
    hb = np.asarray(b, dtype=np.float32) - uniform_filter(b, size=2 * radius + 1, mode="nearest")
    if ha.std() == 0 or hb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ha.ravel(), hb.ravel())[0, 1])


def overshoot_stats(volume: np.ndarray, tissue: np.ndarray,
                    reference: np.ndarray | None = None) -> dict:
    """Overshoot / undershoot QC for a sharpened volume.

    ``sigma`` cannot see a filter that leaves the bulk of the volume alone while
    driving a few thousand voxels at the strongest edges far out of range, so
    this is measured separately.  It caught something real here: at intensity
    min/max the range widens sharply with strength, and the low tail lives in
    **air**, not tissue.

    Split by region on purpose.  The reconstruction clamps tissue at
    ``tissue_floor``, so tissue can only ever be pushed *up*; everything below
    zero is air, and air undershoot is ringing at a strong edge rather than a
    change in any structure.  Reporting the two together would hide that.

    ``reference`` adds the *darkening* measurements, and it is the only way to
    see the defect the user actually reports ("黑边").  ``tissue_below_0_pct``
    cannot: a voxel sitting at ``tissue_floor`` only has to fall 0.01 to be
    counted, so the metric is dominated by voxels that were already black in any
    display window.  ``darken_visible_*`` instead asks how far a voxel that is
    **unambiguously tissue** (``> VISIBLE_FLOOR``) was driven down, which is what
    reads as a dark rim.  Measured on this volume: ``tissue_below_0_pct`` moved
    0.82% -> 0.69% while the visible darkening stayed 0.53, i.e. the first metric
    was blind to the thing being fixed.
    """
    vol = np.asarray(volume, dtype=np.float32)
    tis = np.asarray(tissue, dtype=bool)
    out = {
        "all_min": float(vol.min()),
        "all_max": float(vol.max()),
        "all_below_0_pct": float(100.0 * np.count_nonzero(vol < 0.0) / vol.size),
        "all_above_1_pct": float(100.0 * np.count_nonzero(vol > 1.0) / vol.size),
    }
    if np.count_nonzero(tis):
        t = vol[tis]
        out["tissue_n"] = int(t.size)
        out["tissue_min"] = float(t.min())
        out["tissue_below_0_pct"] = float(100.0 * np.count_nonzero(t < 0.0) / t.size)
        out["tissue_p9999"] = float(np.percentile(t, 99.99))
    else:
        # Key stays present so a consumer never has to guess whether "no
        # undershoot" means 0 or "not measured".
        out["tissue_n"] = 0
        out["tissue_min"] = float("nan")
        out["tissue_below_0_pct"] = float("nan")
        out["tissue_p9999"] = float("nan")

    if reference is None:
        out["darken_max"] = float("nan")
        out["darken_visible_max"] = float("nan")
        out["darken_visible_p999"] = float("nan")
        return out

    ref = np.asarray(reference, dtype=np.float32)
    darken = np.subtract(ref, vol)
    np.maximum(darken, 0.0, out=darken)
    out["darken_max"] = float(darken[tis].max()) if np.count_nonzero(tis) else float("nan")
    visible = tis & (ref > VISIBLE_FLOOR)
    if np.count_nonzero(visible):
        d = darken[visible]
        out["darken_visible_max"] = float(d.max())
        out["darken_visible_p999"] = float(np.percentile(d, 99.9))
    else:
        out["darken_visible_max"] = float("nan")
        out["darken_visible_p999"] = float("nan")
    return out


def _metrics(
    volume: np.ndarray,
    masks: dict[str, np.ndarray],
    tissue: np.ndarray,
    tissue_mean: float,
    input_global: float,
    flat_patch: tuple,
    bone_patch: tuple,
    max_slices: int,
    noisy_patches: dict[str, np.ndarray] | None = None,
    reference: np.ndarray | None = None,
) -> dict:
    banded = {
        band: band_noise_std_immerkaer(volume, mask, max_slices=max_slices)
        for band, mask in masks.items()
    }
    global_sigma = estimate_noise_std_immerkaer(volume, max_slices=max_slices)
    tissue_n = int(np.count_nonzero(tissue))
    flat = np.asarray(volume[flat_patch], dtype=np.float32)
    bone = np.asarray(volume[bone_patch], dtype=np.float32)
    return {
        "global_sigma": global_sigma,
        "bands": banded,
        "band_differentiation_high_over_low": (
            float(banded["high"] / banded["low"]) if banded["low"] else float("nan")
        ),
        "sigma_change_vs_input_pct": float(100.0 * (global_sigma / input_global - 1.0)),
        "mean_offset_on_tissue": (
            float(np.sum(volume[tissue], dtype=np.float64) / tissue_n - tissue_mean)
            if tissue_n else float("nan")
        ),
        "flat_patch_sigma": patch_sigma(flat),
        "bone_patch_sigma": patch_sigma(bone),
        "flat_patch_octaves": octave_bands(flat),
        "bone_patch_octaves": octave_bands(bone),
        "hf_correlation_vs_noisy": (
            {key: hf_correlation(flat if key == "flat" else bone, patch)
             for key, patch in noisy_patches.items()}
            if noisy_patches else None
        ),
        "volume_min": float(volume.min()),
        "volume_max": float(volume.max()),
        "volume_mean": float(volume.mean()),
        "volume_std": float(volume.std()),
        "overshoot": overshoot_stats(volume, tissue, reference),
    }


# --------------------------------------------------------------------------- roi picking


def filter_validity(plane: np.ndarray, size: int, want: str, margin: float = 0.15):
    """Window statistics and the validity mask for one 2D plane.

    Two constraints do the real work, because "most local variance in the
    volume" otherwise lands on the body/air boundary or on a metal filling:

    * **every** voxel of the window must be tissue (no air, no body outline),
    * no voxel may exceed the metal threshold, so enamel fillings are excluded.

    Windows outside the central ``1 - 2*margin`` box are rejected as well; that
    keeps the crop away from the reconstruction's FOV rim, where the intensity
    fades and any measured sigma is meaningless.
    """
    plane = np.asarray(plane, dtype=np.float32)
    ny, nx = plane.shape
    inner = max(3, size // 6)
    local_mean = uniform_filter(plane, size=2 * inner + 1, mode="nearest")
    local_var = uniform_filter(plane * plane, size=2 * inner + 1, mode="nearest") - local_mean**2
    score = uniform_filter(np.sqrt(np.maximum(local_var, 0.0)), size=size, mode="nearest")
    mean_win = uniform_filter(local_mean, size=size, mode="nearest")
    lo_win = minimum_filter(plane, size=size, mode="nearest")
    hi_win = maximum_filter(plane, size=size, mode="nearest")

    box = np.zeros_like(score, dtype=bool)
    box[int(margin * ny):ny - int(margin * ny), int(margin * nx):nx - int(margin * nx)] = True
    if want == "flat":
        # soft tissue throughout, bone nowhere
        valid = box & (lo_win > 0.03) & (hi_win < 0.40) & (mean_win > 0.04) & (mean_win < 0.25)
    elif want == "texture":
        # dense tissue throughout; trabecular marrow can dip, enamel cannot top
        valid = box & (lo_win > 0.10) & (hi_win < 1.6) & (mean_win > 0.35)
    else:
        raise ValueError(f"want must be flat|texture, got {want!r}")
    return valid, score, mean_win


def pick_roi(volume: np.ndarray, *, want: str, size: int, slices: list[int], title: str) -> dict:
    """Best ``size x size`` patch over a list of candidate slices.

    A single pre-chosen slice is not good enough: trabecular bone occupies only
    part of a mandible cross-section (no 24 mm window survives anywhere at the
    ramus), while the flattest soft tissue is not on the slice that has the most
    soft tissue.  So every candidate slice is scored and the best patch wins.
    """
    best: tuple[float, int, int, int] | None = None
    for z in slices:
        valid, score, mean_win = filter_validity(volume[z], size, want)
        if not valid.any():
            continue
        filled = np.where(valid, score, np.inf if want == "flat" else -np.inf)
        index = int(np.argmin(filled)) if want == "flat" else int(np.argmax(filled))
        row, col = np.unravel_index(index, score.shape)
        value = float(score[row, col])
        better = (best is None or (value < best[0] if want == "flat" else value > best[0]))
        if better:
            best = (value, int(z), int(row), int(col), float(mean_win[row, col]))
    if best is None:
        raise SystemExit(
            f"no valid {size}x{size} {want} window on any of {len(slices)} candidate slices; "
            f"lower --{'bone' if want == 'texture' else 'flat'}-roi-size"
        )
    score_value, z, row, col, mean_value = best
    return {
        "z": z, "y": row, "x": col, "size": size,
        "local_std": score_value, "window_mean": mean_value,
        "label": f"{title} · z={z} y={row} x={col} · {size}×{size}",
    }


def pick_rois(volume: np.ndarray, lo: float, hi: float, *, flat_size: int, bone_size: int,
              n_slices: int = 14, pool: list[int] | None = None) -> dict[str, dict]:
    """Auto-select the flat-tissue and trabecular-bone patches.

    ``pool`` restricts the candidate slices.  Pass the very slices the metrics
    will be evaluated on (see ``guided_sharpen_sweep.py``) and the patch is then
    guaranteed to sit on a slice that is actually measured, instead of being
    rounded onto a neighbour 5 slices away.
    """
    key = np.asarray(volume)
    nz = key.shape[0]
    high = key >= hi
    central = np.zeros_like(high)
    central[:, int(0.15 * key.shape[1]):int(0.85 * key.shape[1]),
            int(0.15 * key.shape[2]):int(0.85 * key.shape[2])] = True
    counts = (high & central).sum(axis=(1, 2))
    # bone-bearing slices, best first (ignore the top/bottom 5 % of the stack)
    guard = max(1, int(0.05 * nz))
    candidates = list(pool) if pool else list(range(guard, nz - guard))
    ranked = sorted(candidates, key=lambda z: counts[z], reverse=True)
    bone_slices = [int(z) for z in ranked[:6] if counts[z] > 0]
    flat_slices = ([int(z) for z in candidates] if pool
                   else [int(z) for z in np.linspace(int(0.15 * nz), int(0.85 * nz),
                                                     n_slices).round()])

    return {
        "bone": pick_roi(key, want="texture", size=bone_size, slices=bone_slices,
                         title="骨小梁纹理 ROI"),
        "flat": pick_roi(key, want="flat", size=flat_size, slices=flat_slices,
                         title="平坦软组织 ROI（噪声基准）"),
    }


def patch_slices(roi: dict) -> tuple:
    """Slice tuple of the patch, whose ``y``/``x`` are the window *centre*."""
    half = roi["size"] // 2
    return (roi["z"], slice(roi["y"] - half, roi["y"] - half + roi["size"]),
            slice(roi["x"] - half, roi["x"] - half + roi["size"]))


# --------------------------------------------------------------------------- figures


def _window(reference: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    finite = reference[np.isfinite(reference)]
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def _window_source(order) -> str:
    """Volume the greyscale windows are taken from (the noisiest available)."""
    for name in ("noisy", "plain"):
        if name in order:
            return name
    return order[0]


def render_orthogonal(volumes, shape, out_path: Path, order, window_key: str) -> Path:
    nz, ny, nx = shape
    views = [
        (f"轴位 z={nz // 2}", lambda v: v[nz // 2]),
        (f"冠状位 y={ny // 2}", lambda v: v[:, ny // 2, :]),
        (f"矢状位 x={nx // 2}", lambda v: v[:, :, nx // 2]),
    ]
    vmin, vmax = _window(np.asarray(volumes[window_key]))
    fig, axes = plt.subplots(len(order), len(views), figsize=(13.5, 3.2 * len(order)))
    fig.suptitle(
        "3D 引导滤波锐化 —— 同一层面、同一灰度窗 "
        f"（{vmin:.3f} ~ {vmax:.3f}，取自{METHOD_LABELS.get(window_key, window_key)}）",
        fontsize=14,
    )
    for row, name in enumerate(order):
        for col, (view_label, pick) in enumerate(views):
            ax = axes[row][col]
            ax.imshow(pick(np.asarray(volumes[name])), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(view_label, fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS.get(name, name), fontsize=10, labelpad=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def render_rois(volumes, rois, out_path: Path, order, window_key: str) -> Path:
    keys = ["bone", "flat"]
    fig, axes = plt.subplots(len(order), len(keys), figsize=(4.6 * len(keys), 3.4 * len(order)))
    fig.suptitle(
        "ROI 放大对照（每列一个窗，窗取自同一体积的同一 ROI）\n"
        "左列看骨小梁/牙周膜纹理有没有回来，右列看噪声有没有被放大",
        fontsize=14,
    )
    windows = {k: _window(np.asarray(volumes[window_key])[patch_slices(rois[k])]) for k in keys}
    for row, name in enumerate(order):
        volume = np.asarray(volumes[name])
        for col, key in enumerate(keys):
            vmin, vmax = windows[key]
            ax = axes[row][col]
            ax.imshow(volume[patch_slices(rois[key])], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(rois[key]["label"], fontsize=11, pad=8)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS.get(name, name), fontsize=10, labelpad=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return out_path


def render_metrics(metrics, out_path: Path, order) -> Path:
    names = list(order)
    colors = [METHOD_COLORS.get(n, "#333333") for n in names]
    short = [METHOD_LABELS.get(n, n).split("（")[0] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    bands = ("low", "mid", "high")
    band_labels = ["软组织带\n(μ≤0.2)", "过渡带", "骨带\n(μ≥0.4)"]
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = [metrics[name]["bands"][b] for b in bands]
        positions = np.arange(len(bands)) + index * width - 0.4 + width / 2
        axes[0].bar(positions, values, width=width, color=colors[index], label=short[index],
                    edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(range(len(bands)))
    axes[0].set_xticklabels(band_labels)
    axes[0].set_ylabel("Immerkaer σ（高频能量）")
    axes[0].set_title("按密度带的 σ：锐化把哪一段抬回来了")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    octave_names = ("fine", "mid", "coarse")
    for column, key in enumerate(("flat", "bone")):
        ax = axes[1 + column]
        base = metrics["plain"][f"{key}_patch_octaves"]
        width = 0.8 / len(names)
        for index, name in enumerate(names):
            values = [metrics[name][f"{key}_patch_octaves"][b] for b in octave_names]
            positions = np.arange(len(octave_names)) + index * width - 0.4 + width / 2
            ax.bar(positions, values, width=width, color=colors[index], label=short[index],
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(octave_names)))
        ax.set_xticklabels(["fine\n(1 体素)", "mid\n(骨小梁)", "coarse\n(背景)"])
        ax.set_ylabel("分频段 σ")
        ax.set_title(("平坦 ROI" if key == "flat" else "骨小梁 ROI") + " 的分频段能量")
        ax.grid(axis="y", alpha=0.3)
        if key == "flat":
            ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
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
    parser.add_argument("--dir", type=Path, required=True, help="Folder holding the volumes.")
    parser.add_argument("--input", default="full.bhc.r4b.raw", help="Volume to sharpen.")
    parser.add_argument("--guide", default=None, help="Optional joint-guidance volume.")
    parser.add_argument("--reference", type=_parse_named, action="append", default=None,
                        metavar="NAME=PATH",
                        help="Extra volumes drawn in the figures (repeatable), e.g. "
                             "noisy=full.bhc.raw or r3b=full.bhc.r3b.raw.")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--radius", type=int, default=2,
                        help="Guided-filter window radius, in voxels.  The structure field "
                             "is A*(I - box_r(I)), so r sets the *scale* that gets "
                             "enhanced: r=2 boosts everything finer than ~1.25 mm.")
    parser.add_argument("--eps", type=float, default=None, help="Absolute eps (overrides --eps-k).")
    parser.add_argument("--eps-k", type=float, default=8.0,
                        help="eps = (k * sigma)^2. In a flat region the local Wiener gain is "
                             "1/(1+k^2), so k bounds the fraction of the noise floor that "
                             "gets boosted: k=2 -> 20%%, k=4 -> 5.9%%, k=8 -> 1.5%%. Sweep "
                             "on r4b: the bone/flat selectivity ratio is 3.2 at k=1, 10 at "
                             "k=2, 32 at k=4, 90+ at k=8.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override the measured noise sigma used for eps.")
    parser.add_argument("--sigma-source", choices=("flat", "tissue"), default="flat",
                        help="'flat' (default) uses the auto-picked flat patch, i.e. the "
                             "actual residual noise floor, so k reads as 'k x the noise "
                             "floor'. 'tissue' uses the whole-volume tissue-masked "
                             "Immerkaer sigma, which also counts bone structure and is "
                             "therefore an overestimate of the noise.")
    parser.add_argument("--strength", type=float, default=1.0, help="Detail boost factor.")
    parser.add_argument("--max-darken", type=float, default=None, metavar="D",
                        help="Cap how much darker a voxel may become, in volume units "
                             "(tissue is ~0.2-0.4, so D=0.2 is a mild cap). Removes the "
                             "dark rim a wide high-pass leaves in narrow low-density gaps. "
                             "Costs bone gain: the bound is one-sided on purpose, but even "
                             "so the marrow half of the trabecular contrast is what it "
                             "suppresses. Prefer lowering --radius first: r=2 -> 1 halves "
                             "the worst darkening for only ~13%% of the bone gain.")
    parser.add_argument("--mode", choices=MODES, default="wiener")
    parser.add_argument("--slab", type=int, default=96, help="z-slab size (memory bound).")
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.4)
    parser.add_argument("--max-slices", type=int, default=24)
    parser.add_argument("--flat-roi-size", type=int, default=128,
                        help="Side of the flat soft-tissue patch (voxels).")
    parser.add_argument("--bone-roi-size", type=int, default=80,
                        help="Side of the trabecular-bone patch (voxels). Trabecular bone "
                             "occupies only part of a mandible cross-section, so this has "
                             "to be smaller than the flat patch or no window qualifies.")
    parser.add_argument("--flat-roi", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                        help="Override the flat-patch centre.")
    parser.add_argument("--bone-roi", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                        help="Override the bone-patch centre.")
    parser.add_argument("--roi-cache", type=Path, default=None,
                        help="JSON with flat/bone ROI picks. Written by "
                             "guided_sharpen_sweep.py; reused here so the sweep and this run "
                             "measure exactly the same voxels.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output RAW path (default <dir>/<input stem>.gfsharp.raw).")
    parser.add_argument("--figures", type=Path, default=None, help="defaults to <dir>/figures_gf")
    parser.add_argument("--json", type=Path, default=None, help="defaults to <dir>/gfsharp.json")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="defaults to <dir>/gfsharp_summary.md")
    args = parser.parse_args()

    shape = tuple(int(v) for v in args.shape)
    # Every path flag is relative to --dir unless it is absolute.  Using the
    # process CWD for outputs while --input/--guide use --dir silently scattered
    # deliverables into the repo root, so all four are resolved the same way.
    args.out = _under(args.dir, args.out)
    args.json = _under(args.dir, args.json)
    args.markdown = _under(args.dir, args.markdown)
    args.figures = _under(args.dir, args.figures)
    args.roi_cache = _under(args.dir, args.roi_cache)
    input_path = args.dir / args.input
    references = [(name, path if path.is_absolute() else args.dir / path)
                  for name, path in (args.reference or [])]
    for label, path in [("input", input_path), *references]:
        if not path.exists():
            raise SystemExit(f"missing volume for {label!r}: {path}")
    guide_path = args.dir / args.guide if args.guide else None
    if guide_path is not None and not guide_path.exists():
        raise SystemExit(f"missing guide volume: {guide_path}")

    if not _use_cjk_font():
        print("[warn] no CJK font found; falling back to the default font")

    volume = _load(input_path, shape)
    guide = _load(guide_path, shape) if guide_path else None
    key = np.asarray(volume)
    masks = band_masks(key, args.lo, args.hi)
    tissue = key > TISSUE_FLOOR
    tissue_n = int(np.count_nonzero(tissue))
    tissue_mean = float(np.sum(key[tissue], dtype=np.float64) / tissue_n)

    # ROIs come first: with --sigma-source flat the noise floor they measure is
    # what eps is derived from.
    if args.roi_cache and args.roi_cache.is_file():
        rois = json.loads(args.roi_cache.read_text(encoding="utf-8"))
        print(f"[roi] reused {args.roi_cache}")
    else:
        rois = pick_rois(key, args.lo, args.hi, flat_size=args.flat_roi_size,
                         bone_size=args.bone_roi_size)
    for name, override in (("flat", args.flat_roi), ("bone", args.bone_roi)):
        if override is not None:
            rois[name].update(zip(("z", "y", "x"), (int(v) for v in override)))
            rois[name]["label"] += "（手工指定）"

    flat_patch, bone_patch = patch_slices(rois["flat"]), patch_slices(rois["bone"])
    sigma_tissue = estimate_noise_std_immerkaer(volume, max_slices=args.max_slices)
    sigma_flat = patch_sigma(key[flat_patch])
    sigma = args.sigma
    if sigma is None:
        sigma = sigma_flat if args.sigma_source == "flat" else sigma_tissue
    eps = args.eps if args.eps is not None else eps_from_noise(sigma, args.eps_k)
    print(f"[noise] sigma_tissue={sigma_tissue:.6f} sigma_flat={sigma_flat:.6f} "
          f"-> using sigma={sigma:.6f} (--sigma-source {args.sigma_source})")
    print(f"[cfg] mode={args.mode} radius={args.radius} eps={eps:.3e} eps_k={args.eps_k} "
          f"strength={args.strength} guide={guide_path or 'self'} slab={args.slab}")
    for name, roi in rois.items():
        print(f"[roi] {name:5s} size={roi['size']:3d} z={roi['z']:3d} y={roi['y']:3d} x={roi['x']:3d}"
              f"  window_mean={roi['window_mean']:.3f}  local_sigma={roi['local_std']:.5f}")

    started = time.perf_counter()
    field = guided_structure_field(volume, radius=args.radius, eps=eps, mode=args.mode,
                                   guide=guide, slab=args.slab)
    print(f"[field] {time.perf_counter() - started:.1f}s  std={field.std():.6e} "
          f"mean={field.mean():+.3e} abs_max={np.abs(field).max():.3e}")

    sharpened = np.add(volume, np.float32(args.strength) * field).astype(np.float32)
    if args.max_darken is not None:
        # Whole-volume extremes, so they sit slightly above the sidecar's masked
        # numbers (``darken_max`` is tissue-only, ``darken_visible_max`` further
        # requires the *input* to exceed VISIBLE_FLOOR).  Labelled, because three
        # similar-looking darkening maxima is a fine way to confuse a reader.
        before = np.maximum(volume - sharpened, 0.0).max()
        sharpened = cap_darkening(sharpened, volume, args.max_darken)
        after = np.maximum(volume - sharpened, 0.0).max()
        print(f"[guard] max darkening (whole volume) {before:.4f} -> {after:.4f} "
              f"(cap {args.max_darken:g})")
    out_path = args.out or (args.dir / (Path(args.input).stem + ".gfsharp.raw"))
    _write_raw(out_path, sharpened)
    print(f"[raw] {out_path}  md5={_md5(out_path)}")

    # ------------------------------------------------------------------ metrics
    # References are drawn first so the reading order is: what the reconstruction
    # started with -> the mid-chain reference -> what we sharpen -> the result.
    volumes = {"plain": key, "sharp": sharpened}
    order = ["plain", "sharp"]
    for name, path in references:
        volumes[name] = _load(path, shape)
    order = [n for n, _ in references] + ["plain", "sharp"]

    noisy_patches = None
    if "noisy" in volumes:
        noisy_patches = {
            key: np.asarray(volumes["noisy"][patch_slices(rois[key])], dtype=np.float32)
            for key in ("flat", "bone")
        }

    input_global = estimate_noise_std_immerkaer(volume, max_slices=args.max_slices)
    # The darkening QC is only meaningful for volumes derived from the input by
    # this sharpening; `noisy` / `r3b` are different reconstructions entirely, so
    # "how much darker than plain" would just be measuring their noise.
    metrics = {
        name: _metrics(volumes[name], masks, tissue, tissue_mean, input_global,
                       flat_patch, bone_patch, args.max_slices, noisy_patches,
                       reference=volume if name in ("plain", "sharp") else None)
        for name in order
    }

    header = "method".ljust(8) + "".join(b.rjust(11) for b in
                                         ("low", "mid", "high", "global", "flat", "bone"))
    print(header + "bone/flat".rjust(11))
    for name in order:
        row = metrics[name]
        ratio = row["bone_patch_sigma"] / row["flat_patch_sigma"]
        print(f"{name:8s}"
              + f"{row['bands']['low']:11.5f}{row['bands']['mid']:11.5f}"
              + f"{row['bands']['high']:11.5f}{row['global_sigma']:11.5f}"
              + f"{row['flat_patch_sigma']:11.5f}{row['bone_patch_sigma']:11.5f}"
              + f"{ratio:11.3f}")

    figures = args.figures or (args.dir / "figures_gf")
    window_key = _window_source(order)
    made = [
        render_orthogonal(volumes, shape, figures / "gfsharp_orthogonal.png", order, window_key),
        render_rois(volumes, rois, figures / "gfsharp_roi.png", order, window_key),
        render_metrics(metrics, figures / "gfsharp_metrics.png", order),
    ]
    for path in made:
        print(f"[figure] {path}")

    payload = {
        "input": str(input_path),
        "input_md5": _md5(input_path),
        "guide": str(guide_path) if guide_path else None,
        "output": str(out_path),
        "output_md5": _md5(out_path),
        "shape": list(shape),
        "mode": args.mode,
        "radius": args.radius,
        "eps": eps,
        "eps_k": args.eps_k if args.eps is None else None,
        "sigma_for_eps": sigma,
        "sigma_source": args.sigma_source if args.sigma is None else "explicit",
        "sigma_tissue": sigma_tissue,
        "sigma_flat_patch": sigma_flat,
        "strength": args.strength,
        "max_darken": args.max_darken,
        "slab": args.slab,
        "band_key": "the sharpened volume itself (before sharpening)",
        "lo": args.lo,
        "hi": args.hi,
        "tissue_floor": TISSUE_FLOOR,
        "rois": rois,
        "order": order,
        "metrics": metrics,
        "figures": [str(p) for p in made],
        "note": (
            "Immerkaer sigma is high-frequency energy and cannot separate noise from "
            "texture; the flat patch is the reference for 'how much noise was added' and "
            "the bone patch for 'how much texture came back'. 'fine' is the voxel-scale "
            "octave (mostly noise), 'mid' the trabecular octave, 'coarse' the background. "
            "'hf_correlation_vs_noisy' is a diagnostic only -- a high value means the "
            "sharpened high frequencies track the input's (i.e. the model is acting as a "
            "near-linear high-frequency attenuator), not that they are signal."
        ),
    }
    json_path = args.json or (args.dir / "gfsharp.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[json] {json_path}")

    md_path = args.markdown or (args.dir / "gfsharp_summary.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(metrics, order, rois, payload, md_path)
    print(f"[markdown] {md_path}")
    return 0


def _write_markdown(metrics, order, rois, payload, out_path: Path) -> None:
    plain, sharp = metrics["plain"], metrics["sharp"]
    delta_flat = sharp["flat_patch_sigma"] / plain["flat_patch_sigma"] - 1.0
    delta_bone = sharp["bone_patch_sigma"] / plain["bone_patch_sigma"] - 1.0
    lines = [
        "# 3D 引导滤波锐化 —— 单配置结果",
        "",
        f"- 输入：`{Path(payload['input']).name}`（md5 `{payload['input_md5']}`），"
        f"shape {tuple(payload['shape'])}",
        f"- 参数：mode `{payload['mode']}`，radius {payload['radius']}，"
        f"σ {payload['sigma_for_eps']:.6f}（来源 {payload['sigma_source']}，"
        f"全组织实测 {payload['sigma_tissue']:.6f} / 平坦 ROI 实测 "
        f"{payload['sigma_flat_patch']:.6f}），eps {payload['eps']:.3e}"
        f"（k={payload['eps_k']}），strength {payload['strength']}",
        f"- 输出：`{Path(payload['output']).name}`（md5 `{payload['output_md5']}`）",
        f"- 密度带按锐化前体数据的强度分带，lo={payload['lo']} / hi={payload['hi']}，"
        f"组织阈值 >{payload['tissue_floor']}",
        "",
        "## 指标",
        "",
        "| 方法 | 软组织带 | 过渡带 | 骨带 | 全局 σ | 平坦 ROI σ | 骨 ROI σ | 骨/平 |",
        "| ---- | -------- | ------ | ---- | ------ | ---------- | -------- | ----- |",
    ]
    for name in order:
        row = metrics[name]
        lines.append(
            f"| {METHOD_LABELS.get(name, name)} "
            f"| {row['bands']['low']:.5f} | {row['bands']['mid']:.5f} | {row['bands']['high']:.5f} "
            f"| {row['global_sigma']:.5f} | {row['flat_patch_sigma']:.5f} "
            f"| {row['bone_patch_sigma']:.5f} "
            f"| {row['bone_patch_sigma'] / row['flat_patch_sigma']:.3f} |"
        )
    lines += [
        "",
        "## 分频段能量（ROI 内）",
        "",
        "| 方法 | 平坦 fine | 平坦 mid | 骨 fine | 骨 mid | 骨 coarse |",
        "| ---- | --------- | -------- | ------- | ------ | --------- |",
    ]
    for name in order:
        row = metrics[name]
        f_oct, b_oct = row["flat_patch_octaves"], row["bone_patch_octaves"]
        lines.append(
            f"| {METHOD_LABELS.get(name, name)} | {f_oct['fine']:.5f} | {f_oct['mid']:.5f} "
            f"| {b_oct['fine']:.5f} | {b_oct['mid']:.5f} | {b_oct['coarse']:.5f} |"
        )
    lines += [
        "",
        "## ROI 位置（自动选取，可用 --flat-roi / --bone-roi 覆盖）",
        "",
    ]
    for key_name, roi in rois.items():
        lines.append(f"- `{key_name}`：{roi['label']}，窗内均值 {roi['window_mean']:.3f}，"
                     f"局部 σ {roi['local_std']:.5f}")
    lines += [
        "",
        "「骨/平」= 骨 ROI σ / 平坦 ROI σ。未去噪的 `full.bhc` 上这个比值≈1.2（高频主要是噪声，",
        "与密度无关）；比值越高，说明高频里“与密度有关”的结构成分越多。",
        "",
        "## 读数",
        "",
        f"- 平坦 ROI σ {plain['flat_patch_sigma']:.5f} → {sharp['flat_patch_sigma']:.5f}"
        f"（{100 * delta_flat:+.1f}%）：这是**噪声代价**。",
        f"- 骨 ROI σ {plain['bone_patch_sigma']:.5f} → {sharp['bone_patch_sigma']:.5f}"
        f"（{100 * delta_bone:+.1f}%）：这是**纹理/边缘回收量**（含噪声代价，见上图分频段）。",
        f"- 组织均值偏移 {sharp['mean_offset_on_tissue']:+.2e}，"
        f"体数据极值 [{sharp['volume_min']:.3f}, {sharp['volume_max']:.3f}]"
        f"（锐化前 [{plain['volume_min']:.3f}, {plain['volume_max']:.3f}]）。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
