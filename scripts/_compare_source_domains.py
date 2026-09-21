"""Same denoisers, different inputs: is the *source volume* in the training distribution?

``denoise_multi_models.py`` answers "which round?" for one input, and
``compare_conv_denoise.py`` compares the AI rounds against conventional denoising on one
input.  Neither answers the deployment question: the same checkpoint is about to be
pointed at a volume that did **not** come out of ``AIReduceNoise3`` -- does it still behave
the way it did during validation, or is that input out of distribution?

This script takes N sources (each ``LABEL=<stem>.raw``) together with their
``<stem>.<model>.raw`` outputs and reports, for every (source, state) pair:

* the mask-free Immerkaer sigma -- exactly the global number ``denoise_multi_models``
  prints, so it is comparable with the stored run summaries;
* the three density bands under **each source's own** input mask, so a source that really
  does contain more bone is not judged with another patient's mask;
* the cut % against that source's own input, and the tissue mean offset;
* the same three bands recomputed under a single **common** mask taken from one reference
  source, so the cross-source columns are read off identical voxels.

A figure renders every state of every source on the same ROI, with the grey window taken
from each source's own input crop, so "before vs r3b" is a difference in the data and not
in the display scaling.

Example::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/_compare_source_domains.py \
      --dir outputs/air3_ypdl0027_bhc_5models --shape 360 640 640 \
      --source plain=full.raw --source bhc=full.bhc.raw --source external=full.ex.raw \
      --model r3b --model r4b \
      --out-json outputs/air3_ypdl0027_bhc_5models/source_domains.json \
      --out-md   outputs/air3_ypdl0027_bhc_5models/source_domains.md \
      --out-fig  outputs/air3_ypdl0027_bhc_5models/source_domains.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conv_denoise import (  # noqa: E402
    TISSUE_FLOOR,
    _md5,
    _roi_crop,
    _use_cjk_font,
    _window,
    band_masks,
    select_rois,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.training.metrics import (  # noqa: E402
    band_noise_std_immerkaer,
    estimate_noise_std_immerkaer,
)

DTYPE = np.dtype("<f4")

#: ``full.bhc.raw`` -> ``full.bhc``; the model output is ``<stem>.<model>.raw``.
DEFAULT_MODELS = ("r0", "r1b", "r2b", "r3b", "r4b")


def _stem(filename: str) -> str:
    return filename[: -len(".raw")] if filename.endswith(".raw") else filename


def _measure(
    volume: np.ndarray,
    masks: dict[str, np.ndarray],
    tissue: np.ndarray,
    tissue_mean: float,
    input_global: float,
    max_slices: int,
) -> dict:
    """Same metric definitions as ``denoise_multi_models.py`` (identical voxel sets)."""
    banded = {
        band: band_noise_std_immerkaer(volume, mask, max_slices=max_slices)
        for band, mask in masks.items()
    }
    global_sigma = estimate_noise_std_immerkaer(volume, max_slices=max_slices)
    tissue_n = int(np.count_nonzero(tissue))
    mean_on_tissue = (
        float(np.sum(volume[tissue], dtype=np.float64) / tissue_n) if tissue_n else float("nan")
    )
    return {
        "global_sigma": global_sigma,
        "bands": banded,
        "band_differentiation_high_over_low": (
            float(banded["high"] / banded["low"]) if banded["low"] else float("nan")
        ),
        "sigma_reduction_vs_input_pct": (
            float(100.0 * (1.0 - global_sigma / input_global)) if input_global else float("nan")
        ),
        "mean_offset_on_tissue": mean_on_tissue - tissue_mean,
        "volume_min": float(volume.min()),
        "volume_max": float(volume.max()),
        "volume_mean": float(volume.mean()),
        "volume_std": float(volume.std()),
    }


def _parse_source(text: str) -> tuple[str, str]:
    label, _, filename = text.partition("=")
    if not label or not filename:
        raise argparse.ArgumentTypeError("expected LABEL=<stem>.raw")
    return label, filename


def _select_blocks(volume: np.ndarray, lo: float, hi: float, size: int) -> dict[str, dict]:
    """Pick one provably-flat and one trabecular-bone block on the reference volume.

    Global/masked Immerkaer sigma cannot separate noise from texture, so the number that
    actually decides "did the model stop denoising?" is the ratio between a block that can
    *only* contain noise and a block that is structural.  Both are placed on the mask-ref
    input once, in full-resolution coordinates, so every source and every state is read off
    identical voxels.  The flat block additionally requires 99 % of its voxels to sit in
    the soft-tissue band, so it cannot be an air or bone patch in disguise.
    """
    from scipy.ndimage import uniform_filter

    margin = size // 2
    low = (volume > TISSUE_FLOOR) & (volume <= lo)
    bone = volume >= hi

    edge = np.zeros(volume.shape, dtype=bool)
    for axis in range(3):
        for sl in (slice(0, margin + 1), slice(-(margin + 1), None)):
            idx = [slice(None)] * 3
            idx[axis] = sl
            edge[tuple(idx)] = True

    low_frac = uniform_filter(low.astype(np.float32), size=size, mode="constant")
    mean = uniform_filter(volume, size=size, mode="constant")
    mean_sq = uniform_filter(volume * volume, size=size, mode="constant")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    del mean, mean_sq

    flat_score = np.where(edge | (low_frac < 0.99), np.inf, variance)
    flat_centre = tuple(int(v) for v in np.unravel_index(int(np.argmin(flat_score)), variance.shape))
    bone_frac = uniform_filter(bone.astype(np.float32), size=size, mode="constant")
    bone_score = np.where(edge, -1.0, bone_frac)
    bone_centre = tuple(int(v) for v in np.unravel_index(int(np.argmax(bone_score)), bone_frac.shape))

    boxes = {}
    for key, centre, purity in (("flat", flat_centre, float(low_frac[flat_centre])),
                                ("bone", bone_centre, float(bone_frac[bone_centre]))):
        boxes[key] = {
            "centre": centre,
            "purity": purity,
            "box": tuple((c - margin, c + margin) for c in centre),
        }
    return boxes


def _block_sigma(volume: np.ndarray, box: tuple, max_slices: int) -> float:
    z0, z1 = box[0]
    y0, y1 = box[1]
    x0, x1 = box[2]
    sub = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    return band_noise_std_immerkaer(sub, np.ones(sub.shape, dtype=bool), max_slices=max_slices)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Folder holding the volumes.")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--source", type=_parse_source, action="append", required=True,
                        metavar="LABEL=STEM.RAW", help="Repeatable; 2 or more sources.")
    parser.add_argument("--model", action="append", default=None,
                        help=f"Repeatable; state suffix of <stem>.<model>.raw (default "
                             f"{','.join(DEFAULT_MODELS)}).")
    parser.add_argument("--mask-ref", default=None,
                        help="Source label whose input defines the *common* band mask "
                             "(default: the first --source).")
    parser.add_argument("--lo", type=float, default=0.2, help="Band key: soft-tissue cut.")
    parser.add_argument("--hi", type=float, default=0.4, help="Band key: bone cut.")
    parser.add_argument("--roi-size", type=int, default=192)
    parser.add_argument("--block-size", type=int, default=32,
                        help="Edge of the flat / bone control blocks, in voxels (default 32).")
    parser.add_argument("--share-roi", action="store_true",
                        help="Place every source on the *same* ROI (and the same grey "
                             "window) taken from --mask-ref.  Only meaningful when the "
                             "sources are spatially aligned; it is strictly stronger than "
                             "per-source ROI placement when they are.")
    parser.add_argument("--max-slices", type=int, default=32,
                        help="Slices sampled per banded/global sigma (default 32).")
    parser.add_argument("--fig-models", action="append", default=None,
                        help="Repeatable; subset of --model to render in the figure "
                             "(default: the last two, so a 3-source sheet stays 9 rows).")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-fig", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args(argv)
    if len(args.source) < 2:
        raise SystemExit("--source needs at least two entries to compare anything")
    args.model = args.model or list(DEFAULT_MODELS)
    args.fig_models = args.fig_models or list(args.model[-2:])
    for name in args.fig_models:
        if name not in args.model:
            raise SystemExit(f"--fig-models {name!r} is not one of --model {args.model}")
    labels = [label for label, _ in args.source]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate --source labels in {labels}")
    args.mask_ref = args.mask_ref or labels[0]
    if args.mask_ref not in labels:
        raise SystemExit(f"--mask-ref {args.mask_ref!r} is not one of {labels}")
    return args


def _resolve(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    """``{label: {'input': path, '<model>': path, ...}}`` with size validation."""
    expected = int(np.prod(args.shape)) * 4
    resolved: dict[str, dict[str, Path]] = {}
    for label, filename in args.source:
        stem = _stem(filename)
        paths = {"input": args.dir / filename}
        for model in args.model:
            paths[model] = args.dir / f"{stem}.{model}.raw"
        for state, path in paths.items():
            if not path.is_file():
                raise SystemExit(f"missing {state} volume for {label!r}: {path}")
            size = path.stat().st_size
            if size != expected:
                raise SystemExit(
                    f"{path.name} is {size} B but --shape {tuple(args.shape)} implies {expected} B")
        resolved[label] = paths
    return resolved


def _render_figure(
    args: argparse.Namespace,
    resolved: dict[str, dict[str, Path]],
    rows: dict[str, dict[str, dict]],
    out_path: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_cjk_font()
    labels = [label for label, _ in args.source]
    states = ["input"] + list(args.fig_models)

    # One ROI set per source, unless the caller asserts the sources are aligned and wants
    # every row cropped identically (the stronger comparison when it applies).
    volume_of = {lab: np.memmap(resolved[lab]["input"], dtype=DTYPE, mode="r", shape=args.shape)
                 for lab in labels}
    if args.share_roi:
        shared = select_rois(volume_of[args.mask_ref], volume_of[args.mask_ref],
                             args.lo, args.hi, args.roi_size)
        shared_window = [_window(_roi_crop(volume_of[args.mask_ref], roi)) for roi in shared]
        rois = {lab: shared for lab in labels}
        windows = {lab: shared_window for lab in labels}
    else:
        rois = {lab: select_rois(volume_of[lab], volume_of[lab], args.lo, args.hi, args.roi_size)
                for lab in labels}
        windows = {
            lab: [_window(_roi_crop(volume_of[lab], roi)) for roi in rois[lab]]
            for lab in labels
        }

    n_rows = sum(len(states) for _ in labels)
    n_cols = len(rois[labels[0]])
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 4.4 * n_rows), squeeze=False)

    row_index = 0
    for lab in labels:
        for state, roi in zip(states, rois[lab]):
            volume = np.memmap(resolved[lab][state], dtype=DTYPE, mode="r", shape=args.shape)
            for col, roi in enumerate(rois[lab]):
                ax = axes[row_index][col]
                vmin, vmax = windows[lab][col]
                ax.imshow(_roi_crop(volume, roi), cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_index == 0 or state == "input":
                    ax.set_title(f"{roi['label']}", fontsize=13, pad=8)
                if col == 0:
                    name = dict(args.source)[lab] if state == "input" else state
                    ax.set_ylabel(f"{lab}\n{name}", fontsize=13, labelpad=10)
            row_index += 1

    scope = (f"全部来源共用 `{args.mask_ref}` 的同一 ROI 与灰度窗" if args.share_roi
             else "每块 ROI 与灰度窗取自该来源自身输入")
    fig.suptitle(
        f"同一组 {('/'.join(args.fig_models))} 权重、不同来源体数据的降噪对照"
        f"（{scope}，灰度窗 p0.5–p99.5；带阈值 {args.lo}/{args.hi}，组织 >{TISSUE_FLOOR}）",
        fontsize=16, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    notes = [f"  {lab}: " + " | ".join(
        f"{r['key']} axis={r['axis']} idx={r['index']} box={r['box']}" for r in rois[lab])
        for lab in labels]
    return notes


def _write_markdown(args: argparse.Namespace, report: dict, out_path: Path) -> None:
    labels = [label for label, _ in args.source]
    states = ["input"] + list(args.model)
    lines = [
        "# 同一权重、不同来源体数据：降噪行为对照",
        "",
        f"- 目录 `{report['dir']}`，shape `{tuple(args.shape)}`，float32",
        f"- 分带 lo={args.lo} / hi={args.hi}，组织下限 {TISSUE_FLOOR}",
        f"- 全局 σ = `estimate_noise_std_immerkaer`（无掩码，内部按各卷自身组织自动加权）",
        f"- 共掩码来自 `{args.mask_ref}` 的**输入**，逐体素锁定；自有掩码来自各来源自身输入",
        "",
        "## 1. 各来源自有掩码（各卷自己的组织分类）",
        "",
        "| 来源 | 状态 | 全局 σ | 削减 % | low | mid | high | diff | 组织均值偏移 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lab in labels:
        for state in states:
            m = report["sources"][lab]["states"][state]
            lines.append(
                f"| {lab} | {state} | {m['global_sigma']:.5f} | "
                f"{m['sigma_reduction_vs_input_pct']:.1f} | {m['bands']['low']:.5f} | "
                f"{m['bands']['mid']:.5f} | {m['bands']['high']:.5f} | "
                f"{m['band_differentiation_high_over_low']:.3f} | "
                f"{m['mean_offset_on_tissue']:+.5f} |")
    lines += ["", "各来源自身掩码的体素数（移植性差异会直接体现在这里）：", "",
              "| 来源 | low | mid | high |", "| --- | --- | --- | --- |"]
    for lab in labels:
        n = report["own_band_voxels"][lab]
        lines.append(f"| {lab} | {n['low']} | {n['mid']} | {n['high']} |")

    common = report["common_mask_source"]
    lines += ["", f"## 2. 共掩码（来自 `{common}` 输入，逐体素锁定）", "",
              "| 来源 | 状态 | low | mid | high | diff |", "| --- | --- | --- | --- | --- | --- |"]
    for lab in labels:
        for state in states:
            m = report["sources"][lab]["states"][state]["common_bands"]
            diff = m["high"] / m["low"] if m["low"] else float("nan")
            lines.append(f"| {lab} | {state} | {m['low']:.5f} | {m['mid']:.5f} | "
                         f"{m['high']:.5f} | {diff:.3f} |")

    lines += ["", "## 3. 来源自身几何/对齐（相对第一个来源的输入）", "",
              "| 来源 | md5 | 体 data min | max | mean | std |", "| --- | --- | --- | --- | --- | --- |"]
    for lab in labels:
        s = report["sources"][lab]
        g = s["states"]["input"]
        lines.append(f"| {lab} | `{s['md5']}` | {g['volume_min']:+.4f} | {g['volume_max']:+.4f} | "
                     f"{g['volume_mean']:+.5f} | {g['volume_std']:.5f} |")

    blocks = report.get("control_blocks")
    if blocks:
        size = blocks["size"]
        lines += ["",
                  f"## 4. 对照块（{size}³ 体素，置于 `{blocks['placed_on']}` 的输入上，逐源同体素）", "",
                  "全局 σ 分不开噪声与纹理。平坦块只可能装噪声，骨块则被结构限底；"
                  "两列各自给出相对**本源输入**的削减倍数。", "",
                  "| 来源 | 状态 | 平坦块 σ | 较本源输入 | 骨块 σ | 较本源输入 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for lab in labels:
            base = report["sources"][lab]["states"]["input"]["control_blocks"]
            for state in states:
                cb = report["sources"][lab]["states"][state]["control_blocks"]
                lines.append(
                    f"| {lab} | {state} | {cb['flat']:.5f} | {base['flat'] / cb['flat']:.2f}x | "
                    f"{cb['bone']:.5f} | {base['bone'] / cb['bone']:.2f}x |")
        det = blocks["blocks"]
        lines += ["",
                  f"- 平坦块中心 {tuple(det['flat']['centre'])}，块内低密度体素占比 "
                  f"{det['flat']['purity']:.4f}",
                  f"- 骨块中心 {tuple(det['bone']['centre'])}，块内 >{args.hi} 体素占比 "
                  f"{det['bone']['purity']:.4f}"]
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved = _resolve(args)
    labels = [label for label, _ in args.source]
    states = ["input"] + list(args.model)

    report: dict = {
        "dir": str(args.dir),
        "shape": list(args.shape),
        "lo": args.lo,
        "hi": args.hi,
        "tissue_floor": TISSUE_FLOOR,
        "max_slices": args.max_slices,
        "models": list(args.model),
        "common_mask_source": args.mask_ref,
        "sources": {},
        "own_band_voxels": {},
    }

    common_key = np.asarray(np.memmap(resolved[args.mask_ref]["input"], dtype=DTYPE, mode="r",
                                      shape=args.shape))
    common_masks = band_masks(common_key, args.lo, args.hi)

    for lab in labels:
        input_volume = np.asarray(np.memmap(resolved[lab]["input"], dtype=DTYPE, mode="r",
                                            shape=args.shape))
        tissue = input_volume > TISSUE_FLOOR
        tissue_mean = float(np.sum(input_volume[tissue], dtype=np.float64)
                            / max(int(np.count_nonzero(tissue)), 1))
        own_masks = band_masks(input_volume, args.lo, args.hi)
        input_global = estimate_noise_std_immerkaer(input_volume, max_slices=args.max_slices)

        report["own_band_voxels"][lab] = {
            band: int(np.count_nonzero(mask)) for band, mask in own_masks.items()
        }
        entry: dict = {
            "input": str(resolved[lab]["input"]),
            "md5": _md5(resolved[lab]["input"]),
            "states": {},
        }
        print(f"\n[{lab}] input={resolved[lab]['input'].name} global_sigma={input_global:.5f}",
              flush=True)
        for state in states:
            path = resolved[lab][state]
            volume = np.memmap(path, dtype=DTYPE, mode="r", shape=args.shape)
            row = _measure(volume, own_masks, tissue, tissue_mean, input_global,
                           args.max_slices)
            row["common_bands"] = {
                band: band_noise_std_immerkaer(volume, mask, max_slices=args.max_slices)
                for band, mask in common_masks.items()
            }
            entry["states"][state] = row
            print(f"  {state:>6}  sigma={row['global_sigma']:.5f} "
                  f"cut={row['sigma_reduction_vs_input_pct']:+6.1f}%  "
                  f"bands(low/mid/high)={row['bands']['low']:.5f}/"
                  f"{row['bands']['mid']:.5f}/{row['bands']['high']:.5f}  "
                  f"diff={row['band_differentiation_high_over_low']:.3f}  "
                  f"common_high={row['common_bands']['high']:.5f}", flush=True)
        report["sources"][lab] = entry

    # Iron rule for reading Immerkaer sigma: a global number cannot separate noise from
    # texture, so also measure two provably-homogeneous blocks -- one that can only hold
    # noise (flat soft tissue) and one that is structure-limited (trabecular bone).  Both
    # are placed once on the mask-ref input and read off identical voxels for every source.
    blocks = _select_blocks(common_key, args.lo, args.hi, args.block_size)
    report["control_blocks"] = {
        "size": args.block_size,
        "placed_on": args.mask_ref,
        "blocks": {
            key: {"centre": list(blk["centre"]), "box": [list(b) for b in blk["box"]],
                  "purity": blk["purity"]}
            for key, blk in blocks.items()
        },
    }
    print(f"\n[blocks] {args.block_size}^3 on {args.mask_ref}: "
          f"flat centre={blocks['flat']['centre']} soft-fraction={blocks['flat']['purity']:.3f} | "
          f"bone centre={blocks['bone']['centre']} bone-fraction={blocks['bone']['purity']:.3f}",
          flush=True)
    print("state".ljust(20) + "flat".rjust(11) + "vs input".rjust(11)
          + "bone".rjust(11) + "vs input".rjust(11), flush=True)
    for lab in labels:
        base = None
        for state in states:
            volume = np.memmap(resolved[lab][state], dtype=DTYPE, mode="r", shape=args.shape)
            sigma = {key: _block_sigma(volume, blk["box"], args.max_slices)
                     for key, blk in blocks.items()}
            report["sources"][lab]["states"][state]["control_blocks"] = sigma
            if base is None:
                base = sigma
            print(((lab + " " + state).ljust(20)
                   + f"{sigma['flat']:.5f}".rjust(11)
                   + f"{base['flat'] / sigma['flat']:.2f}x".rjust(11)
                   + f"{sigma['bone']:.5f}".rjust(11)
                   + f"{base['bone'] / sigma['bone']:.2f}x".rjust(11)), flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[json] {args.out_json}", flush=True)

    if args.out_md:
        _write_markdown(args, report, args.out_md)
        print(f"[md]   {args.out_md}", flush=True)
    if args.out_fig:
        for note in _render_figure(args, resolved, report["sources"], args.out_fig):
            print(f"[roi] {note}", flush=True)
        print(f"[fig]  {args.out_fig}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
