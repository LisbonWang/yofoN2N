"""Side-by-side ROI zoom of several conventional-denoise variants of one study.

``compare_conv_denoise.py`` renders the whole method ladder (input + conventional + AI
rounds) at survey scale.  When the conventional parameters themselves are being swept --
``filter.xml`` edits, PWLS ``beta``, the ReduceNoise / Anisotropic blocks -- a survey
figure is too small and carries too many rows that are not under test.  This script keeps
only the variants that matter: the undenoised reference plus every conventional volume
found by a glob, rendered large, on one shared gray window per ROI.

The reference volume also supplies the ROI placement (via the same band logic the survey
figure uses), so all rows are cropped identically and every difference is attributable to
the parameters rather than to a shifted crop.

Example::

    python scripts/compare_conv_zoom.py \
      --dir outputs/air3_ypdl0027_bhc_5models --shape 360 640 640 \
      --out outputs/air3_ypdl0027_bhc_5models/figures_pwls600/zoom.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conv_denoise import TISSUE_FLOOR, _use_cjk_font, select_rois  # noqa: E402

DEFAULT_PATTERN = "full.bhc.conv_rn_ani*.raw"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Folder holding the volumes.")
    parser.add_argument("--input", default="full.bhc.raw", help="Undenoised reference volume.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="Glob (relative to --dir) selecting the variants to compare.")
    parser.add_argument("--volume", type=_named_path, action="append", default=None,
                        metavar="LABEL=PATH", help="Repeatable; extra volume to include.")
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--lo", type=float, default=0.2, help="Band key: soft-tissue cut.")
    parser.add_argument("--hi", type=float, default=0.4, help="Band key: bone cut.")
    parser.add_argument("--roi-size", type=int, default=192)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=110)
    return parser.parse_args(argv)


def _named_path(text: str) -> tuple[str, Path]:
    label, _, raw = text.partition("=")
    if not raw:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw)


def _label(name: str, prefix: str = "full.bhc.", suffix: str = ".raw") -> str:
    """``full.bhc.conv_rn_ani.pwls600.raw`` -> ``conv_rn_ani.pwls600``."""
    stem = name
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem or name


def _crop(volume: np.ndarray, axis: int, index: int, box) -> np.ndarray:
    a0, a1, b0, b1 = box
    if axis == 0:
        return np.asarray(volume[index, a0:a1, b0:b1])
    return np.asarray(volume[a0:a1, index, b0:b1])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shape = tuple(int(v) for v in args.shape)

    entries: list[tuple[str, Path]] = [(f"输入 {args.input}（无降噪）", args.dir / args.input)]
    # Shortest name first: the base volume ``full.bhc.conv_rn_ani.raw`` has no variant
    # suffix, so it sorts ahead of ``...pwls600.raw`` and every sweep reads top-down.
    found = sorted(args.dir.glob(args.pattern), key=lambda p: (p.name.count("."), p.name))
    for path in found:
        entries.append((_label(path.name), path))
    for label, path in (args.volume or []):
        entries.append((label, path if path.is_absolute() else args.dir / path))
    if len(entries) < 2:
        raise SystemExit(f"nothing to compare: {args.pattern!r} matched no file under {args.dir}")

    for label, path in entries:
        if not path.is_file():
            raise SystemExit(f"missing volume for {label!r}: {path}")
        size = path.stat().st_size
        expected = int(np.prod(shape)) * 4
        if size != expected:
            raise SystemExit(
                f"{path.name} is {size} B but --shape {shape} implies {expected} B")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_cjk_font()
    volumes = {label: np.memmap(path, dtype="<f4", mode="r", shape=shape) for label, path in entries}
    reference = volumes[entries[0][0]]
    # Materialise the reference once: select_rois needs random access to it, and reading a
    # 590 MB volume twice (once per argument) would double both the I/O and the footprint.
    reference_array = np.asarray(reference)
    rois = select_rois(reference_array, reference_array, args.lo, args.hi, args.roi_size)

    rows, cols = len(entries), len(rois)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.6 * rows), squeeze=False)
    for col, roi in enumerate(rois):
        ref_crop = _crop(reference, roi["axis"], roi["index"], roi["box"])
        vmin, vmax = np.percentile(ref_crop, 0.5), np.percentile(ref_crop, 99.5)
        for row, (label, _path) in enumerate(entries):
            ax = axes[row][col]
            ax.imshow(_crop(volumes[label], roi["axis"], roi["index"], roi["box"]),
                      cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(roi["label"], fontsize=15, pad=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=14, labelpad=12)
    spacing = float(np.percentile(reference_array, 0.5))
    fig.suptitle(
        f"常规降噪变体对照（同一 study / 同一 ROI / 同一灰度窗；"
        f"{args.roi_size}×{args.roi_size} 体素，组织阈值 >{TISSUE_FLOOR}，p0.5={spacing:.4g}）",
        fontsize=17, y=0.975,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[zoom] {args.out}")
    for label, _path in entries:
        print(f"  row: {label}")
    for roi in rois:
        print(f"  roi: {roi['key']} {roi['label']} box={roi['box']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
