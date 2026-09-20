"""Render input-vs-denoised comparison figures for a denoise_test_set run.

Reads the summary written by ``scripts/denoise_test_set.py`` and renders, per
volume, a 2x3 panel (input / denoised by axial / coronal / sagittal) plus a
summary bar chart of the tissue-masked Immerkaer noise drop per group.

Example:
  python scripts/make_test11_figures.py --run-root outputs/test11_cross_mode
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: {path.stat().st_size} bytes != {expected} for {shape}")
    return np.fromfile(path, dtype="<f4").reshape(shape)


def _clim(reference: np.ndarray) -> tuple[float, float]:
    finite = reference[np.isfinite(reference)]
    return float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))


def _input_of(entry: dict) -> Path:
    """Locate the staged input volume for a summary entry.

    Uses the explicit ``input_path`` field when present; otherwise derives it
    from the recorded output path by swapping the ``denoised`` folder for
    ``inputs`` and stripping the ``_denoised`` suffix from the filename.  (The
    ``input`` key holds the input *statistics*, not a path.)
    """
    explicit = entry.get("input_path")
    if isinstance(explicit, str) and explicit:
        return Path(explicit)
    output = Path(entry["output"])
    parts = list(output.parts)
    for index, part in enumerate(parts):
        if part == "denoised":
            parts[index] = "inputs"
            break
    return Path(*parts).with_name(output.name.replace("_denoised", ""))


def _uid_tail(study_uid: str) -> str:
    """Short study discriminator: the last dotted OID component.

    ``case_id`` alone is not unique -- one device can contribute several
    studies to the test set (e.g. three ``Mode3_1609`` studies on
    ``YPDL-26A-0026``) -- so the figure name has to carry a study suffix or
    the panels silently overwrite each other.
    """
    parts = [part for part in str(study_uid).split(".") if part]
    return parts[-1] if parts else "uid"


def figure_name(entry: dict) -> str:
    stem = Path(entry["output"]).name.replace("_denoised", "")
    return f"{entry['group']}__{entry['case_id']}__{_uid_tail(entry['study_uid'])}__{Path(stem).stem}.png"


def render_volume_case(entry: dict, out_dir: Path) -> Path:
    shape = tuple(int(v) for v in entry["shape"])
    source_path = _input_of(entry)
    denoised_path = Path(entry["output"])
    source = _load(source_path, shape)
    denoised = _load(denoised_path, shape)

    n, h, w = shape
    lo, hi = _clim(source)
    views = [
        ("axial z=%d" % (n // 2), source[n // 2], denoised[n // 2]),
        ("coronal y=%d" % (h // 2), source[:, h // 2, :], denoised[:, h // 2, :]),
        ("sagittal x=%d" % (w // 2), source[:, :, w // 2], denoised[:, :, w // 2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for column, (title, before, after) in enumerate(views):
        for row, (image, label) in enumerate(((before, "input"), (after, "denoised"))):
            ax = axes[row][column]
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"{label} | {title}", fontsize=11)
            ax.axis("off")

    fig.suptitle(
        f"{entry['group']} | {entry['case_id']} | {source_path.name}\n"
        f"sigma {entry['input']['sigma_immerkaer']:.5f} -> "
        f"{entry['denoised']['sigma_immerkaer']:.5f} "
        f"({entry['sigma_drop_pct']:+.1f}%)   "
        f"tissue mean shift {entry['tissue_shift_rel_pct']:+.2f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / figure_name(entry)
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination


def render_noise_summary(records: list[dict], out_dir: Path) -> Path:
    groups = sorted({entry["group"] for entry in records})
    domains = sorted({entry["domain"] for entry in records})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    width = 0.8 / max(1, len(domains))
    for index, domain in enumerate(domains):
        positions, values = [], []
        for group_index, group in enumerate(groups):
            entries = [e for e in records if e["group"] == group and e["domain"] == domain]
            if not entries:
                continue
            positions.append(group_index + index * width - 0.4 + width / 2)
            values.append(float(np.mean([e["sigma_drop_pct"] for e in entries])))
        axes[0].bar(positions, values, width=width, label=domain)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("Immerkaer noise drop [%]")
    axes[0].set_title("Noise reduction by group and domain")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    shifts = [e["tissue_shift_rel_pct"] for e in records]
    labels = [f"{e['group'].split('_')[0]}\n{e['domain'].replace('.raw', '')}" for e in records]
    colors = ["#d62728" if abs(value) > 1.0 else "#1f77b4" for value in shifts]
    axes[1].bar(range(len(shifts)), shifts, color=colors)
    axes[1].set_xticks(range(len(shifts)))
    axes[1].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[1].set_ylabel("tissue mean shift [%]")
    axes[1].set_title("Tissue mean shift (red = |shift| > 1%)")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "summary_noise_drop.png"
    fig.savefig(destination, dpi=130)
    plt.close(fig)

    print(f"{'group':14s} {'domain':22s} {'n':>3s} {'sigma_in':>10s} {'sigma_out':>10s} {'drop%':>8s} {'shift%':>8s}")
    for group in groups:
        for domain in domains:
            entries = [e for e in records if e["group"] == group and e["domain"] == domain]
            if not entries:
                continue
            sigma_in = float(np.mean([e["input"]["sigma_immerkaer"] for e in entries]))
            sigma_out = float(np.mean([e["denoised"]["sigma_immerkaer"] for e in entries]))
            drop = float(np.mean([e["sigma_drop_pct"] for e in entries]))
            shift = float(np.mean([e["tissue_shift_rel_pct"] for e in entries]))
            print(f"{group:14s} {domain:22s} {len(entries):3d} {sigma_in:10.5f} "
                  f"{sigma_out:10.5f} {drop:7.1f}% {shift:7.2f}%")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/test11_cross_mode"))
    parser.add_argument("--summary", type=Path,
                        help="defaults to <run-root>/denoise_summary.json")
    parser.add_argument("--figures", type=Path,
                        help="defaults to <run-root>/figures")
    parser.add_argument("--skip-volumes", action="store_true")
    args = parser.parse_args()

    summary_path = args.summary or (args.run_root / "denoise_summary.json")
    figures = args.figures or (args.run_root / "figures")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    records = report["records"]

    if not args.skip_volumes:
        names = [figure_name(entry) for entry in records]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(
                "figure names are not unique; panels would overwrite each other: "
                f"{duplicated}"
            )
        for index, entry in enumerate(records, start=1):
            destination = render_volume_case(entry, figures)
            print(f"[{index}/{len(records)}] {destination.name}", flush=True)

    render_noise_summary(records, figures)
    print(f"figures -> {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
