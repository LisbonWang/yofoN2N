"""Historical AIReduceNoise figure-generation utility with fixed manifests.

Render side-by-side denoising comparison figures for selected test cases.

For each (config, checkpoint, manifest, case-key) tuple this loads the
even-view source, runs whole-volume 3D inference, and writes a 3x3 axial
mosaic (rows = z slices, cols = even input / denoised / full reference)
to ``outputs/eval_figures/``.  The greyscale window is fixed from the
full-reference percentiles so the three columns are directly comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import torch

import train_voxel
from cbct_n2n.config import load_config
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import evaluate_volume

import test_raw_3d as t3


def _load_case(manifest_path: Path, key: str):
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recmap = {str(r["key"]): r for r in m["records"]}
    if key not in recmap:
        matches = [k for k in recmap if k.startswith(key + "::")]
        if len(matches) != 1:
            raise KeyError(f"key {key!r} matched {len(matches)} records")
        key = matches[0]
    rec = recmap[key]
    base = Path(manifest_path).parent
    full = t3._load_raw_volume(base, rec, "full")
    even = t3._load_raw_volume(base, rec, "even")
    if full is None or even is None:
        raise RuntimeError(f"missing volumes for {key}")
    return rec, np.asarray(even), np.asarray(full)


def make_figure(config_path, ckpt_path, manifest_path, key, out_path, title):
    config = load_config(Path(config_path))
    voxel_depth = int(config["data"].get("voxel_depth", 16))
    data_range = float(config["data"].get("metric_data_range", 4.0))
    arch = config.get("model", {}).get("arch", "unet3d")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = train_voxel.build_model(arch, config["model"]).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    _, even, full = _load_case(manifest_path, key)
    denoised = t3.infer_volume_3d(model, even, voxel_depth, device, batch_size=2)
    metrics = evaluate_volume(denoised, full, data_range)

    vmin, vmax = np.percentile(full, [0.5, 99.5])
    zs = [90, 180, 270]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    for r, z in enumerate(zs):
        for c, (img, name) in enumerate(
            [(even, "even input"), (denoised, "denoised"), (full, "full ref")]
        ):
            axes[r, c].imshow(img[z], cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"{name}  z={z}", fontsize=10)
            axes[r, c].axis("off")
    fig.suptitle(
        f"{title}\nPSNR={metrics['psnr']:.2f} dB  SSIM={metrics['ssim']:.4f}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


CASES = [
    # (tag, config, checkpoint, manifest, case key)
    (
        "data2_h2f_Mode2",
        "configs/1609v2_unet3d_h2f.yaml",
        "outputs/eval_checkpoints/best_1609v2_h2f.pt",
        "Z:/raid0/yyc/AIReduceNoise2/splits_1609v2.json",
        "1.2.156.89797079.3901595658.1272365966.3427696804",
        "data2 base16 h2f - Mode2",
    ),
    (
        "data2_h2f_Mode3",
        "configs/1609v2_unet3d_h2f.yaml",
        "outputs/eval_checkpoints/best_1609v2_h2f.pt",
        "Z:/raid0/yyc/AIReduceNoise2/splits_1609v2.json",
        "1.2.156.89797079.1997082915.1324480055.822784412",
        "data2 base16 h2f - Mode3",
    ),
    (
        "data1_h2h_Mode2",
        "configs/1609_unet3d_base8.yaml",
        "outputs/eval_checkpoints/best_1609_base8.pt",
        "Z:/raid0/yyc/AIReduceNoise/splits_1609.json",
        "1.2.156.89797079.1121860508.1155729735.2306050440",
        "data1 base8 h2h - Mode2",
    ),
    (
        "data1_h2h_Mode3",
        "configs/1609_unet3d_base8.yaml",
        "outputs/eval_checkpoints/best_1609_base8.pt",
        "Z:/raid0/yyc/AIReduceNoise/splits_1609.json",
        "1.2.826.0.1.3680043.2.461.16112399.2216418224",
        "data1 base8 h2h - Mode3",
    ),
]

if __name__ == "__main__":
    out_dir = Path("outputs/eval_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, cfg, ckpt, mf, key, title in CASES:
        make_figure(cfg, ckpt, mf, key, out_dir / f"{tag}.png", title)
    print("done", flush=True)
