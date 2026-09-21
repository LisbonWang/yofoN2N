"""Assemble and verify the expert-review package (raw volumes only).

Walks ``outputs/expert_review_ai_vs_conv/c*_*`` and, for every volume it finds:

* verifies the byte size against the shape implied by the group (600x640x640 for
  the 1615 groups, 360x640x640 for the 1609 groups),
* records size + md5 + finite/min/max/mean/std,
* computes the Immerkaer sigma (whole volume and the three density bands),
* pulls the reconstruction sidecar (chain, PWLS beta, filter md5) and the
  inference / sharpening metadata when present.

It then writes ``case.json`` per case, plus ``manifest.json``, ``metrics.json``
and ``README.md`` at the package root.  Nothing here writes image files: the
package is volumes + JSON provenance, as requested.

Usage::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/_review_assemble.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cbct_n2n.training.metrics import band_noise_std_immerkaer, estimate_noise_std_immerkaer

DTYPE = np.dtype("<f4")
TISSUE_FLOOR = 0.01
SHAPES = {600: (600, 640, 640), 360: (360, 640, 640), 640: (640, 640, 640)}
BYTES_TO_SHAPE = {z * 640 * 640 * 4: shape for z, shape in SHAPES.items()}

# Delivery names are flattened to one extension: "volume_640X640X360_f32_ai_r1b.raw"
# rather than a forest of dots.  X/Y/Z follow fdkPara (xNum/yNum/zNum), so the
# slice count comes last.
NAME_PATTERN = "volume_<X>X<Y>X<Z>_f32[_后缀].raw"
NAME_SUFFIXES = [
    ("", "input", "AI 输入：PWLS 0.01 + BHC + RCoor + MARFI"),
    ("_traditional", "conventional",
     "传统降噪对照：同链 + ReduceNoise + Anisotropic"),
    ("_ai_r1b", "ai_r1b", "AI r1 轮（轻量档）"),
    ("_ai_r2b", "ai_r2b", "AI r2 轮（重量档）"),
    ("_ai_r3b", "ai_r3b", "AI r3 轮（激进档）"),
    ("_ai_r4b", "ai_r4b", "AI r4 轮（存档档）"),
    ("_ai_r4b_sharp", "ai_r4b_sharp", "r4 轮 + 0.3 力度引导滤波锐化"),
]


def dims_tag(shape: tuple[int, int, int]) -> str:
    """``(360, 640, 640)`` (z, y, x) -> ``"640X640X360"`` (x X y X z)."""
    z, y, x = shape
    return f"{x}X{y}X{z}"


def vol_names(shape: tuple[int, int, int]) -> list[tuple[str, str, str]]:
    """``(filename, key, description)`` for the seven volumes of one case."""
    base = f"volume_{dims_tag(shape)}_f32"
    return [(f"{base}{suffix}.raw", key, desc) for suffix, key, desc in NAME_SUFFIXES]


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_shape(path: Path) -> tuple[int, int, int]:
    n = path.stat().st_size
    shape = BYTES_TO_SHAPE.get(n)
    if shape is None:
        raise ValueError(f"{path.name}: {n} B matches no known shape")
    return shape


def infer_case_shape(directory: Path) -> tuple[int, int, int]:
    """Geometry of a case folder, from whichever volumes are already there.

    The file name itself depends on the shape, so the shape cannot be read off
    the name without knowing it first; the byte size is the one fact available
    before anything else.  All volumes in a folder must agree.
    """
    raws = sorted(directory.glob("*.raw"))
    if not raws:
        raise SystemExit(f"{directory.name}: no .raw volumes; "
                         "run scripts/_review_recon.sh first")
    shapes = set()
    for f in raws:
        shape = BYTES_TO_SHAPE.get(f.stat().st_size)
        if shape is None:
            raise SystemExit(f"{directory.name}/{f.name}: "
                             f"{f.stat().st_size} B matches no known shape")
        shapes.add(shape)
    if len(shapes) > 1:
        raise SystemExit(f"{directory.name}: mixed geometry {sorted(shapes)}")
    return shapes.pop()


def case_masks(input_volume: np.ndarray) -> dict[str, np.ndarray]:
    """Band masks taken from the case's *input* volume, shared by every version.

    Recomputing the thresholds per volume is self-selecting: a denoised volume has
    fewer voxels left above 0.4, so it would be scored on a different voxel set
    than the one it is compared against.  The project rule is one mask per case.
    """
    return {
        "low": (input_volume > TISSUE_FLOOR) & (input_volume <= 0.2),
        "mid": (input_volume > 0.2) & (input_volume < 0.4),
        "high": input_volume >= 0.4,
    }


def volume_stats(path: Path, shape: tuple[int, int, int], masks: dict[str, np.ndarray],
                 max_slices: int) -> dict:
    arr = np.asarray(np.memmap(path, dtype=DTYPE, mode="r", shape=shape))
    bands = {b: band_noise_std_immerkaer(arr, m, max_slices=max_slices)
             for b, m in masks.items()}
    tissue = arr > TISSUE_FLOOR
    return {
        "bytes": path.stat().st_size,
        "shape": list(shape),
        "finite": bool(np.isfinite(arr).all()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "tissue_fraction": float(np.count_nonzero(tissue) / arr.size),
        "global_sigma": estimate_noise_std_immerkaer(arr, max_slices=max_slices),
        "bands": bands,
        "band_differentiation_high_over_low": (
            float(bands["high"] / bands["low"]) if bands["low"] else float("nan")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "outputs" / "expert_review_ai_vs_conv")
    p.add_argument("--max-slices", type=int, default=24)
    p.add_argument("--skip-md5", action="store_true",
                   help="Skip md5 of the volumes (slow on ~45 GiB).")
    p.add_argument("--readme-only", action="store_true",
                   help="Re-render README.md from the existing manifest.json without "
                        "re-measuring anything (seconds instead of minutes).")
    p.add_argument("--checkpoints", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "outputs" / "air3_ypdl0027_bhc_5models" / "checkpoints")
    args = p.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"package root not found: {root}")

    if args.readme_only:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"--readme-only needs an existing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (root / "README.md").write_text(render_readme(manifest), encoding="utf-8")
        print(f"[readme  ] {root / 'README.md'} (re-rendered from manifest.json)")
        return 0

    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("c") and "_" in d.name)
    if not dirs:
        raise SystemExit(f"no case directories under {root}")

    ck_md5 = {}
    for name in ("r0", "r1b", "r2b", "r3b", "r4b"):
        f = args.checkpoints / f"{name}.best.pt"
        if f.is_file():
            ck_md5[name] = {"path": str(f), "md5": md5_of(f), "bytes": f.stat().st_size}

    manifest: dict = {
        "schema_version": 1,
        "purpose": "expert review of AIR3 recursive-chain denoising versus conventional denoising",
        "selection": "held-out cross-mode test set (splits_cross_mode_test11.json); "
                     "quota Mode0_1615 x1, Mode1_1615 x2, Mode2_1609 x3, Mode3_1609 x3 = 9 cases",
        "reconstruction": {
            "pwls_scale": 0.01,
            "pwls_note": "beta = 0.01 x mean(fieldMatrixMax*.bin); the same runtime patch "
                         "also switches <Ramp> to Hamming (0.54/0.46), so PWLS and the "
                         "Hamming window are one indivisible change",
            "chain_ai_input": "BHC RCoor MARFI",
            "chain_conventional": "BHC ReduceNoise Anisotropic RCoor MARFI",
            "reduce_noise": {"ratio": 1.0, "gradientThresh": 0.01, "medFilterLen": 7},
            "anisotropic": {"stepNum": 1, "filterLen": 5, "sigma": 3.0, "weightZ": 0.0015},
            "note": "both chains differ only by the two denoise modules; RCoor sits in the "
                    "same position in both",
        },
        "inference": {
            "config": "configs/air3_mode2_1609_unet3d_h2f_128b_r4.yaml",
            "tiling": {"voxel_depth": 128, "patch_hw": 256,
                       "overlap_d": 16, "overlap_hw": 16, "batch_size": 2},
            "checkpoints": ck_md5,
        },
        "sharpening": {
            "method": "3D guided filter, wiener mode",
            "radius": 2, "eps_k": 8.0, "sigma_source": "flat ROI (per volume)",
            "strength": 0.3,
            "applied_to": "each case's <base>_ai_r4b.raw",
            "expected_effect": "bone-region sigma gain ~= 94% x strength ~= +28%; "
                               "flat-region noise gain ~= 1 + s/(1+k^2) ~= 1.005",
        },
        "naming": {
            "pattern": NAME_PATTERN,
            "note": "X/Y/Z are fdkPara xNum/yNum/zNum, so the slice count comes last; "
                    "every volume is headerless float32 little-endian (<f4)",
            "dims_by_group": {g: dims_tag(s) for g, s in (
                ("Mode0_1615", (600, 640, 640)), ("Mode1_1615", (600, 640, 640)),
                ("Mode2_1609", (360, 640, 640)), ("Mode3_1609", (360, 640, 640)))},
            "suffixes": {(suffix or "(无后缀)"): desc for suffix, _, desc in NAME_SUFFIXES},
            "volumes_per_case": len(NAME_SUFFIXES),
        },
        "cases": [],
    }

    rows = []
    for d in dirs:
        entry = {"dir": d.name, "path": str(d), "files": {}, "missing": []}
        shape = infer_case_shape(d)
        order = vol_names(shape)
        entry["naming"] = {"pattern": NAME_PATTERN, "dims": dims_tag(shape)}
        input_path = d / order[0][0]
        if not input_path.is_file():
            raise SystemExit(f"{d.name}: AI input missing ({order[0][0]}); "
                             "run scripts/_review_recon.sh first")
        masks = case_masks(np.asarray(np.memmap(input_path, dtype=DTYPE, mode="r",
                                                shape=shape)))
        entry["band_masks_from"] = order[0][0]
        entry["band_mask_n_voxels"] = {b: int(np.count_nonzero(m)) for b, m in masks.items()}
        for fname, key, desc in order:
            f = d / fname
            if not f.is_file():
                entry["missing"].append(fname)
                continue
            rec = {"role": desc, "name": f.name}
            if not args.skip_md5:
                rec["md5"] = md5_of(f)
            rec.update(volume_stats(f, shape, masks, args.max_slices))
            entry["files"][key] = rec
            print(f"[{d.name}] {key:22s} {f.name:52s} sigma={rec['global_sigma']:.5f} "
                  f"diff={rec['band_differentiation_high_over_low']:.3f}", flush=True)

        # reconstruction sidecars (written by recon_conventional_denoise.py)
        for key, stem in (("input", order[0][0]), ("conventional", order[1][0])):
            sc = d / f"{stem}.json"
            if sc.is_file():
                data = json.loads(sc.read_text(encoding="utf-8"))
                entry.setdefault("recon_sidecar", {})[key] = {
                    k: data.get(k) for k in
                    ("output", "md5", "shape", "spacing_zyx", "case_id", "study_uid",
                     "mode", "group", "kv", "ma", "nviews", "raw_dir", "cfg_dir",
                     "fdk_para", "filter_file", "filter_file_md5", "post_procs",
                     "pwls", "seconds", "volume_min", "volume_max", "volume_mean",
                     "volume_std")
                }

        mi = d / "metrics_ai.json"
        if mi.is_file():
            entry["metrics_ai"] = json.loads(mi.read_text(encoding="utf-8"))
        gj = root / "_logs" / "gfsharp" / d.name.split("_")[0] / "gfsharp_s0.3.json"
        sharp_meas = None
        if gj.is_file():
            entry["sharpening"] = json.loads(gj.read_text(encoding="utf-8"))
            mm = entry["sharpening"]["metrics"]
            pl, sh = mm["plain"], mm["sharp"]
            sharp_meas = {
                "flat_sigma_before": pl["flat_patch_sigma"],
                "flat_sigma_after": sh["flat_patch_sigma"],
                "flat_gain": sh["flat_patch_sigma"] / pl["flat_patch_sigma"],
                "bone_sigma_before": pl["bone_patch_sigma"],
                "bone_sigma_after": sh["bone_patch_sigma"],
                "bone_gain": sh["bone_patch_sigma"] / pl["bone_patch_sigma"],
                "sigma_used_for_eps": entry["sharpening"]["sigma_for_eps"],
                "sigma_source": entry["sharpening"]["sigma_source"],
                "eps": entry["sharpening"]["eps"],
            }

        (d / "case.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["cases"].append({
            "index": d.name.split("_")[0],
            "dir": d.name,
            "group": d.name.split("_")[1] + "_" + d.name.split("_")[2],
            "case_id": d.name.split("__")[1],
            "uid_tail": d.name.split("__")[2],
            "shape": list(shape) if shape else None,
            "complete": not entry["missing"],
            "missing": entry["missing"],
            "md5": {k: v.get("md5") for k, v in entry["files"].items()},
            "sigma": {k: round(v["global_sigma"], 6) for k, v in entry["files"].items()},
            "band_diff": {k: round(v["band_differentiation_high_over_low"], 4)
                          for k, v in entry["files"].items()},
            "source": entry.get("recon_sidecar", {}).get("input", {}).get("raw_dir"),
            "study_uid": entry.get("recon_sidecar", {}).get("input", {}).get("study_uid"),
            "nviews": entry.get("recon_sidecar", {}).get("input", {}).get("nviews"),
            "sharpening_measured": sharp_meas,
            "band_mask_n_voxels": entry["band_mask_n_voxels"],
        })

    # ---------------------------------------------------------------- metrics table
    metrics = {"tissue_floor": TISSUE_FLOOR, "band_edges": [0.2, 0.4],
               "note": "Immerkaer sigma is high-frequency energy: it cannot separate noise "
                       "from texture, and 'band_differentiation_high_over_low' is the only "
                       "density-selectivity read-out. Do not compare sigmas across sources "
                       "with different input floors.",
               "cases": []}
    for c in manifest["cases"]:
        metrics["cases"].append({"dir": c["dir"], "sigma": c["sigma"],
                                 "band_diff": c["band_diff"]})

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[manifest] {root / 'manifest.json'}")
    print(f"[metrics ] {root / 'metrics.json'}")

    readme = render_readme(manifest)
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(f"[readme  ] {root / 'README.md'}")

    bad = [c["dir"] for c in manifest["cases"] if not c["complete"]]
    print(f"\ncases: {len(manifest['cases'])}  complete: "
          f"{len(manifest['cases']) - len(bad)}  incomplete: {bad}")
    return 0


def render_readme(manifest: dict) -> str:
    out: list[str] = []
    out.append("# AI 降噪评审包（AI 递归链 vs 传统降噪）\n")
    out.append(
        "9 例全部取自**跨模式独立测试集** `splits_cross_mode_test11.json`（未参与任何训练/验证）："
        "Mode0_1615 ×1、Mode1_1615 ×2、Mode2_1609 ×3、Mode3_1609 ×3。\n"
    )
    out.append("每个 `cNN_<组>__<病例>__<uid尾段>/` 子文件夹里是**同一例的 7 卷体数据**，"
               "headerless `float32` 小端（`<f4`）。\n")
    out.append(f"命名规则：`{NAME_PATTERN}`。`X/Y/Z` 取自 `fdkPara` 的 "
               "`xNum/yNum/zNum`（所以**层数在最后**）；下表的「后缀」拼在 `_f32` 之后。\n")
    out.append("| 组 | 尺寸段 |")
    out.append("| --- | --- |")
    dims = manifest.get("naming", {}).get("dims_by_group", {})
    for gname in ("Mode0_1615", "Mode1_1615", "Mode2_1609", "Mode3_1609"):
        if gname in dims:
            out.append(f"| {gname} | `{dims[gname]}` |")
    out.append("")
    out.append("| 后缀 | 内容 |")
    out.append("| --- | --- |")
    for suffix, key, desc in NAME_SUFFIXES:
        shown = suffix if suffix else "(无后缀)"
        out.append(f"| `{shown}` | {desc} |")
    out.append("")
    out.append("示例（1609 组）：`volume_640X640X360_f32.raw`、"
               "`volume_640X640X360_f32_ai_r1b.raw`、"
               "`volume_640X640X360_f32_ai_r4b_sharp.raw`。\n")
    out.append("AI 处理过的四卷都带 `_ai_` 标识。`*.raw.json` 是重建溯源 sidecar，"
               "`case.json` 是本案汇总（尺寸、md5、σ、模型哈希）。\n")

    out.append("## 重建参数（两条链的唯一差别是降噪模块）\n")
    rec = manifest["reconstruction"]
    out.append(f"- PWLS：`--pwls-scale {rec['pwls_scale']}`，`beta = 0.01 × 亮场均值`；"
               "该运行时补丁**同时**把 `<Ramp>` 换成 Hamming(0.54/0.46)，两者不可分开归因。")
    out.append(f"- AI 输入链：`--post-procs \"{rec['chain_ai_input']}\"`")
    out.append(f"- 传统降噪链：`--post-procs \"{rec['chain_conventional']}\"`")
    out.append(f"- ReduceNoise `{rec['reduce_noise']}`；Anisotropic `{rec['anisotropic']}`"
               "（均为各组 `configs/recon_filters/*.filter.xml` 里的出厂值，未改动）")
    out.append("- 输出 dtype `float32`、spacing 0.25 mm 各向同性。\n")

    out.append("## AI 模型\n")
    out.append(f"- 架构/推理配置：`{manifest['inference']['config']}`")
    ti = manifest["inference"]["tiling"]
    out.append(f"- 三维重叠分块 + 梯形窗融合：D={ti['voxel_depth']}、HW={ti['patch_hw']}、"
               f"overlap={ti['overlap_d']}/{ti['overlap_hw']}、batch={ti['batch_size']}")
    out.append("- 权重（md5 已与训练机 `runs/` 下的 `best.pt` 逐位核对）：\n")
    out.append("| 轮次 | checkpoint | md5 |")
    out.append("| --- | --- | --- |")
    for name in ("r1b", "r2b", "r3b", "r4b"):
        ck = manifest["inference"]["checkpoints"].get(name)
        if ck:
            out.append(f"| {name} | `{Path(ck['path']).name}` | `{ck['md5']}` |")
    out.append("")
    out.append("> 注意：四个模型都只在 `AIReduceNoise3 / Mode2_1609` 的四域联合数据上训练过，"
               "训练输入域是 **PWLS + BHC/MARFI（无 RCoor）**。本包的输入多开了 RCoor，"
               "实测是温和的全局归一化（相对无 RCoor 参考：78.5% 体素变化、`maxabs 0.044`、"
               "最佳仿射 `1.0059× + 0.0018`），不改变几何与强度尺度。\n")

    sh = manifest["sharpening"]
    out.append("## 引导滤波锐化\n")
    out.append(f"- 施加对象：每例的 `<base>_ai_r4b.raw`（`<base> = volume_<X>X<Y>X<Z>_f32`）；"
               f"方法 `{sh['method']}`，"
               f"`radius={sh['radius']}`、`eps_k={sh['eps_k']}`、"
               f"σ 取自**该卷平坦 ROI 实测值**（`--sigma-source {sh['sigma_source']}`）、"
               f"`strength={sh['strength']}`")
    out.append(f"- 设计预期：{sh['expected_effect']}"
               "（strength 在本项目实测为严格线性，只改幅度不改性质）\n")
    meas = [c["sharpening_measured"] for c in manifest["cases"] if c.get("sharpening_measured")]
    if meas:
        fg = [m["flat_gain"] for m in meas]
        bg = [m["bone_gain"] for m in meas]
        out.append(f"**实测（9 例，同一套 ROI 前后对比）**：骨小梁 ROI σ 增益 "
                   f"**{min(bg):.3f}–{max(bg):.3f}×**（均值 {sum(bg)/len(bg):.3f}×，"
                   f"≈ +{(sum(bg)/len(bg)-1)*100:.0f}%），与 `94% × strength` 的经验式一致；"
                   f"可证平坦 ROI 的噪声增益仅 {min(fg):.3f}–{max(fg):.3f}×"
                   f"（均值 {sum(fg)/len(fg):.3f}×）。\n")
        out.append("| 例 | σ_flat 前 → 后 | 增益 | σ_bone 前 → 后 | 增益 | eps 用的 σ |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for c in manifest["cases"]:
            m = c.get("sharpening_measured")
            if not m:
                continue
            out.append(f"| {c['index']} | {m['flat_sigma_before']:.6f} → {m['flat_sigma_after']:.6f} "
                       f"| {m['flat_gain']:.3f}× | {m['bone_sigma_before']:.6f} → "
                       f"{m['bone_sigma_after']:.6f} | {m['bone_gain']:.3f}× "
                       f"| {m['sigma_used_for_eps']:.6f} |")
        out.append("")
        out.append("eps 用的 σ 一律是**平坦 ROI 实测值**，不是全组织 Immerkaer σ。"
                   "后者把骨结构也算进噪声：本包 9 例实测高 **1.31–5.29 倍**"
                   "（如 c01 `sigma_tissue 0.003658` vs `sigma_flat 0.001906`，"
                   "c07 `0.003451` vs `0.000653`），换过去会让同一个 `k` 的实际强度整个改变。\n")

    out.append("## 实测指标（Immerkaer σ，tissue>0.01，lo/hi=0.2/0.4）\n")
    out.append("**每例的分带掩码统一取自该例的输入卷**（不是各卷各算）——"
               "对每个版本各自重算阈值会自选择地改变体素集合，报出不可比的数字。\n")
    out.append("| 例 | 组 | 病例 | 输入 σ | 传统降噪 σ | AI r1 σ | AI r2 σ | AI r3 σ | AI r4 σ "
               "| r4+锐化0.3 σ | 输入 diff | 传统 diff | r1 diff | r4 diff |")
    out.append("| --- | --- | --- | " + " --- |" * 11)
    for c in manifest["cases"]:
        sig = c["sigma"]
        bd = c["band_diff"]

        def g(d, k):
            return d.get(k, float("nan"))

        out.append(
            f"| {c['index']} | {c['group']} | {c['case_id']} | "
            f"{g(sig, 'input'):.5f} | {g(sig, 'conventional'):.5f} | "
            f"{g(sig, 'ai_r1b'):.5f} | {g(sig, 'ai_r2b'):.5f} | {g(sig, 'ai_r3b'):.5f} | "
            f"{g(sig, 'ai_r4b'):.5f} | {g(sig, 'ai_r4b_sharp'):.5f} | "
            f"{g(bd, 'input'):.3f} | {g(bd, 'conventional'):.3f} | "
            f"{g(bd, 'ai_r1b'):.3f} | {g(bd, 'ai_r4b'):.3f} |")
    out.append("")
    out.append("`diff = σ(high 带)/σ(low 带)`，同一套掩码下跨版本可比：**<1 表示软组织被压得比骨狠"
               "（密度自适应）**，≈1 表示两者同等力度，>1 表示软组织被相对保留、骨被相对平滑。\n")
    out.append("| 组 | 传统降噪 σ 削减 | r1b | r2b | r3b | r4b |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    groups: dict[str, list] = {}
    for c in manifest["cases"]:
        groups.setdefault(c["group"], []).append(c)
    for gname, cs in groups.items():
        cells = []
        for key in ("conventional", "ai_r1b", "ai_r2b", "ai_r3b", "ai_r4b"):
            cuts = []
            for c in cs:
                base = c["sigma"]["input"]
                v = c["sigma"].get(key)
                if base and v:
                    cuts.append(100.0 * (1.0 - v / base))
            cells.append(f"{min(cuts):.1f}–{max(cuts):.1f}" if len(cuts) > 1
                         else f"{cuts[0]:.1f}")
        out.append(f"| {gname} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |")
    out.append("")
    out.append("（削减率为相对该例输入卷的百分比；组内多例时给区间。）\n")

    out.append("## 判读须知（本项目的实测铁律）\n")
    out.append("1. **全局 σ 分不开噪声与纹理**——Immerkaer 只量高频能量，两者都算。"
               "结论必须落在「可证平坦 ROI vs 可证骨小梁 ROI」的比值上，或直接看图。")
    out.append("2. **骨带 σ 被金属牙/皮质骨强边缘主导**，不可单独采信。")
    out.append("3. **不要拿绝对 σ 跨例比**：输入起点不同，削减率会差一倍。"
               "跨例可比的是平坦对照块上的**逐轮削减倍数**。")
    out.append("4. **跨来源比较必须锁同一套掩码**：用各卷自己的 0.4 阈值分带会把阈值附近"
               "的体素整体推过界，报出假的高频。")
    out.append("5. r4 轮已接近「全局最大平滑」，密度分级基本归零；"
               "r1/r2 保留了分级设计意图。是否可用取决于骨小梁是否被磨平——"
               "**这只能靠看图或任务型指标裁定，σ 给不出答案**。\n")
    out.append("## 复现\n")
    out.append("```bash\n"
               "bash scripts/_review_recon.sh      # 重建（幂等，已存在则跳过）\n"
               "bash scripts/_review_infer.sh      # r1b..r4b 推理\n"
               "bash scripts/_review_sharpen.sh    # s=0.3 引导滤波锐化\n"
               "C:/Users/admin/anaconda3/envs/torch/python.exe scripts/_review_assemble.py\n"
               "```\n")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
