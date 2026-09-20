"""Render the banded-rounds validation JSON into a Markdown report.

Pure stdlib (no numpy / torch), so it runs anywhere -- including on a laptop,
from the JSON ``scripts/eval_rounds_banded.py`` produced on the GPU box.
Reusable for any round chain: the round columns come from the JSON's
``checkpoints`` mapping, and the numbers are not r3-specific.

    python scripts/report_banded_rounds.py --json outputs/eval_..._banded_test.json \
        --out docs/reports/r3-banded-validation.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(v: float, digits: int = 5) -> str:
    return "nan" if v is None or v != v else f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "nan" if v != v else f"{100.0 * (1.0 - v):+.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description="Render the banded validation JSON as Markdown.")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="递归轮次输出层分带验证")
    args = p.parse_args()

    d = json.loads(args.json.read_text(encoding="utf-8"))
    names = list(d["checkpoints"].keys())
    ref, new = (names[-2], names[-1]) if len(names) >= 2 else (names[0], names[0])
    bands = d["bands"]
    glob = d["global"]
    v = d.get("verdict", {})

    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append(f"- 划分：`{d['split']}`，{d['n_studies']} 例；产物 JSON：`{Path(args.json).as_posix()}`")
    lines.append(f"- 推理：`{d['arch']}`，{'分块 tiled（H/W+ D）' if d.get('tiled') else 'D 窗口'} "
                 f"patch_hw={d['patch_hw']} overlap_d={d['overlap_d']} overlap_hw={d['overlap_hw']}，"
                 f"voxel_depth={d['voxel_depth']}")
    lines.append(f"- 分带键：`{d['band_key']}` 输出（去噪后 μ 值，> tissue_floor={d['tissue_floor']}），"
                 f"lo={d['lo']} / hi={d['hi']}（低=水/软组织，中=过渡，高=骨/牙釉质）")
    lines.append(f"- 检查点：" + "、".join(f"`{n}`={pth}" for n, pth in d["checkpoints"].items()))
    lines.append("")
    lines.append("## 分带 Immerkaer σ（10 例均值）")
    lines.append("")
    lines.append("| 密度带 | " + " | ".join(names) + " | full（含噪） | " + f"{new}/{ref} | {new} vs full |")
    lines.append("| --- |" + " --- |" * (len(names) + 3))
    for band in ("low", "mid", "high"):
        b = bands[band]
        ratio = b[new] / b[ref] if b[ref] else float("nan")
        vs_full = b[new] / b["full"] if b["full"] else float("nan")
        lines.append(f"| {band} | " + " | ".join(_fmt(b[n]) for n in names) + f" | {_fmt(b['full'])} "
                     f"| {_fmt(ratio, 3)}x | {_pct(vs_full)} |")
    ratio_g = glob[new] / glob[ref] if glob[ref] else float("nan")
    lines.append(f"| **全局** | " + " | ".join(_fmt(glob[n]) for n in names) + f" | {_fmt(glob['full'])} "
                 f"| {_fmt(ratio_g, 3)}x | {_pct(glob[new] / glob['full'])} |")
    lines.append("")
    lines.append("## 判据")
    lines.append("")
    low_r = v.get("low_ratio_new_over_ref", float("nan"))
    high_r = v.get("high_ratio_new_over_ref", float("nan"))
    mid_r = v.get("mid_ratio_new_over_ref", float("nan"))
    lines.append(f"1. **软组织封顶**（{new}/{ref} ≤ 1.02）：{_fmt(low_r, 3)}x → "
                 f"{'✅ 成立' if v.get('soft_tissue_capped') else '❌ 不成立'}"
                 f"（软组织未越过 {ref}，≤2% 视为持平）")
    lines.append(f"2. **骨区超越**（{new}/{ref} < 1）：{_fmt(high_r, 3)}x → "
                 f"{'✅ 成立' if v.get('bone_pushed_past_ref') else '❌ 不成立'}"
                 f"（骨/牙釉质在 {ref} 上再递归一层）")
    lines.append(f"3. **全带低于 full**：{'✅ 成立' if v.get('all_bands_below_full') else '❌ 不成立'}"
                 f"（去噪方向正确）")
    lines.append("")
    lines.append(f"- 过渡带 {new}/{ref} = {_fmt(mid_r, 3)}x（smoothstep 混合的中间态）")
    lines.append(f"- 全局：{new} 相对 {ref} {_fmt(ratio_g, 3)}x；相对 full 削减 "
                 f"{_fmt(v.get('global_reduction_vs_full_pct', float('nan')), 1)}%")
    lines.append(f"- **门控结论：{'PASS' if v.get('gate_pass') else 'FAIL'}**")
    lines.append("")
    lines.append("## 逐例明细（比值）")
    lines.append("")
    lines.append(f"| # | case | group | {ref}→{new} low | mid | high | {new} vs full（全局） |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    n_low = n_high = n_all = 0
    for i, row in enumerate(d["per_case"], 1):
        b = row["bands"]
        lr = b["low"][new] / b["low"][ref] if b["low"][ref] else float("nan")
        mr = b["mid"][new] / b["mid"][ref] if b["mid"][ref] else float("nan")
        hr = b["high"][new] / b["high"][ref] if b["high"][ref] else float("nan")
        gr = row["global"][new] / row["global"]["full"] if row["global"]["full"] else float("nan")
        n_low += lr <= 1.02
        n_high += hr < 1.0
        n_all += all(b[bd][new] < b[bd]["full"] for bd in ("low", "mid", "high"))
        short = row["key"].split("::")[0].split(".")[-1]
        lines.append(f"| {i} | …{short} | {row.get('group')} | {_fmt(lr, 3)} | {_fmt(mr, 3)} "
                     f"| {_fmt(hr, 3)} | {_pct(gr)} |")
    lines.append("")
    n = len(d["per_case"])
    lines.append(f"**一致性**：软组织封顶 {n_low}/{n} 例、骨区超越 {n_high}/{n} 例、"
                 f"全带低于 full {n_all}/{n} 例满足。")
    lines.append("")
    lines.append("> 说明：这里是**输出层**判据（模型输出的实测噪声）。目标层的分布检查见 "
                 "`blend_dn_targets.py` 的 `blend_full_*_overview.json`——两者测的是不同的东西，"
                 "因为 half→full 训练下学生输出比监督目标再干净一层。")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
