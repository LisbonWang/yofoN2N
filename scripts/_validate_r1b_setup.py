"""Validate the AIReduceNoise3 r1b config/dataset wiring before the chain is launched."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbct_n2n.config import load_config  # noqa: E402

CFG = Path("configs/air3_mode2_1609_unet3d_h2f_128b_r1.yaml")
R0 = Path("configs/air3_mode2_1609_unet3d_h2f_128_r0.yaml")

cfg = load_config(CFG)
r0 = load_config(R0)
data = cfg["data"]
manifest_path = Path(data["manifest"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
base = manifest_path.parent

print(f"mappings      : {data['mappings']}")
print(f"residuals     : {data.get('residuals')}")
print(f"voxel/stride/patch: {data.get('voxel_depth')}/{data.get('stride')}/{data.get('patch_size')}")
print(f"manifest      : {manifest_path}")
print(f"train/val/test: {len(manifest['splits']['train'])}/{len(manifest['splits']['val'])}/"
      f"{len(manifest['splits'].get('test', []))}")
print(f"training      : device={cfg['training']['device']} patience={cfg['training']['patience']} "
      f"run_dir={cfg['training']['run_dir']} epochs={cfg['training']['epochs']}")

# control variable: r1b must differ from r0 only in supervision target + run_dir
diff = []
for section in ("model", "training"):
    for key, value in r0[section].items():
        cur = cfg[section].get(key)
        if cur != value and not (key == "run_dir"):
            diff.append(f"{section}.{key}: r0={value} r1b={cur}")
for key, value in r0["data"].items():
    if key in ("mappings", "residuals"):
        continue
    if cfg["data"].get(key) != value:
        diff.append(f"data.{key}: r0={value} r1b={cfg['data'].get(key)}")
print(f"unexpected r0<->r1b diffs: {diff if diff else 'none (only mappings+run_dir differ)'}")

shape = tuple(int(v) for v in manifest["records"][0]["shape"])
key0 = str(manifest["records"][0]["key"])
record = next(r for r in manifest["records"] if str(r["key"]) == key0)
target = f"full.{data['mappings'][0].split('->')[1].split('.', 1)[1]}"
print(f"target key    : {target}")
print(f"registered?   : {target + '.raw' in record['outputs']} "
      f"({record['outputs'].get(target + '.raw')})")
print(f"shape x4 bytes: {shape} -> {shape[0]*shape[1]*shape[2]*4:,} B/volume")
print(f"residual vols : {[k for k in record['outputs'] if k.startswith('full.')]}")
