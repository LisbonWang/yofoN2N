"""Pre-launch validation for the AIR3 r2b..r4b chain (run on the training host).

Checks, without touching any data:
  1. the three new round configs parse and are control variables of r0
     (only ``data.mappings`` + ``training.run_dir`` may differ),
  2. the chain's round loop resolves the right teacher checkpoint for each round,
  3. the manifest already carries the r1 target key (so r2's teacher input is real),
  4. the updated prepare/gate modules import and the gate runs in prep-only mode
     against the *real* r0 overview (legacy overview without ``rows``).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# the script is copied to /tmp on the training host, so resolve the repo explicitly
ROOT = Path(os.environ.get("N2N_ROOT", "/home/yyc/workspace/N2N"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

CONFIGS = ROOT / "configs"
MANIFEST = Path("/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json")
AIR3 = Path("/data/raid0/yyc/AIReduceNoise3")

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def run_dir(round_index: int) -> str:
    return f"runs/air3_mode2_1609_unet3d_h2f_128b_r{round_index}"


print("== 1. round configs are control variables of r0 ==")
r0 = yaml.safe_load((CONFIGS / "air3_mode2_1609_unet3d_h2f_128_r0.yaml").read_text(encoding="utf-8"))
for rnd in (1, 2, 3, 4):
    path = CONFIGS / f"air3_mode2_1609_unet3d_h2f_128b_r{rnd}.yaml"
    check(path.exists(), f"config exists: {path.name}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    suffix = f"full.dn_3d_128b_r{rnd}"
    check(cfg["data"]["mappings"] == [f"even->{suffix}", f"odd->{suffix}"],
          f"r{rnd}b mappings -> {suffix}")
    check(cfg["training"]["run_dir"] == run_dir(rnd), f"r{rnd}b run_dir")
    check(cfg["data"]["manifest"] == r0["data"]["manifest"], f"r{rnd}b manifest == r0")
    diffs_data = {k for k, v in r0["data"].items() if cfg["data"].get(k) != v}
    diffs_train = {k for k, v in r0["training"].items() if cfg["training"].get(k) != v}
    check(diffs_data <= {"mappings"}, f"r{rnd}b data diffs only mappings (got {sorted(diffs_data)})")
    check(diffs_train <= {"run_dir"}, f"r{rnd}b training diffs only run_dir (got {sorted(diffs_train)})")
    check(cfg["model"] == r0["model"] and cfg["reconstruction"] == r0["reconstruction"],
          f"r{rnd}b model+reconstruction identical to r0")

print("== 2. teacher resolution per round ==")
for rnd in (2, 3, 4):
    teacher = ROOT / run_dir(rnd - 1) / "best.pt"
    state = "present" if teacher.exists() else ("expected later" if rnd > 2 else "PRESENT?")
    print(f"  r{rnd}b teacher -> {run_dir(rnd - 1)}/best.pt [{state}]")
check((ROOT / run_dir(1) / "best.pt").exists(), "r1b best.pt exists (r2b teacher)")
check(not (CONFIGS / "air3_mode2_1609_unet3d_h2f_128b_r5.yaml").exists(), "no r5 config")

print("== 3. manifest already carries the r1 target key ==")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
keys = manifest["splits"]["train"] + manifest["splits"]["val"]
records = {str(rec["key"]): rec for rec in manifest["records"]}
with_r1 = sum(1 for k in keys if "full.dn_3d_128b_r1.raw" in (records[k].get("outputs") or {}))
with_r0 = sum(1 for k in keys if "full.dn_3d_128b_r0.raw" in (records[k].get("outputs") or {}))
check(len(keys) == 200, f"train+val = 200 (got {len(keys)})")
check(with_r1 == 200, f"r1 target registered on 200/200 records (got {with_r1})")
check(with_r0 == 200, f"r0 target registered on 200/200 records (got {with_r0})")
check(len(manifest["splits"].get("test", [])) == 0, "internal test split is empty")

print("== 4. gate + prepare modules import and prep-only mode works on the real r0 data ==")
import prepare_full_dn_targets as prep_mod  # noqa: E402
from check_r1b_target_gates import main as gate_main  # noqa: E402

check(hasattr(prep_mod, "merge_overview_rows"), "prepare exposes merge_overview_rows")
r0_overview = AIR3 / "prepare_full_dn_3d_128b_r0_overview.json"
check(r0_overview.exists(), f"real r0 overview present: {r0_overview.name}")
argv = sys.argv
sys.argv = ["check_r1b_target_gates.py", "--prep-overview", str(r0_overview),
            "--expect-studies", "200", "--label", "r2b"]
try:
    code = gate_main()
finally:
    sys.argv = argv
check(code == 0, "prep-only gate PASSES on the real r0 overview (legacy, no 'rows' key)")

print()
if failures:
    print(f"VALIDATION FAILED ({len(failures)}):")
    for line in failures:
        print("  - " + line)
    raise SystemExit(1)
print("VALIDATION PASSED -- safe to launch scripts/air3_r234b_chain.sh")
