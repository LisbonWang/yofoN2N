# External CBCT Pair Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configuration-driven Mode2 reconstruction command that writes pairs only to the dedicated Z-drive pair directory.

**Architecture:** Keep `scripts/prepare_pairs.py` as its explicit-path generic CLI. Add a focused wrapper that loads reconstruction settings from YAML, always passes `mode="Mode2"`, and uses existing discovery/preparation functions. Training remains unchanged except for manifest examples that reference the external pair root.

**Tech Stack:** Python 3.11, pytest, PyYAML, xday-kits, XDay 617.002.

## Global Constraints

- Read source data only from `Z:\raid5\共享数据\典型图像\长安管口-PDL`.
- Write only pair artifacts under `Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs`.
- Force Mode2 discovery.
- Do not write manifests, checkpoints, metrics, or denoised volumes under the pair root.
- Preserve the generic `prepare_pairs.py` interface.

---

### Task 1: Configuration-Driven Mode2 Reconstruction CLI

**Files:**
- Create: `scripts/reconstruct_mode2_pairs.py`
- Modify: `configs/default.yaml`
- Modify: `configs/smoke.yaml`
- Test: `tests/test_cli.py`

**Interfaces:**
- `build_parser() -> argparse.ArgumentParser` exposes `--config`, `--dry-run`, `--overwrite`, `--keep-workspace`, `--continue-on-error`, and `--per-view-file`.
- `main() -> int` calls `discover_cases(source_root, mode="Mode2")` and `prepare_case(case, output_root, options)`.

- [ ] **Step 1: Write failing CLI help test**

```python
def test_mode2_reconstruction_cli_help_exposes_config():
    result = subprocess.run([sys.executable, str(root / "scripts" / "reconstruct_mode2_pairs.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout
```

- [ ] **Step 2: Run the focused test**

Run: `& $py -m pytest -q tests/test_cli.py`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create the CLI**

```python
config = load_config(args.config)
recon = config["reconstruction"]
cases = discover_cases(recon["source_root"], continue_on_error=args.continue_on_error, mode="Mode2")
options = PrepareOptions(Path(recon["xday_dir"]), recon.get("post_procs", ""), recon.get("dtype", "float32"), recon.get("link_mode", "auto"), args.overwrite, args.keep_workspace, per_view_files)
for case in cases:
    prepare_case(case, Path(recon["output_root"]), options)
```

`--dry-run` prints the selected Mode2 cases and exits without calling `prepare_case`.

- [ ] **Step 4: Set external pair roots**

Set both configuration files' `reconstruction.output_root` to `Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs`. Keep `training.run_dir` local (`runs/default` or `runs/smoke-raw`) and keep manifest paths local.

- [ ] **Step 5: Run focused test and help check**

Run:

```powershell
& $py -m pytest -q tests/test_cli.py
& $py scripts/reconstruct_mode2_pairs.py --help
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add scripts/reconstruct_mode2_pairs.py configs/default.yaml configs/smoke.yaml tests/test_cli.py
git commit -m "feat: separate cbct pair reconstruction storage"
```

### Task 2: Documentation And External Root Smoke Check

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the reconstruction command**

```powershell
& $py scripts/reconstruct_mode2_pairs.py --config configs/default.yaml --dry-run
& $py scripts/reconstruct_mode2_pairs.py --config configs/default.yaml
```

State that only pairs are written to the external root and training runs remain local.

- [ ] **Step 2: Verify external-root dry run**

Run the first command and confirm it reports Mode2 cases, exits 0, and leaves `Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs` absent or unchanged.

- [ ] **Step 3: Run final verification**

```powershell
& $py -m pytest -q
& $py -m compileall -q src scripts
& $py scripts/reconstruct_mode2_pairs.py --help
git diff --check
```

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: document external cbct pair reconstruction"
```
