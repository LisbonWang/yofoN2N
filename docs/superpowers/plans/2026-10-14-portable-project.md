# Portable Project Implementation Plan

> **For agentic workers:** Use this plan task by task in the current session. Preserve all pre-existing user changes. Do not rewrite private-data experiment snapshots as generic configs.

**Goal:** Make the repository's supported setup, data, training, and inference workflows reusable without requiring the original author's host, account, storage layout, or private dataset.

**Architecture:** Keep paths supplied by config/CLI/environment at workflow boundaries. Add a sanitized generic 3D config and manifest/data-format guide, make reusable launcher defaults portable, and identify fixed AIR3 artifacts as experiment snapshots. Preserve model and supervision behavior.

**Tech Stack:** Python 3.10+, PyTorch, NumPy memmap RAW volumes, YAML configs, JSON manifests, Bash, pytest already present in the repository.

## Global Constraints

- Do not require `yyc`, a fixed IP address, a particular mount, GPU model, Conda location, or private AIR3 data to run a generic workflow.
- Do not commit private RAW volumes, identifying manifests, credentials, or machine-local environment files.
- Preserve exact historical configs and reports; label them as experiment-specific rather than changing their paths silently.
- Keep training mappings, model behavior, split semantics, and checkpoint behavior unchanged.
- Do not add or run tests in this execution unless the user separately asks for verification; use static inspection and non-training CLI checks only.
- Do not commit or push implementation changes unless asked.

## File Map

- `README.md`: generic setup, architecture, local/remote workflows, and links to data/config docs.
- `docs/data-format.md`: portable manifest schema, relative path rules, split isolation, RAW sizing, and private-data handling.
- `configs/examples/unet3d_n2n.yaml`: sanitized example task config with no personal paths.
- `src/cbct_n2n/config.py`: documented environment expansion for config values if needed by the selected CLI path.
- `scripts/train_voxel.py`, inference/reconstruction CLI scripts: accept input/output locations explicitly and avoid unusable private defaults.
- `scripts/air3_n2n_r0_fused_chain.sh`: caller-supplied Python, manifest, run root, and GPU visibility; repair its stale target-generation entry point only if it remains a supported runnable script.
- `configs/README.md`: classify portable examples versus fixed-data experiment snapshots.
- `docs/training_plan_1609.md`, `docs/plans/*.md`, `docs/reports/*.md`, `configs/air3_*.yaml`: retain historical facts and add concise snapshot notices where appropriate.
- `pyproject.toml`, `environment.yml`: make optional reconstruction SDK requirements explicit without making ordinary training installation depend on a workstation-local SDK.

## Tasks

### Task 1: Document the portable data contract

**Files:** Create `docs/data-format.md`; modify `README.md`.

- [x] Describe headerless little-endian float32 RAW and the role of `shape`, `dtype`, `outputs`, stable record IDs, and train/val/test splits.
- [x] Document that relative output paths resolve from the manifest directory, matching `N2N3DDataset` behavior; include a synthetic, non-identifying JSON example.
- [x] State that each independent case/patient belongs to only one split and that private source volumes/manifests are not shipped by the repository.
- [x] Replace README operational sections that assume `yyc@192.168.36.102`, `/home/yyc`, `/data/raid0`, `Z:`, or a local admin profile with variables such as `PROJECT_ROOT`, `DATA_ROOT`, `MANIFEST`, `RUN_ROOT`, and `XDAY_DIR`.
- [x] Explain local, SSH, tmux, and scheduler use generically; present the original server only in a clearly dated AIR3 experiment note if needed.

### Task 2: Provide a generic task configuration and install profile

**Files:** Create `configs/examples/unet3d_n2n.yaml`; modify `configs/README.md`, `pyproject.toml`, and `environment.yml` only as required.

- [x] Copy the smallest compatible 3D N2N config shape and remove dataset, person, GPU-server, and absolute-home paths. Use documented user-supplied manifest/run paths.
- [x] Keep the required `reconstruction`, `data`, `model`, and `training` sections valid under `load_config`.
- [x] Split ordinary training dependencies from optional XDay/reconstruction dependencies so installing the Python package does not require an unavailable SDK.
- [x] Document supported Python/PyTorch installation and GPU runtime selection without pinning a specific GPU or CUDA build as universal.
- [x] Add an inventory explaining which existing configs are portable examples and which are AIReduceNoise experiment snapshots.

### Task 3: Remove unusable private defaults from supported CLIs

**Files:** Audit and modify user-facing scripts including `train.py`, `train_voxel.py`, `test.py`, `denoise_raw.py`, `export_raw.py`, reconstruction entry points, and RAW inference/evaluation scripts that currently default to private drive or mount paths.

- [x] For each CLI, require a path that cannot have a meaningful generic default, or accept it from a documented environment variable; provide a clear parser error when missing.
- [x] Keep repository-relative defaults only when the referenced file actually exists and is a generic example.
- [x] Ensure manifest-relative RAW lookup stays the default for manifests and does not require a second private data-root constant.
- [x] Make output paths explicit and create output directories only as the command's documented behavior.
- [x] Check `--help` output and static references to ensure examples match actual argument names.

### Task 4: Make training launch configuration caller-controlled

**Files:** Modify `scripts/air3_n2n_r0_fused_chain.sh`; modify chain/config docs.

- [x] Resolve repository root relative to the script, not `/home/yyc/workspace/N2N`.
- [x] Read Python executable, manifest, run root, and visible GPU list from arguments or environment; validate all required paths before starting.
- [x] Use configurable stage config/checkpoint paths and unique output run directories; refuse accidental reuse of an existing run directory unless explicit resume was requested.
- [x] Preserve stage ordering, `set -euo pipefail`, and test-split exclusion. Confirm the target-generation script named by the chain exists and its CLI matches before advertising the chain as supported.
- [x] Avoid adding a user account, host address, Conda path, or shared-drive path to the launcher.

### Task 5: Preserve historical reproducibility while labeling snapshots

**Files:** Modify applicable AIR3 config comments, `docs/training_plan_1609.md`, historical plans/reports, and add snapshot guidance to `configs/README.md`.

- [x] Add a short notice to each currently discoverable fixed-data plan/report/config family that identifies it as tied to the named experiment and not a portable default.
- [x] Keep actual historical paths, metrics, dates, and hyperparameters intact.
- [x] Identify the 200-record/stride-112 and 400-record/stride-64 variants separately; document any known manifest metadata disagreement as a dated observation, not a permanent schema rule.
- [x] Do not normalize old AIReduceNoise2 and AIReduceNoise3 paths into one dataset root.

### Task 6: Final static verification

**Files:** All changed files.

- [x] Run `git diff --check` on the changed files.
- [x] Search generic documentation/config examples for usernames, private host addresses, personal home directories, private mounts, and workstation drive letters; remaining hits must be confined to labeled experiment snapshots.
- [x] Confirm every config named as an example exists and passes `load_config` through a non-training CLI/config inspection command.
- [x] Inspect `git status` and the final diff, confirming unrelated user changes and private data remain untouched.

## Self-Review

- Data portability is covered by Task 1; generic config/install is covered by Task 2; code entry points and launchers are covered by Tasks 3 and 4; provenance is covered by Task 5; verification is covered by Task 6.
- Generic examples use only user-selected paths; fixed paths remain in clearly identified archival snapshots.
- The 200-record and 400-record experiments remain distinct and their manifests are never interchangeable by default.
