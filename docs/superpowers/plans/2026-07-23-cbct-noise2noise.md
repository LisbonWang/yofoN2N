# CBCT Noise2Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested project that reconstructs odd/even CBCT projection pairs with xday-kits and trains/tests a full-slice 2D UNet Noise2Noise model.

**Architecture:** A `src`-layout Python package separates XDay data preparation, manifest-backed full-slice loading, the UNet, and training/inference services. Thin scripts load YAML configuration and invoke these services, while all large data and generated artifacts remain outside Git.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, PyYAML, scikit-image, tqdm, pytest, xday-kits.

## Global Constraints

- Treat `Z:\raid5\共享数据\典型图像\长安管口-PDL` as read-only.
- Train only on complete 2D axial slices padded to a multiple of 32; do not implement 2D random patch sampling.
- Split train, validation, and test data by case.
- Use xday-kits public Python APIs for reconstruction.
- Never silently reconstruct with inconsistent per-view geometry/configuration.

---

### Task 1: Package Skeleton And Configuration

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md`, `configs/default.yaml`, `src/cbct_n2n/__init__.py`, `src/cbct_n2n/config.py`, and `tests/test_config.py`.

**Interfaces:** Produce `load_config(path: Path) -> dict` and a console-ready package with development dependencies.

- [ ] Write tests that load the default YAML, reject a missing file, and verify required reconstruction/data/model/training keys.
- [ ] Run `pytest tests/test_config.py -v` and confirm failure because the package is absent.
- [ ] Implement the package metadata, ignored artifact patterns, default values, and strict configuration loader.
- [ ] Run the focused tests and `python -m compileall src`.
- [ ] Commit with `chore: initialize cbct noise2noise package`.

### Task 2: Projection And Configuration Subsetting

**Files:** Create `src/cbct_n2n/reconstruction/subset.py` and `tests/reconstruction/test_subset.py`.

**Interfaces:** Produce `numbered_raw_files(raw_dir) -> list[Path]`, `split_projection_indices(count) -> tuple[list[int], list[int]]`, and `build_subset_workspace(raw_dir, cfg_dir, indices, destination, link_mode) -> SubsetWorkspace`.

- [ ] Write tests with unsorted numbered files, an even and odd projection count, `fdkPara.xml`, a 21-float-per-view `projectionMatrix.bin`, and per-view scalar data.
- [ ] Run the tests and verify imports/functions fail.
- [ ] Implement numeric sorting, positional odd/even selection, safe hard-link/copy materialization with consecutive output names, XML parsing through `ElementTree`, and validated binary row slicing.
- [ ] Reject empty subsets, mismatched binary lengths, existing destinations without overwrite, and missing `noView`.
- [ ] Run `pytest tests/reconstruction/test_subset.py -v` and commit `feat: prepare geometry-safe projection subsets`.

### Task 3: Case Discovery And XDay Pair Reconstruction

**Files:** Create `src/cbct_n2n/reconstruction/discovery.py`, `src/cbct_n2n/reconstruction/prepare.py`, `scripts/prepare_pairs.py`, and tests `tests/reconstruction/test_discovery.py`, `tests/reconstruction/test_prepare.py`.

**Interfaces:** Produce `discover_cases(root) -> list[ScanCase]`, `prepare_case(case, output_dir, options, reconstruct_fn=invoke_xday) -> PairRecord`, and CLI flags for source/output/SDK/filter/overwrite/error continuation.

- [ ] Write discovery fixtures matching the observed UID/raw/config layout and a mock reconstruction returning same-shaped arrays and spacing.
- [ ] Verify tests fail before implementation.
- [ ] Implement unambiguous raw/config pairing, kV/mA parsing via xday-kits, temporary subset orchestration, matching-shape validation, atomic `.npy`/JSON output, and cleanup.
- [ ] Implement a dry-run mode that prints discovered cases and intended outputs without loading the SDK.
- [ ] Run both focused test modules and `python scripts/prepare_pairs.py --help`; commit `feat: reconstruct odd even cbct pairs with xday`.

### Task 4: Case Manifests And Full-Slice Dataset

**Files:** Create `src/cbct_n2n/data/manifest.py`, `src/cbct_n2n/data/dataset.py`, `scripts/create_splits.py`, `tests/data/test_manifest.py`, and `tests/data/test_dataset.py`.

**Interfaces:** Produce `create_case_splits(records, ratios, seed) -> SplitManifest`, `PairedSliceDataset(manifest, split, window, augment=False)`, `pad_to_multiple`, and `unpad`.

- [ ] Write tests proving case-level deterministic separation, ratio validation, shared normalization, full-slice padding, synchronized augmentation, and unpadding.
- [ ] Verify focused tests fail.
- [ ] Implement JSON manifest IO, memory-mapped `.npy` volume loading, axial slice indexing, fixed intensity-window normalization, paired transforms, and tensor conversion.
- [ ] Ensure there is no random crop path or crop-size configuration.
- [ ] Run focused tests and CLI help; commit `feat: add full-slice paired cbct dataset`.

### Task 5: 2D UNet

**Files:** Create `src/cbct_n2n/models/unet.py`, `src/cbct_n2n/models/__init__.py`, and `tests/models/test_unet.py`.

**Interfaces:** Produce `UNet2D(in_channels=1, out_channels=1, base_channels=32, depth=4, residual=True)` with shape-preserving forward behavior.

- [ ] Write tests for 320x320 and padded non-square inputs, gradient propagation, and invalid depth/channel values.
- [ ] Verify tests fail.
- [ ] Implement double-convolution blocks, down/up blocks with skip alignment, initialization, and residual output.
- [ ] Run model tests and commit `feat: add residual 2d unet`.

### Task 6: Metrics, Checkpoints, And Training

**Files:** Create `src/cbct_n2n/training/metrics.py`, `src/cbct_n2n/training/checkpoint.py`, `src/cbct_n2n/training/trainer.py`, `scripts/train.py`, and tests `tests/training/test_metrics.py`, `tests/training/test_trainer.py`.

**Interfaces:** Produce `compute_metrics(prediction, target, data_range)`, `save_checkpoint`, `load_checkpoint`, `train_one_epoch`, `validate`, and `fit`.

- [ ] Write numeric metric tests, checkpoint round-trip tests, and a tiny CPU dataset/model test proving one optimizer step and resume epoch behavior.
- [ ] Verify tests fail.
- [ ] Implement L1 training, AdamW, scheduler state, optional CUDA AMP, clipping, deterministic seeds, JSONL history, best/last checkpoints, and early stopping.
- [ ] Wire configuration and operational path overrides into `scripts/train.py`.
- [ ] Run focused tests and training CLI help; commit `feat: add noise2noise training pipeline`.

### Task 7: Volume Inference And Test CLI

**Files:** Create `src/cbct_n2n/training/inference.py`, `scripts/test.py`, and `tests/training/test_inference.py`.

**Interfaces:** Produce `infer_volume(model, volume, window, device, batch_size) -> np.ndarray` and per-case proxy metric reports.

- [ ] Write a test with an identity model and non-multiple-of-32 dimensions proving exact output shape and intensity round-trip.
- [ ] Verify the test fails.
- [ ] Implement batched axial inference, padding removal, inverse normalization, atomic `.npy` outputs, and PSNR/SSIM/MAE/RMSE reports against the paired target.
- [ ] Wire split/checkpoint/output arguments into `scripts/test.py` and label metrics as independent-reconstruction proxies.
- [ ] Run focused tests and CLI help; commit `feat: add cbct volume testing pipeline`.

### Task 8: End-To-End Documentation And Verification

**Files:** Modify `README.md`; create `tests/test_cli.py`.

**Interfaces:** Document installation, XDAY_DIR setup, dry-run, pair preparation, split creation, training, resume, testing, artifact layout, and limitations.

- [ ] Add subprocess smoke tests for all four script `--help` commands.
- [ ] Document exact PowerShell commands using the provided Z-drive path while keeping outputs local.
- [ ] Run `pytest -q`, `python -m compileall src scripts`, and every CLI help command.
- [ ] When XDay DLL is available, run SDK version/GPU checks and one `prepare_pairs --dry-run`; record actual results without starting a full batch reconstruction.
- [ ] Commit `docs: document cbct noise2noise workflow`.
