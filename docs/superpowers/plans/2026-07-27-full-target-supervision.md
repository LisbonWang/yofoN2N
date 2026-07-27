# Full-Target CBCT Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 2D UNet with even-to-full and odd-to-full CBCT pairs reconstructed from the same Mode2 scan.

**Architecture:** Extend the pair-preparation record with a full-view reconstruction while retaining odd/even subset workspaces. The Dataset expands every axial slice into two source-to-full samples; inference emits separate even and odd predictions and compares both to the full reconstruction.

**Tech Stack:** Python 3.11, PyTorch 2.12 CUDA, NumPy, pytest, xday-kits, XDay 617.002.

## Global Constraints

- Treat `Z:\raid5\共享数据\典型图像\长安管口-PDL` as read-only.
- Process Mode2 cases only when `--mode Mode2` is specified.
- Reconstruct full targets using all original projections and original configuration.
- Preserve case-level train/val/test separation.
- Optimize normalized L1 only; do not add L2 or SSIM loss in this migration.
- Label full targets as correlated all-view references, not clean ground truth or independent Noise2Noise targets.

---

### Task 1: Full Reconstruction Artifacts

**Files:**
- Modify: `src/cbct_n2n/reconstruction/prepare.py`
- Modify: `tests/reconstruction/test_prepare.py`

**Interfaces:**
- `PairRecord` exposes `full_path: Path`.
- `prepare_case(...)` writes `even.npy`, `odd.npy`, `full.npy`, and `pair.json`.

- [ ] **Step 1: Write failing reconstruction test**

Extend the mock reconstructor fixture so it receives original raw/config paths for the full call and subset paths for even/odd calls. Assert that metadata has `full_path`, all three arrays have matching shape, and mismatched full spacing raises `ValueError`.

- [ ] **Step 2: Run focused test**

Run: `python -m pytest -q tests/reconstruction/test_prepare.py`
Expected: FAIL because PairRecord and metadata have no full path.

- [ ] **Step 3: Implement full reconstruction**

Call the existing reconstructor once with `case.raw_dir` and `case.cfg_dir`; validate full/even/odd shapes and spacings; atomically save full.npy; include `full_path` and full projection provenance in pair.json; include full.npy in overwrite checks.

- [ ] **Step 4: Verify focused tests**

Run: `python -m pytest -q tests/reconstruction/test_prepare.py`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/cbct_n2n/reconstruction/prepare.py tests/reconstruction/test_prepare.py && git commit -m "feat: reconstruct full cbct targets"`

### Task 2: Two Source-To-Full Dataset Samples

**Files:**
- Modify: `src/cbct_n2n/data/dataset.py`
- Modify: `tests/data/test_dataset.py`

**Interfaces:**
- `PairedSliceDataset` reads `even_path`, `odd_path`, and `full_path`.
- One case slice expands into two samples with `source_kind` equal to `even` or `odd`; both use `full` as target.

- [ ] **Step 1: Write failing Dataset test**

Create one 2-slice fixture with distinct even, odd, and full values. Assert Dataset length is four, samples 0/1 are even/full and odd/full for z=0, samples 2/3 are even/full and odd/full for z=1, and a DataLoader batches `source`, `target`, and `source_kind`.

- [ ] **Step 2: Run focused test**

Run: `python -m pytest -q tests/data/test_dataset.py`
Expected: FAIL because the Dataset indexes one sample per slice and targets odd.

- [ ] **Step 3: Implement source expansion**

Load full.npy with memory mapping; index each z slice twice; normalize and pad source/full together; apply synchronized flips and square-only rotations; return source_kind; keep validation deterministic.

- [ ] **Step 4: Verify focused tests**

Run: `python -m pytest -q tests/data/test_dataset.py tests/training/test_trainer.py`
Expected: PASS; trainer consumes source and full target with its existing L1 loss.

- [ ] **Step 5: Commit**

Run: `git add src/cbct_n2n/data/dataset.py tests/data/test_dataset.py && git commit -m "feat: train cbct subsets against full targets"`

### Task 3: Full-Reference Inference And Documentation

**Files:**
- Modify: `scripts/test.py`
- Modify: `src/cbct_n2n/training/inference.py`
- Modify: `README.md`
- Modify: `tests/training/test_inference.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Test CLI accepts `--input-kind even|odd|both`, default `both`.
- Each output path is `<case>/<input-kind>_denoised.npy` with a matching metrics JSON whose reference is `correlated full-view reconstruction`.

- [ ] **Step 1: Write failing inference and CLI tests**

Assert two inference outputs for `--input-kind both`, distinct source-kind filenames, and metric metadata naming the correlated full-view reference. Extend CLI help smoke test to require `--input-kind`.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest -q tests/training/test_inference.py tests/test_cli.py`
Expected: FAIL because test.py only uses even input and odd reference.

- [ ] **Step 3: Implement separate evaluation paths**

Choose `[even, odd]` for `both`; load the corresponding input volume and full_path target; call infer_volume; atomically save `<kind>_denoised.npy`; evaluate against full; write source kind and correlated reference label into metrics JSON.

- [ ] **Step 4: Document loss and limitation**

Update README commands and descriptions to say that each Mode2 case generates even/odd/full; training uses normalized L1; testing compares each subset prediction to the correlated full-view reconstruction.

- [ ] **Step 5: Run final verification**

Run: `python -m pytest -q`; `python -m compileall -q src scripts`; `python scripts/prepare_pairs.py --help`; `python scripts/test.py --help`.
Expected: all tests pass, compilation succeeds, and CLI help exposes mode and input-kind controls.

- [ ] **Step 6: Commit**

Run: `git add scripts/test.py src/cbct_n2n/training/inference.py README.md tests/training/test_inference.py tests/test_cli.py && git commit -m "feat: evaluate cbct denoising against full targets"`
