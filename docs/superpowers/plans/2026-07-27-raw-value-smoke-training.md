# Raw-Value CBCT Smoke Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the full-target 2D U-Net for 10 epochs on a small, case-isolated raw-value Mode2 dataset.

**Architecture:** Remove window normalization from Dataset and inference, while retaining a dedicated metric range for PSNR/SSIM. Add an optional deterministic per-case axial-slice limit, then provide a smoke configuration that uses 16 slices per case, batch size 2, and 10 epochs. Reconstruct two Mode2 cases into a local output directory, assigning one to train and one to validation.

**Tech Stack:** Python 3.11, PyTorch 2.12 CUDA, NumPy, pytest, XDay 617.002.

## Global Constraints

- Source data on `Z:\raid5\共享数据\典型图像\长安管口-PDL` is read-only.
- Continue using only Mode2 scans.
- Training and validation cases must be disjoint.
- Inputs, targets, residual output, and L1 loss use raw float32 reconstruction values.
- `metric_data_range: 4.0` is evaluation-only and must not clip or rescale values.
- No random 2D patch mode is introduced.
- The smoke run uses 10 epochs, batch size 2, and at most 16 axial slices per case.

---

### Task 1: Raw-Value Dataset And Inference

**Files:**
- Modify: `src/cbct_n2n/data/dataset.py`
- Modify: `src/cbct_n2n/training/inference.py`
- Modify: `scripts/train.py`
- Modify: `scripts/test.py`
- Modify: `configs/default.yaml`
- Test: `tests/data/test_dataset.py`
- Test: `tests/training/test_inference.py`

**Interfaces:**
- `PairedSliceDataset(manifest, split, augment=False, max_slices_per_case=None)` returns raw float32 source and target tensors.
- `infer_volume(model, volume, device, batch_size=4)` returns raw float32 predictions.
- `evaluate_volume(prediction, target, metric_data_range)` uses the range only for metrics.

- [ ] **Step 1: Write failing Dataset tests**

```python
def test_dataset_preserves_raw_values_and_limits_slices_per_case(tmp_path):
    # use values below -1 and above 3 in a two-slice fixture
    dataset = PairedSliceDataset(manifest, "train", augment=False, max_slices_per_case=1)
    assert len(dataset) == 2
    np.testing.assert_allclose(dataset[0]["source"][..., :3, :5], even[0][None])
    np.testing.assert_allclose(dataset[0]["target"][..., :3, :5], full[0][None])
```

- [ ] **Step 2: Run the focused Dataset test**

Run: `& $py -m pytest -q tests/data/test_dataset.py`

Expected: FAIL because Dataset requires a window and clips values.

- [ ] **Step 3: Implement raw Dataset values and deterministic limit**

```python
def __init__(self, manifest, split, augment=False, max_slices_per_case=None):
    if max_slices_per_case is not None and max_slices_per_case < 1:
        raise ValueError("max_slices_per_case must be positive when provided")
    ...
    slice_count = min(even.shape[0], max_slices_per_case or even.shape[0])
    self.slices.extend((index, z, kind) for z in range(slice_count) for kind in ("even", "odd"))
```

Use `np.asarray(image, dtype=np.float32)` without clipping for source and target.

- [ ] **Step 4: Write failing raw inference test**

```python
def test_identity_inference_preserves_raw_values_without_window_clipping():
    volume = np.array([[[-2.0, 4.5]]], dtype=np.float32)
    output = infer_volume(torch.nn.Identity(), volume, torch.device("cpu"), batch_size=1)
    np.testing.assert_allclose(output, volume)
```

- [ ] **Step 5: Implement raw inference and metric-only range**

Remove window arguments and scale operations from inference. Change metric evaluation to accept a `metric_data_range: float`; wire `data.metric_data_range` through the train/test scripts.

- [ ] **Step 6: Run focused tests**

Run: `& $py -m pytest -q tests/data/test_dataset.py tests/training/test_inference.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/cbct_n2n/data/dataset.py src/cbct_n2n/training/inference.py scripts/train.py scripts/test.py configs/default.yaml tests/data/test_dataset.py tests/training/test_inference.py
git commit -m "feat: train cbct denoiser on raw values"
```

### Task 2: Ten-Epoch Smoke Configuration And Training

**Files:**
- Create: `configs/smoke.yaml`
- Modify: `README.md`

**Interfaces:**
- `configs/smoke.yaml` sets `training.epochs: 10`, `training.batch_size: 2`, `data.max_slices_per_case: 16`, and a separate run directory.
- The smoke manifest has one training case and one validation case, with no overlapping case IDs.

- [ ] **Step 1: Add smoke configuration and README command**

```yaml
reconstruction:
  source_root: 'Z:\\raid5\\共享数据\\典型图像\\长安管口-PDL'
  output_root: outputs/smoke-raw/pairs
  xday_dir: 'D:\\XDay Release\\XDay617.2'
  post_procs: ''
  dtype: float32
  link_mode: auto
  per_view_files:
    projectionMatrix.bin: 84
data:
  manifest: outputs/smoke-raw/splits.json
  max_slices_per_case: 16
  metric_data_range: 4.0
  split_ratios: [0.8, 0.1, 0.1]
  seed: 42
  augment: true
model:
  in_channels: 1
  out_channels: 1
  base_channels: 32
  depth: 4
  residual: true
training:
  epochs: 10
  batch_size: 2
  run_dir: runs/smoke-raw
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 0
  amp: true
  gradient_clip: 1.0
  patience: 15
  device: cuda
```

The smoke configuration is complete because `load_config` does not implement configuration inheritance.

- [ ] **Step 2: Verify configuration loading**

Run: `& $py -c "from cbct_n2n.config import load_config; c=load_config('configs/smoke.yaml'); assert c['training']['epochs']==10 and c['data']['max_slices_per_case']==16"`

Expected: exits with code 0.

- [ ] **Step 3: Reconstruct two Mode2 cases locally**

Run `prepare_pairs.py` with `--mode Mode2`, XDay 617.002, and a source root narrowed to one selected case at a time. Store results below `outputs/smoke-raw/pairs`; do not use `--overwrite` unless the target case directory has been inspected.

- [ ] **Step 4: Create a case-isolated smoke manifest**

Write `outputs/smoke-raw/splits.json` with the first reconstructed case in `train`, the second in `val`, and an empty `test` list. Confirm case ID sets are disjoint.

- [ ] **Step 5: Run the 10-epoch CUDA smoke training**

Run: `& $py scripts/train.py --config configs/smoke.yaml --manifest outputs/smoke-raw/splits.json --run-dir runs/smoke-raw`

Expected: `best.pt`, `last.pt`, and a ten-line or early-stopped `history.jsonl` are written. Report device, epochs executed, final train/validation L1, and checkpoint path.

- [ ] **Step 6: Run one-case both-input evaluation**

Run test inference for the validation split with `--input-kind both`, then confirm even/odd denoised arrays and correlated-full metric JSON files exist.

- [ ] **Step 7: Commit configuration and documentation**

```powershell
git add configs/smoke.yaml README.md
git commit -m "config: add raw-value cbct smoke training"
```

### Task 3: Final Verification

- [ ] **Step 1: Run full verification**

Run:

```powershell
& $py -m pytest -q
& $py -m compileall -q src scripts
& $py scripts/train.py --help
& $py scripts/test.py --help
git diff --check
```

Expected: all tests pass, compilation succeeds, and CLI help exits 0.
