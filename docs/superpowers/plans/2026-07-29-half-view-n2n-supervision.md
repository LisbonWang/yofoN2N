# Half-View Noise2Noise Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable 50%-view to 50%-view Noise2Noise target mode while preserving full-view supervision by default.

**Architecture:** `PairedSliceDataset` owns source-target selection. Its new `supervision_mode` parameter selects either full-view targets or cross-subset targets. `scripts/train.py` reads the setting from `data.supervision_mode` and passes it to both train and validation datasets.

**Tech Stack:** Python 3.11, NumPy memory maps, PyTorch Dataset, PyYAML, pytest.

## Global Constraints

- `full_view` remains the default and maps `even -> full`, `odd -> full`.
- `half_view_n2n` maps `even -> odd`, `odd -> even`.
- No random 2D patch path is introduced.
- Reconstructed data and manifests are unchanged.

---

### Task 1: Add Dataset Supervision Selection

**Files:**
- Modify: `src/cbct_n2n/data/dataset.py`
- Modify: `tests/data/test_dataset.py`

**Interfaces:**
- Produces: `PairedSliceDataset(..., supervision_mode: str = "full_view")`.
- Consumes: `"full_view"` or `"half_view_n2n"`.

- [ ] **Step 1: Write failing tests**

```python
half_view = PairedSliceDataset(manifest, "train", supervision_mode="half_view_n2n")
np.testing.assert_allclose(half_view[0]["target"][..., :3, :5], odd[0][None])
np.testing.assert_allclose(half_view[1]["target"][..., :3, :5], even[0][None])
```

- [ ] **Step 2: Run the dataset test and verify it fails**

Run: `conda run --no-capture-output -n torch python -m pytest tests/data/test_dataset.py -q`

Expected: `TypeError` because `supervision_mode` is not accepted.

- [ ] **Step 3: Implement mode validation and target selection**

```python
if supervision_mode not in {"full_view", "half_view_n2n"}:
    raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
target_volume = full_volume if self.supervision_mode == "full_view" else (odd_volume if source_kind == "even" else even_volume)
```

- [ ] **Step 4: Run the dataset test and verify it passes**

Run: `conda run --no-capture-output -n torch python -m pytest tests/data/test_dataset.py -q`

Expected: all dataset tests pass.

### Task 2: Wire Configuration Into Training

**Files:**
- Modify: `configs/default.yaml`
- Modify: `scripts/train.py`

**Interfaces:**
- Consumes: optional `data.supervision_mode` from YAML.
- Produces: train and validation datasets constructed with the same mode.

- [ ] **Step 1: Add `data.supervision_mode: full_view` to the default configuration**

- [ ] **Step 2: Read once in `scripts/train.py` and pass it to both datasets**

```python
supervision_mode = str(data_cfg.get("supervision_mode", "full_view"))
train_data = PairedSliceDataset(..., supervision_mode=supervision_mode)
val_data = PairedSliceDataset(..., supervision_mode=supervision_mode)
```

- [ ] **Step 3: Smoke-check argument parsing**

Run: `conda run --no-capture-output -n torch python scripts/train.py --help`

Expected: exit code 0.

### Task 3: Verify the Feature

**Files:**
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Run focused test suites**

Run: `conda run --no-capture-output -n torch python -m pytest tests/data/test_dataset.py tests/training/test_trainer.py -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect the working tree**

Run: `git diff --check`

Expected: no whitespace errors.
