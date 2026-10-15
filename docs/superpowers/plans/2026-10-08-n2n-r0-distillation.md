# N2N and R0 Distillation Implementation Plan

> **For agentic workers:** Implement this plan inline in the current workspace. Keep the existing uncommitted JIROX2 reconstruction changes intact.

**Goal:** Make the three-stage N2N, R0, and fused distillation training scheme executable with the current 3D training pipeline.

**Architecture:** Reuse `N2N3DDataset`, `train_voxel.py`, and `prepare_full_dn_targets.py`. Extend mapping sources to include `full`, allow target generation from a registered `full.*` volume, and add three matching AIR3 Mode2_1609 configs with sequential target-generation instructions.

**Tech Stack:** Python, PyTorch, NumPy memmaps, YAML manifests/configs.

## Global Constraints

- Keep all three models on the same UNet3D architecture and parameter count: base_channels=16, depth=4.
- Keep the AIReduceNoise3 Mode2_1609 study split and generate teacher targets only for train and validation.
- Do not run or add tests in this task.
- Do not alter unrelated or existing user changes.

---

### Task 1: Support full-volume mapping sources

**Files:**
- Modify: `src/cbct_n2n/data/n2n_dataset.py`

**Interfaces:**
- `N2NDataset` continues to accept mappings as `<source>-><target>`.
- Sources include `even`, `odd`, and `full`; targets keep current standard and `full.*` support.

- [x] Expand and document allowed mapping sources, preserving current mapping validation and residual augmentation behavior.

### Task 2: Chain teacher target generation

**Files:**
- Modify: `scripts/prepare_full_dn_targets.py`

**Interfaces:**
- Add `--input-suffix` defaulting to empty, where empty resolves `full.raw` and a value such as `by_r0` resolves `full.by_r0.raw`.
- Keep `--target-suffix` output as `full.<target-suffix>.raw`.
- Record the actual source key and teacher checkpoint in the overview.

- [x] Resolve each input volume from the manifest and use it for inference and source-versus-target diagnostics.
- [x] Preserve test-split exclusion, memmap shape handling, and manifest registration.

### Task 3: Add stage configurations

**Files:**
- Create: `configs/air3_mode2_1609_unet3d_n2n.yaml`
- Create: `configs/air3_mode2_1609_unet3d_r0.yaml`
- Create: `configs/air3_mode2_1609_unet3d_fused.yaml`

**Interfaces:**
- N2N uses `even->odd` and `odd->even`.
- R0 uses `even->full` and `odd->full`.
- Fused uses `full->full.by_r0_by_n2n`.
- All three use the same manifest, residual domains, model dimensions, and training hyperparameters.

- [x] Add reproducible comments describing target generation order and training commands.

### Task 4: Update the final scheme documentation

**Files:**
- Modify: `docs/plans/recursive-blend-at-r1.md`

- [x] Document exact config names, command sequence, target suffixes, expected checkpoint paths, and test isolation.
- [x] Keep the old r0-r4b material clearly marked as historical evidence.

### Task 5: Static verification

**Files:**
- No additional files.

- [x] Inspect the final diff and config/mapping consistency.
- [x] Do not execute the test suite, per the task constraint.
