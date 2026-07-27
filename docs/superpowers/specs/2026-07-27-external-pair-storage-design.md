# External CBCT Pair Storage Design

## Objective

Separate CBCT reconstruction artifacts from training artifacts. Reconstruct
Mode2 scans into the dedicated Z-drive directory:

```text
Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs
```

Training reads manifests referencing that directory while checkpoints, history,
and inference outputs remain local to the repository's ignored `runs/` and
`outputs/` directories.

## Boundaries

The source root remains read-only:

```text
Z:\raid5\共享数据\典型图像\长安管口-PDL
```

The dedicated pair directory may contain only reconstructed data products:

```text
<case_id>/even.npy
<case_id>/odd.npy
<case_id>/full.npy
<case_id>/pair.json
```

No checkpoints, training history, split manifests, metrics, or denoised volumes
are written to the pair directory.

## Reconstruction Entry Point

Add `scripts/reconstruct_mode2_pairs.py`. It accepts a configuration path and
optional operational flags supported by pair preparation (`--dry-run`,
`--overwrite`, `--keep-workspace`, `--continue-on-error`, and explicit
per-view binary widths). It loads `reconstruction.source_root`,
`reconstruction.output_root`, and XDay settings from configuration, forces
`Mode2`, and delegates to the existing `prepare_case` workflow.

The script creates the exact configured pair root and case subdirectories when
needed. It must never create or alter files beneath the source root.

## Configuration

Set `configs/default.yaml` and `configs/smoke.yaml` reconstruction output roots
to the dedicated pair directory. Training configuration keeps its local run
directory. `data.manifest` remains an explicit input and is not stored beneath
the pair root.

## Verification

Tests must prove that the new CLI loads source/output paths from configuration,
discovers only Mode2 cases, writes outputs under the configured pair root, and
exposes its command-line help. Existing pair preparation behavior remains
covered by its current tests.
