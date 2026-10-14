# Portable Project Design

## Goal

Make the repository usable by researchers on their own Windows or Linux systems without requiring the original author's account, workstation, GPU server, network shares, or private datasets. Keep existing experiment results reproducible by preserving their dataset-specific configuration as clearly labeled snapshots.

## Scope

- Replace personal paths and machine assumptions in the main README and current training entry points with user-provided roots, environment variables, or command-line options.
- Provide generic example configuration and manifest documentation that can be copied for a new dataset.
- Make reusable training chain scripts accept repository, Python, manifest, data-root, output-root, and device settings instead of embedding the original host setup.
- Separate generic installation requirements from optional XDay SDK and private-data requirements.
- Mark existing AIReduceNoise/AIReduceNoise2/AIReduceNoise3 configurations, plans, and reports as experiment-specific snapshots where their paths and dataset assumptions are intentional.
- Keep the existing model, dataset, and training semantics unchanged unless a hard-coded path currently prevents reuse.

## Repository Architecture

The documentation will explain the current boundaries: reconstruction and indexing under `src/cbct_n2n/reconstruction`; manifest and RAW window datasets under `src/cbct_n2n/data`; 2D/3D architectures under `src/cbct_n2n/models`; shared optimization, checkpoints, metrics, and inference under `src/cbct_n2n/training`; executable workflows under `scripts`; task-specific parameters under `configs`.

New tasks should normally add a manifest and a YAML configuration, not fork the trainer. The trainer or Dataset should change only when the new data format, mapping semantics, or model interface cannot be represented by current configuration fields.

## Portability Contract

- Repository code contains no default requirement for a particular username, host, drive letter, mount point, or private dataset.
- Data and output locations are supplied by config or CLI. Environment-variable expansion, if added, is explicit and documented.
- Manifests describe stable record IDs, split membership, dtype, shape, and volume paths. Relative paths resolve against a declared manifest/data root; no patient or study is shared between train, validation, and test splits.
- Generic setup documents supported Python/package installation and optional GPU/XDay setup. Hardware-specific commands are examples, not requirements.
- Existing experiment snapshots retain exact paths and hyperparameters, with a visible note naming the experiment context and warning that the files are not portable defaults.

## User Workflows

1. Install the package and optional dependencies on a local workstation or remote compute host.
2. Prepare or obtain RAW volumes and a manifest, then run a path/shape/split preflight.
3. Copy a generic config, set mappings, model/window/optimizer parameters and a fresh run directory.
4. Launch the same training entry point locally or over SSH, optionally inside tmux or a scheduler allocation.
5. Inspect history/checkpoints and evaluate on a held-out test split.

## Compatibility and Risks

The old AIR3 manifests and configs encode real experiment data layout and must not be silently rewritten. Shell chain scripts must preserve stage ordering, fail-fast behavior, and checkpoint isolation while allowing caller-supplied paths. The current 400-record manifest metadata has a `target_total` value that disagrees with its record and split counts; documentation should explain validation, and code should not trust this field without checking the actual records.

## Verification

Verify that generic examples contain no personal host or home paths; scripts accept user-supplied paths and fail clearly when unset; historical artifacts are explicitly labeled; documented config keys match the loaders; and `git diff --check` is clean. Run focused CLI/config tests for changed behavior, plus the repository test suite if the shared loader or manifest handling changes.
