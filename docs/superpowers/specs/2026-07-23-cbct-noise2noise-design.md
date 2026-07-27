# CBCT Image-Domain Noise2Noise Design

## Goal

Build a reproducible Python project for preparing paired CBCT reconstructions from odd/even projection subsets, training a 2D UNet with Noise2Noise supervision, and testing trained checkpoints on complete axial slices.

## Data Source And Safety

- Source root: `Z:\raid5\共享数据\典型图像\长安管口-PDL`.
- The source tree is read-only from this project's perspective.
- Generated projection subsets, copied configuration files, reconstructed volumes, manifests, checkpoints, and predictions are written below a user-selected output directory.
- Dataset splits are made by case, never by slice, to prevent leakage.

## Reconstruction Pair Preparation

The preparation command discovers each case's numbered projection directory and matching reconstruction configuration directory. Projection files are sorted by their numeric stem and divided by sequence position into even and odd subsets.

Each subset receives a private workspace. Projection files are materialized with hard links when possible and copies as a fallback. Per-view configuration data is subset with the identical index vector. At minimum, the preparation code updates `fdkPara.xml/noView` and subsets `projectionMatrix.bin`; supported per-view binary layouts are validated against the original view count before slicing. Unknown or inconsistent layouts fail with a case-specific error rather than producing a questionable reconstruction.

The two subsets are reconstructed independently through the public `xday-kits` Python API. The output for each case contains `even.npy`, `odd.npy`, and `pair.json`. Metadata records source paths, selected projection names, array dtype and shape, spacing, reconstruction settings, and preparation status. Temporary subset workspaces may be retained for diagnosis through an explicit CLI option.

## Dataset

Training uses complete 2D axial slices from paired reconstructed volumes. No random 2D patch mode is provided. Each slice is padded on the bottom and right to a multiple of 32 and the padding is removed when writing test predictions.

Input and target are normalized using the same configurable intensity window. Training may apply synchronized flips and rotations to both members of a pair. Validation and testing are deterministic. Odd-to-even and even-to-odd orientation is selected deterministically per sample during training to avoid a fixed subset bias.

A manifest command creates deterministic train, validation, and test partitions by case using a fixed seed and configurable ratios.

## Model And Training

The model is a single-channel 2D UNet with four resolution levels, skip connections, normalization, and a residual output. Default training uses L1 loss, AdamW, automatic mixed precision on CUDA, gradient clipping, checkpoint resume, best-validation checkpoint selection, early stopping, and deterministic seeding.

Configuration is stored in YAML. CLI arguments select the configuration file and may override operational paths such as data root, run directory, and checkpoint. Training history is written as JSON Lines and checkpoints contain model, optimizer, scheduler, epoch, metrics, and configuration.

## Testing And Outputs

Testing runs slice-wise inference over an entire paired volume, removes model padding, restores the configured intensity scale, and writes a denoised `.npy` volume plus a JSON metrics report per case. PSNR, SSIM, MAE, and RMSE are computed against the independent odd/even target. These are Noise2Noise proxy metrics, not clean-reference metrics, and documentation labels them accordingly.

## Error Handling

- Missing source, configuration, projection, SDK, or checkpoint paths fail before expensive work begins.
- Ambiguous case layouts are reported and skipped only when `--continue-on-error` is explicitly enabled.
- Odd/even counts must differ by at most one and both must be non-empty.
- Reconstructed pair shapes and spacing must match exactly.
- Existing outputs are not overwritten unless `--overwrite` is supplied.
- Interrupted preparation writes no completed manifest entry until both volumes and metadata are atomically finalized.

## Project Structure

- `src/cbct_n2n/reconstruction/`: discovery, projection/config subsetting, and XDay orchestration.
- `src/cbct_n2n/data/`: manifests, normalization, full-slice datasets, and synchronized transforms.
- `src/cbct_n2n/models/`: 2D UNet.
- `src/cbct_n2n/training/`: training loop, checkpoints, metrics, and inference.
- `scripts/`: data preparation, split creation, training, and testing entry points.
- `configs/`: default YAML configuration.
- `tests/`: unit and smoke tests; XDay is mocked unless the actual SDK is explicitly available.

## Verification

Unit tests cover projection sorting and splitting, XML and binary configuration subsetting, case discovery, volume manifest validation, synchronized full-slice transforms, padding, normalization, UNet shape preservation, one CPU training step, checkpoint round-trip, metrics, and volume inference. CLI help and package compilation are smoke-tested. A real XDay reconstruction is run only when the SDK path and a deliberately selected case are available; otherwise the limitation is reported explicitly.
