# Raw-Value CBCT Training Design

## Objective

Remove window normalization, clipping, and inverse normalization from the
full-target CBCT training and inference path. The model will consume and
produce the reconstructed float32 values directly.

## Rationale

The current `[-1, 3]` window clips values outside that interval. The user
reports that approximately 90% of voxels lie in the range, so clipping the
remaining values risks losing clinically and physically relevant extremes.
The reconstruction values are already on a compact, stable scale, making
additional training normalization unnecessary.

## Data Flow

For every axial slice, the Dataset emits one of these pairs without numerical
value transformation:

```text
even[z] -> full[z]
odd[z]  -> full[z]
```

Volumes remain memory-mapped until individual slices are converted to float32.
Padding to a multiple of 32 remains zero padding. No random patch mode is
introduced. Geometric augmentation remains synchronized between source and
target.

Inference accepts a raw float32 volume, pads each slice, runs the model, and
removes padding. It no longer scales input values or rescales output values.

## Model And Loss

The existing residual 2D U-Net remains unchanged. Its residual output and L1
loss operate in the original reconstruction units:

```text
loss = mean(abs(model(subset) - full))
```

This remains correlated full-target supervision, not independent Noise2Noise
supervision.

## Evaluation

MAE and RMSE use raw reconstruction units. PSNR and SSIM still require a
positive `data_range`; this is retained as a dedicated evaluation setting in
configuration and must not be used to transform training or inference values.

The default evaluation range is 4.0, corresponding to the established
`[-1, 3]` display/assessment span. It is an explicit metric convention, not a
clipping boundary.

## Configuration And Compatibility

Replace `data.window` with `data.metric_data_range: 4.0`. Update Dataset,
inference, training and test scripts to consume the new setting. Existing
manifests and `even.npy`/`odd.npy`/`full.npy` artifacts remain compatible.

## Verification

Focused tests must show that Dataset samples preserve values outside `[-1, 3]`,
identity inference preserves raw values without clipping or rescaling, and the
test CLI passes `metric_data_range` solely to evaluation. Full tests, compile,
and CLI help checks must pass.
