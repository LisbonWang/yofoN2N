# Half-View Noise2Noise Supervision

## Goal

Make the training target configurable so the existing 50%-angle reconstructions
can train a Noise2Noise model without using the full-view reconstruction as its
target.

## Configuration

`data.supervision_mode` has two values:

- `full_view` (default): `even -> full` and `odd -> full`.
- `half_view_n2n`: `even -> odd` and `odd -> even`.

The default preserves the current training and validation behavior. The new
mode uses independent odd/even projection subsets on both sides of each
supervised pair.

## Scope

`PairedSliceDataset` accepts the selected supervision mode and chooses the
target slice accordingly. Both training and validation construct the dataset
from the same configuration value. Manifests and reconstructed volumes remain
unchanged.

## Verification

Dataset tests cover the exact source-target values for `full_view` and
`half_view_n2n`, including both directions. Invalid mode values fail early.
