# Full-Target CBCT Supervision

## Goal

Replace independent even/odd Noise2Noise pairing with two subset-to-full training pairs for each Mode2 case:

- even.npy -> full.npy
- odd.npy -> full.npy

Here, full.npy is reconstructed from all 640 projections of the same scan.

## Data Preparation

- The source Z-drive remains read-only.
- For every selected Mode2 case, prepare one untouched full reconstruction using the original raw files and configuration, plus even and odd reconstructions using 320-view subset workspaces.
- All three volumes must have identical 3D shape and z/y/x spacing before artifacts are written.
- Each pair metadata file contains absolute paths for even, odd, and full volumes, projection provenance, dtype, shape, and spacing.

## Training And Validation

- Case-level splits remain unchanged; no slices from a test case may enter train or validation.
- The Dataset emits two samples per axial slice: even -> full and odd -> full.
- Synchronized geometric augmentation applies to source and target. There is no direction swapping.
- Full-slice padding to multiples of 32 remains the only spatial preprocessing.

## Testing And Reporting

- Inference supports even and odd inputs separately and compares each output with full.npy.
- Test reports label full reconstruction as a correlated all-view reference, not a clean ground truth or an independent Noise2Noise reference.
- Per-case output names include the source kind to prevent even and odd results from overwriting each other.

## Limitation

The full target includes the input subset projections, so this is correlated supervision rather than strict independent Noise2Noise. It is expected to reduce noise toward the all-view reconstruction but must not be presented as an unbiased clean-target validation.
