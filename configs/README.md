# Configuration Inventory

`load_config` currently requires `reconstruction`, `data`, `model`, and `training` sections for every task config. `train_voxel.py` requires an explicit manifest and supports a configured model architecture, mappings, residual tags, 3D window settings, optimizer settings, and run directory.

## Portable starting point

- [`examples/unet3d_n2n.yaml`](examples/unet3d_n2n.yaml) is a sanitized template. Replace the manifest and run directory when launching a task. Its relative reconstruction paths are illustrative and do not imply data is bundled with this repository.

## Experiment snapshots

- `air3_mode2_1609_*.yaml` belongs to specific AIReduceNoise3 Mode2_1609 experiments. These configs use private manifests, a specific reconstruction layout, and experiment-specific stride/model choices. They are preserved to reproduce those runs and are not portable defaults. The 200-record stride-112 baseline and 400-record stride-64 fine-tuning variants are separate snapshots; do not assume their manifests or checkpoints are interchangeable.
- `1609v2_*.yaml` belongs to AIReduceNoise2 experiments and uses a separate dataset layout. Do not mix its manifest or checkpoints with AIReduceNoise3.
- Other dated or round-numbered configs document earlier experiments; inspect their headers and paths before running.

`recon_filters/` contains site- and scanner-specific XDay filter templates. Their embedded geometry and output paths are reconstruction inputs, not general defaults; copy and adapt them for the local SDK and acquisition protocol before use.

For new data, copy the portable template, choose source/target mappings supported by the Dataset, set a new run directory, and pass the manifest explicitly. See [the data format guide](../docs/data-format.md).
