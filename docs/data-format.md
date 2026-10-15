# Dataset and Manifest Format

The training code does not require a particular dataset name or storage server. A user supplies reconstructed volumes and a JSON manifest through the command line. Keep private raw volumes and manifests outside the source repository.

## RAW volumes

The 3D dataset reads headerless little-endian float32 volumes by default. Store the voxel shape explicitly as `[Z, Y, X]`; the expected file size is `Z * Y * X * 4` bytes. If a dataset uses another dtype, record its NumPy dtype string in each record and ensure the reader supports it. Values are used directly; normalization or unit conversion must be consistent across all source and target volumes and documented in the task config or dataset notes.

## Manifest

The manifest is JSON with a `records` array and a `splits` object. `train`, `val`, and optionally `test` contain record keys. A record has a stable `key` (or legacy `case_id`), `shape`, optional `dtype`, and an `outputs` map of logical volume names to paths. Mapping names such as `even`, `odd`, `full`, and `full.teacher_v1` are selected by `data.mappings` in the training config.

```json
{
  "seed": 42,
  "records": [
    {
      "key": "case_0001",
      "case_id": "case_0001",
      "shape": [256, 512, 512],
      "dtype": "<f4",
      "outputs": {
        "even.raw": "volumes/case_0001/even.raw",
        "odd.raw": "volumes/case_0001/odd.raw",
        "full.raw": "volumes/case_0001/full.raw"
      }
    }
  ],
  "splits": {
    "train": ["case_0001"],
    "val": ["case_0002"],
    "test": ["case_0003"]
  }
}
```

Relative `outputs` paths are resolved against the manifest file's parent directory. Absolute paths are also accepted. Split entries must refer to record keys; each case/patient must appear in exactly one split. Exclude test records from teacher-target generation and all model-selection decisions.

## Preflight checklist

- Confirm every split key exists exactly once in `records` and that train/val/test do not overlap.
- Confirm each record has every logical volume required by the selected mapping and residual augmentation.
- Confirm all referenced files exist and each file size matches `shape` and `dtype`.
- Confirm source and target volumes share the same shape, voxel geometry, units, and preprocessing convention.
- Record dataset provenance, preprocessing, split seed, and any generated teacher checkpoint beside the manifest.
- Keep identifiers and paths in manifests non-identifying when sharing them; manifests can expose patient/study IDs and storage topology.

The AIReduceNoise configs under `configs/air3_*` are experiment snapshots, not examples of a portable manifest. See [configs/README.md](../configs/README.md) for the config inventory.

## Experiment-specific observation (2026-10-14)

The AIR3 training manifest inspected for the 400-record stride-64 experiment contained 400 records and split counts of 360 train, 40 validation, and 0 test, while its `target_total` metadata still said 200. Consumers should derive counts from `records` and `splits`, validate their consistency, and not treat `target_total` as authoritative. This observation applies to that experiment snapshot only.
