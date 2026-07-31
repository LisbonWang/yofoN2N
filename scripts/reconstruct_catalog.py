from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.reconstruction import build_subset_workspace, split_projection_indices, numbered_raw_files


def _atomic_raw(path: Path, volume: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    np.asarray(volume, dtype="<f4").tofile(temp)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct full/even/odd float32 RAW volumes from a scan catalog.")
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--xday-dir", type=Path, required=True)
    parser.add_argument("--post-procs", default="")
    parser.add_argument("--link-mode", choices=["auto", "copy", "hardlink"], default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from xday import invoke_xday, load_xday

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    source_root = Path(catalog["source_root"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sdk = load_xday(args.xday_dir)
    records = [record for record in catalog["records"] if record.get("status") == "ready"]
    if args.limit is not None:
        records = records[:args.limit]
    output_records, failures = [], []
    for record in records:
        case_id = record["case_id"]
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_dir, cfg_dir = source_root / record["raw_dir"], source_root / record["cfg_dir"]
            files = numbered_raw_files(raw_dir)
            even_indices, odd_indices = split_projection_indices(len(files))
            with tempfile.TemporaryDirectory(prefix=f"xday_{case_id}_") as workspace:
                workspace_root = Path(workspace)
                even = build_subset_workspace(raw_dir, cfg_dir, even_indices, workspace_root / "even", args.link_mode)
                odd = build_subset_workspace(raw_dir, cfg_dir, odd_indices, workspace_root / "odd", args.link_mode)
                common = dict(xday_install_dir=args.xday_dir, post_procs=args.post_procs, dtype="float32", show_progress=False, sdk=sdk)
                full_result = invoke_xday(raw_dir, cfg_dir, record["kv"], record["ma"], **common)
                even_result = invoke_xday(even.raw_dir, even.cfg_dir, record["kv"], record["ma"], **common)
                odd_result = invoke_xday(odd.raw_dir, odd.cfg_dir, record["kv"], record["ma"], **common)
            volumes = {"full": full_result, "even": even_result, "odd": odd_result}
            shapes = {name: list(result.volume.shape) for name, result in volumes.items()}
            if len({tuple(shape) for shape in shapes.values()}) != 1:
                raise ValueError(f"shape mismatch: {shapes}")
            outputs = {}
            for name, result in volumes.items():
                path = case_dir / f"{name}.raw"
                _atomic_raw(path, result.volume)
                outputs[name] = path.relative_to(output_root).as_posix()
            metadata = dict(record)
            metadata.update({"status": "reconstructed", "outputs": outputs, "shape": shapes["full"], "dtype": "float32",
                             "spacing_zyx": [float(full_result.volume_info.z_spacing), float(full_result.volume_info.y_spacing), float(full_result.volume_info.x_spacing)],
                             "full_projections": len(files), "even_projections": len(even_indices), "odd_projections": len(odd_indices)})
            (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            output_records.append(metadata)
            print(f"{case_id}: reconstructed", flush=True)
        except Exception as exc:
            failure = {"case_id": case_id, "status": "reconstruction_failed", "error": str(exc)}
            failures.append(failure)
            print(f"{case_id}: ERROR {exc}", flush=True)
            if not args.continue_on_error:
                raise
    output_index = {"schema_version": 2, "source_root": catalog["source_root"], "selection": catalog["selection"],
                    "ratios": catalog["ratios"], "seed": catalog["seed"], "records": output_records,
                    "splits": {name: [record["case_id"] for record in output_records if record["case_id"] in ids] for name, ids in catalog["splits"].items()},
                    "failures": failures}
    (output_root / "reconstruction-index.json").write_text(json.dumps(output_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_root / 'reconstruction-index.json'}: reconstructed={len(output_records)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
