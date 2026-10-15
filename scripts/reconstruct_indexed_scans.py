from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cbct_n2n.reconstruction import build_subset_workspace, split_projection_indices
from cbct_n2n.reconstruction.prepare import _parse_kv_ma


def _write_raw(path: Path, volume: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    np.asarray(volume, dtype="<f4").tofile(temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct full/even/odd RAW volumes from an indexed scan set.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--xday-dir", type=Path, required=True)
    parser.add_argument("--post-procs", default="")
    parser.add_argument("--link-mode", choices=["auto", "copy", "hardlink"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = Path(dataset["source_root"])
    output_root = manifest_path.parent
    try:
        from xday import invoke_xday
    except ImportError as exc:
        raise SystemExit("XDay is required for reconstruction. Install the XDay SDK and xday-kits package using your site's supported installation procedure.") from exc
    if args.dry_run:
        for item in dataset["records"]:
            print(f"{item['case_id']}\t{item['raw_dir']}\t{item['cfg_dir']}")
        return 0

    failures = 0
    for item in dataset["records"]:
        case_dir = output_root / item["case_id"]
        metadata_path = case_dir / "metadata.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not args.overwrite and all((case_dir / f"{name}.raw").is_file() for name in ("full", "even", "odd")):
                item["reconstruction_status"] = "success"
                continue
            raw_dir, cfg_dir = source_root / item["raw_dir"], source_root / item["cfg_dir"]
            kv, ma = _parse_kv_ma(raw_dir.name)
            files_count = len(list(raw_dir.glob("*.raw")))
            even_indices, odd_indices = split_projection_indices(files_count)
            with tempfile.TemporaryDirectory(prefix=f"xday_{item['case_id']}_") as workspace:
                even_ws = build_subset_workspace(raw_dir, cfg_dir, even_indices, Path(workspace) / "even", args.link_mode)
                odd_ws = build_subset_workspace(raw_dir, cfg_dir, odd_indices, Path(workspace) / "odd", args.link_mode)
                common = dict(xday_install_dir=args.xday_dir, post_procs=args.post_procs, dtype="float32")
                full_result = invoke_xday(raw_dir, cfg_dir, kv, ma, **common)
                even_result = invoke_xday(even_ws.raw_dir, even_ws.cfg_dir, kv, ma, **common)
                odd_result = invoke_xday(odd_ws.raw_dir, odd_ws.cfg_dir, kv, ma, **common)
            results = {"full": full_result, "even": even_result, "odd": odd_result}
            shape = tuple(int(value) for value in full_result.volume.shape)
            if any(tuple(result.volume.shape) != shape for result in results.values()):
                raise ValueError("full/even/odd reconstructed shapes differ")
            for name, result in results.items():
                _write_raw(case_dir / f"{name}.raw", result.volume)
            info = full_result.volume_info
            item.update({
                "reconstruction_status": "success",
                "shape": list(shape),
                "dtype": "<f4",
                "spacing_zyx": [float(info.z_spacing), float(info.y_spacing), float(info.x_spacing)],
                "outputs": {name: f"{item['case_id']}/{name}.raw" for name in results},
                "metadata": f"{item['case_id']}/metadata.json",
                "kv": kv,
                "ma": ma,
            })
            _write_json(metadata_path, {"source_root": "../", "record": item})
            print(f"{item['case_id']}: success", flush=True)
        except Exception as exc:
            failures += 1
            item["reconstruction_status"] = "error"
            item["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(metadata_path, {"record": item})
            print(f"{item['case_id']}: ERROR {item['error']}", flush=True)
            if not args.continue_on_error:
                raise
    _write_json(output_root / "reconstruction-index.json", dataset)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
