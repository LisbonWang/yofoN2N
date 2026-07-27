from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.data.manifest import load_manifest
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import evaluate_volume, infer_volume, resolve_input_kinds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full-volume CBCT Noise2Noise inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-kind", choices=["even", "odd", "both"], default="both")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(args.manifest or config["data"]["manifest"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = UNet2D(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    records = {str(item["case_id"]): item for item in manifest["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in manifest["splits"][args.split]:
        record = records[case_id]
        full = np.load(record["full_path"], mmap_mode="r")
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for input_kind in resolve_input_kinds(args.input_kind):
            source = np.load(record[f"{input_kind}_path"], mmap_mode="r")
            prediction = infer_volume(model, source, device, args.batch_size)
            np.save(case_dir / f"{input_kind}_denoised.npy", prediction)
            metrics = evaluate_volume(prediction, full, float(config["data"]["metric_data_range"]))
            (case_dir / f"{input_kind}_metrics.json").write_text(
                json.dumps(
                    {"case_id": case_id, "source_kind": input_kind, "reference": "correlated full-view reconstruction", "metrics": metrics},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{case_id} {input_kind}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
