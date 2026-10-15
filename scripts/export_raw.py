from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.data.manifest import load_manifest
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import infer_volume, write_raw_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CBCT source and denoised volumes as RAW.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Task YAML config matching the selected checkpoint.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(args.manifest or config["data"]["manifest"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = UNet2D(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    records = {str(item["case_id"]): item for item in manifest["records"]}

    for case_id in manifest["splits"][args.split]:
        record = records[case_id]
        case_dir = args.output_dir / case_id
        for source_kind in ("odd", "even", "full"):
            source = np.load(record[f"{source_kind}_path"], mmap_mode="r")
            write_raw_volume(case_dir / f"{source_kind}.raw", source)
            prediction = infer_volume(model, source, device, args.batch_size)
            write_raw_volume(case_dir / f"{source_kind}_denoised.raw", prediction)
            print(f"{case_id} {source_kind}: exported", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
