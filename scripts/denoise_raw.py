from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import infer_volume, write_raw_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Denoise a headerless float32 RAW CBCT volume.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Task YAML config matching the selected checkpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    expected_bytes = int(np.prod(args.shape)) * np.dtype("<f4").itemsize
    if args.input.stat().st_size != expected_bytes:
        raise ValueError(f"Input size must be {expected_bytes} bytes for shape {tuple(args.shape)} float32 RAW.")
    config = load_config(args.config)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = UNet2D(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    source = np.memmap(args.input, dtype="<f4", mode="r", shape=tuple(args.shape))
    prediction = infer_volume(model, source, device, args.batch_size)
    write_raw_volume(args.output, prediction)
    print(f"device={device} output={args.output} shape={prediction.shape} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
