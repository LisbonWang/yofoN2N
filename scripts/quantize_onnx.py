"""Static INT8 quantization (QDQ) of an exported ONNX denoiser with calibration slices.

Calibration data is drawn from the N2N pairs tree (same distribution as training):
axial slices of even.npy / odd.npy are padded to a multiple of 32 and fed to the
network exactly as the training pipeline does (see cbct_n2n.data.dataset).

NOTE (measured on the 7.76M demo UNet2D): int8 static quantization is NOT a good
fit for this model family. Deep denoisers normalised with GroupNorm have very small
activation ranges (~1e-2..1e-1); the int8 step is large relative to that, so the
QDQ model showed ~29% relative error vs fp32 on real projection slices (MinMax
calibration, 16 real slices), and the QDQ graph ran ~3.5x SLOWER on CPU here.
fp16 export (export_onnx.py --fp16) remains the practical choice for this model;
real size cuts come from architecture changes (base_channels/depth), not int8.
The script is kept for models with wider activation ranges.

Usage:
    python scripts/quantize_onnx.py --onnx outputs/demo/best.onnx
    python scripts/quantize_onnx.py --onnx outputs/demo/best.onnx --num-cases 4 --slices-per-case 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

DEFAULT_PAIRS_ROOT = Path(r"Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs")


def _pad_to_multiple(image: np.ndarray, multiple: int = 32) -> np.ndarray:
    h, w = image.shape
    ph, pw = h + (-h % multiple), w + (-w % multiple)
    padded = np.zeros((ph, pw), dtype=image.dtype)
    padded[:h, :w] = image
    return padded


class PairSliceReader(CalibrationDataReader):
    """Yield axial slices from the N2N pairs tree as calibration samples."""

    def __init__(self, pairs_root: Path, input_name: str, num_cases: int, slices_per_case: int, seed: int = 0):
        case_dirs = sorted(d for d in pairs_root.iterdir() if d.is_dir() and (d / "even.npy").is_file())
        if not case_dirs:
            raise FileNotFoundError(f"no case dirs with even.npy under {pairs_root}")
        rng = np.random.default_rng(seed)
        selected = rng.choice(case_dirs, size=min(num_cases, len(case_dirs)), replace=False)
        self.samples: list[np.ndarray] = []
        for case_dir in selected:
            vol = np.load(case_dir / "even.npy", mmap_mode="r")
            n_slices = vol.shape[0]
            for z in rng.choice(n_slices, size=min(slices_per_case, n_slices), replace=False):
                self.samples.append(_pad_to_multiple(np.asarray(vol[int(z)], dtype=np.float32)))
        if not self.samples:
            raise RuntimeError("no calibration slices produced")
        self.input_name = input_name
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self.samples):
            return None
        x = self.samples[self._index][None, None]  # -> (1, 1, H, W)
        self._index += 1
        return {self.input_name: x}

    def rewind(self) -> None:
        self._index = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=Path("outputs/demo/best.onnx"))
    parser.add_argument("--output", type=Path, default=None, help="default: <onnx dir>/<stem>_int8.onnx")
    parser.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT,
                        help="N2N pairs tree used for calibration slices")
    parser.add_argument("--num-cases", type=int, default=2, help="number of cases to sample (default: 2)")
    parser.add_argument("--slices-per-case", type=int, default=8, help="slices per case (default: 8)")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if not args.onnx.is_file():
        parser.error(f"onnx model not found: {args.onnx}")
    output = args.output or args.onnx.with_name(args.onnx.stem + "_int8.onnx")

    reader = PairSliceReader(args.pairs_root, "input", args.num_cases, args.slices_per_case)
    print(f"[quant] calibration : {len(reader.samples)} slices from {args.pairs_root}", flush=True)
    print(f"[quant] sample shape : {reader.samples[0].shape}", flush=True)

    quantize_static(
        str(args.onnx),
        str(output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        extra_options={"MinimumRealRange": 0.0001},
    )

    onnx.checker.check_model(onnx.load(str(output)))
    print(f"[quant] saved        : {output} ({output.stat().st_size / 1e6:.2f} MB, "
          f"was {args.onnx.stat().st_size / 1e6:.2f} MB)", flush=True)

    if not args.no_validate:
        # validate on a held-out real slice (not random noise): distribution match matters
        import numpy as np
        rng = np.random.default_rng(7)
        case_dirs = sorted(d for d in args.pairs_root.iterdir() if d.is_dir() and (d / "even.npy").is_file())
        vol = np.load(rng.choice(case_dirs) / "even.npy", mmap_mode="r")
        x = _pad_to_multiple(np.asarray(vol[int(rng.integers(vol.shape[0]))], dtype=np.float32))[None, None]
        fp32_sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        q_sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ref = fp32_sess.run(None, {"input": x})[0]
        t0 = time.perf_counter()
        got = q_sess.run(None, {"input": x})[0]
        ms = (time.perf_counter() - t0) * 1000
        diff = float(np.abs(got - ref).max())
        rel = diff / max(float(np.abs(ref).max()), 1e-12)
        print(f"[quant] fp32 vs int8  : max|diff|={diff:.3e}  rel={rel:.3e}  cpu={ms:.1f} ms "
              f"(input {x.shape[2]}x{x.shape[3]})", flush=True)
        if rel < 1e-2:
            print(f"[quant] PASS (rel < 1%)", flush=True)
        else:
            print(f"[quant] WARNING: rel error {rel*100:.1f}% — int8 is a poor fit for this model "
                  f"(narrow GroupNorm activations). Prefer export_onnx.py --fp16.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
