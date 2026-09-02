from __future__ import annotations

"""Live monitor for a running training run.

While ``scripts/train.py`` is running in the background, this script polls
``<run-dir>/history.jsonl`` every few seconds and shows:

* the latest epoch metrics (train/val loss, learning rate, psnr/ssim if enabled)
* elapsed time and estimated time per epoch
* best-val-loss epoch seen so far

It also re-plots ``training_curves.png`` on every refresh by default, so the
figure stays up to date while training runs.

Usage::

    python scripts\\monitor_training.py --run-dir runs\\default
    python scripts\\monitor_training.py --run-dir runs\\default --interval 10 --no-curve

Press Ctrl+C to stop monitoring (training is unaffected).
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.training.plot import load_history  # noqa: E402


def _fmt(value: float, width: int = 12) -> str:
    return f"{value:>{width}.6g}"


def _latest_status(rows: list[dict]) -> str:
    if not rows:
        return "  (no data yet - waiting for the first epoch...)"
    last = rows[-1]
    epoch = int(last.get("epoch", 0)) + 1
    pieces = [
        f"epoch {epoch:<4}",
        f"train_l1={_fmt(last.get('train_loss', float('nan')))}",
        f"val_l1={_fmt(last.get('val_loss', float('nan')))}",
        f"lr={last.get('learning_rate', 0):.3e}",
    ]
    for key in ("psnr", "ssim", "mae", "rmse"):
        if key in last:
            pieces.append(f"{key}={_fmt(last[key], 8)}")
    return "  ".join(pieces)


def _best_status(rows: list[dict]) -> str:
    best_epoch, best_val = None, float("inf")
    for row in rows:
        value = float(row.get("val_loss", float("inf")))
        if value < best_val:
            best_val, best_epoch = value, int(row.get("epoch", 0))
    if best_epoch is None:
        return "best: (none)"
    return f"best val_l1={best_val:.6g} @ epoch {best_epoch + 1}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live monitor for a running training run")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"), help="Directory containing history.jsonl")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--no-curve", action="store_true", help="Do not re-plot training_curves.png on refresh")
    args = parser.parse_args()

    history_path = args.run_dir / "history.jsonl"
    started = time.time()
    last_count = -1
    print(f"Monitoring {args.run_dir.resolve()} (Ctrl+C to stop)\n", flush=True)
    try:
        while True:
            rows = []
            if history_path.is_file():
                try:
                    rows = load_history(args.run_dir)
                except ValueError:
                    rows = []
            elapsed = time.time() - started
            if rows:
                rate = elapsed / len(rows)
                eta = "  ".join([
                    f"elapsed={int(elapsed // 60)}m{int(elapsed % 60):02d}s",
                    f"{rate:.1f}s/epoch",
                ])
            else:
                eta = f"elapsed={int(elapsed // 60)}m{int(elapsed % 60):02d}s (waiting)"
            print(f"[{datetime.now():%H:%M:%S}] {eta}", flush=True)
            print("  " + _latest_status(rows), flush=True)
            print("  " + _best_status(rows), flush=True)
            if rows and len(rows) != last_count and not args.no_curve:
                try:
                    from cbct_n2n.training.plot import plot_history

                    plot_history(args.run_dir)
                    print(f"  curve refreshed -> {args.run_dir / 'training_curves.png'}", flush=True)
                except Exception as exc:  # curve refresh must never kill the monitor
                    print(f"  [WARN] curve refresh failed: {exc}", flush=True)
                last_count = len(rows)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped (training continues)", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
