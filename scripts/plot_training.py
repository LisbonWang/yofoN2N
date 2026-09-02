from __future__ import annotations

"""Manual CLI to plot training progress from ``history.jsonl``.

See ``cbct_n2n.training.plot`` for the shared implementation; the same figure
is also refreshed automatically during training when ``training.plot_interval``
is set in the config.

Usage::

    python scripts\\plot_training.py --run-dir runs\\default
    python scripts\\plot_training.py --run-dir runs\\default --smooth 0.6 --logy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.training.plot import plot_history  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training progress from history.jsonl")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"), help="Directory containing history.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <run-dir>/training_curves.png)")
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing factor in [0, 1); 0 disables")
    parser.add_argument("--logy", action="store_true", help="Log-scale loss axes")
    args = parser.parse_args()
    try:
        destination = plot_history(args.run_dir, args.out, args.smooth, args.logy)
        rows = 0
        with (Path(args.run_dir) / "history.jsonl").open(encoding="utf-8") as stream:
            rows = sum(1 for line in stream if line.strip())
        print(f"Plotted {rows} row(s) to {destination}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
