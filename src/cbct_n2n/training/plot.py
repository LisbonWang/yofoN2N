from __future__ import annotations

"""Plot training progress from ``history.jsonl``.

Shared by ``scripts.plot_training`` (manual CLI) and
``cbct_n2n.training.trainer.fit`` (automatic refresh during training).
"""

import json
import os
import sys
from pathlib import Path

# Multiple OpenMP runtimes coexist in torch+conda-MKL environments
# (libiomp5md.dll from torch AND from numpy's MKL build). matplotlib's
# frequent small-matrix inversions can then abort the process with
# "Fatal Python error: Aborted". Intel's documented workaround:
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

#: keys always plotted as the "loss" family; everything else with a numeric
#: column in history.jsonl becomes its own panel (up to MAX_PANELS).
LOSS_KEYS = ("train_loss", "val_loss")
MAX_PANELS = 6

_PLOTTER: object | None = None  # lazily imported matplotlib.pyplot


def _pyplot():
    """Import matplotlib lazily so this module works without it installed."""
    global _PLOTTER
    if _PLOTTER is None:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe; no display needed
        import matplotlib.pyplot as plt

        _PLOTTER = plt
    return _PLOTTER


def load_history(run_dir: str | Path) -> list[dict]:
    """Parse ``<run_dir>/history.jsonl`` into a list of metric dicts."""
    path = Path(run_dir) / "history.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No history.jsonl in {run_dir}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping malformed line {line_no}: {exc}", file=sys.stderr)
    if not rows:
        raise ValueError(f"{path} contains no valid rows")
    return rows


def _column(rows: list[dict], key: str) -> np.ndarray:
    values = [float(row[key]) for row in rows if key in row]
    return np.asarray(values, dtype=float)


def _epochs(rows: list[dict]) -> np.ndarray:
    epochs = _column(rows, "epoch")
    if epochs.size == 0:  # tolerate missing epoch field
        return np.arange(len(rows), dtype=float)
    return epochs


def smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (alpha in [0, 1), 0 = no smoothing)."""
    if alpha <= 0 or values.size < 2:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def _final_value(rows: list[dict], key: str) -> float | None:
    for row in reversed(rows):
        if key in row:
            return float(row[key])
    return None


def plot_history(
    run_dir: str | Path,
    out: str | Path | None = None,
    smooth_alpha: float = 0.0,
    logy: bool = False,
) -> Path:
    """Plot all numeric columns in ``history.jsonl`` and save a PNG.

    Returns the path of the written figure.
    """
    plt = _pyplot()
    rows = load_history(run_dir)
    epochs = _epochs(rows)

    # collect all numeric columns, loss first
    keys: list[str] = []
    for key in LOSS_KEYS:
        if any(key in row for row in rows) and key not in keys:
            keys.append(key)
    for row in rows:
        for key, value in row.items():
            if key in ("epoch",) or key in keys:
                continue
            if isinstance(value, (int, float)):
                keys.append(key)
    keys = keys[:MAX_PANELS]

    run_name = Path(run_dir).resolve().name
    figure, axes = plt.subplots(len(keys), 1, figsize=(9, 3.2 * len(keys)), squeeze=False)
    colors = {"train_loss": "#d62728", "val_loss": "#1f77b4", "learning_rate": "#2ca02c"}
    for ax, key in zip((a[0] for a in axes), keys):
        values = _column(rows, key)
        epochs_for = epochs[: values.size]
        ax.plot(epochs_for + 1, values, color=colors.get(key, "#7f7f7f"), alpha=0.35, linewidth=1.2, label=key)
        if smooth_alpha > 0:
            ax.plot(epochs_for + 1, smooth(values, smooth_alpha), color=colors.get(key, "#7f7f7f"),
                    linewidth=1.8, label=f"{key} (smooth)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        if logy and _column(rows, key).min() > 0:
            ax.set_yscale("log")
        final_value = _final_value(rows, key)
        ax.legend(loc="upper right", fontsize=8)
        if final_value is not None:
            ax.set_title(f"{key}  (final {final_value:.4g})", fontsize=10)
    figure.suptitle(f"Training progress - {run_name}", fontsize=12)
    # NOTE: avoid figure.tight_layout() here -- on conda numpy+MKL builds it
    # triggers numpy.linalg.inv inside matplotlib transforms and can abort the
    # whole process (Fatal Python error: Aborted). Fixed margins are safe.
    figure.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.06, hspace=0.45)

    destination = Path(out) if out else Path(run_dir) / "training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
