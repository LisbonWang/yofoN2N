import json

import numpy as np
import pytest

from cbct_n2n.training.plot import load_history, plot_history, smooth


def _write_history(run_dir, rows: list[dict]):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return run_dir


def test_load_history_parses_rows_and_skips_malformed(tmp_path, capsys):
    run_dir = _write_history(tmp_path, [
        {"epoch": 0, "train_loss": 0.5, "val_loss": 0.4},
        {"epoch": 1, "train_loss": 0.3, "val_loss": 0.2},
    ])
    with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{not json}\n")
    rows = load_history(run_dir)
    assert len(rows) == 2
    assert rows[1]["val_loss"] == 0.2
    assert "malformed" in capsys.readouterr().err


def test_load_history_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path)


def test_smooth_identity_and_ema():
    values = np.asarray([1.0, 2.0, 3.0], dtype=float)
    assert np.array_equal(smooth(values, 0.0), values)
    out = smooth(values, 0.5)
    assert out[0] == 1.0
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.25)


def test_plot_history_writes_png_with_all_columns(tmp_path):
    rows = [
        {"epoch": i, "train_loss": 0.2 - i * 0.001, "val_loss": 0.18 - i * 0.001,
         "learning_rate": 1e-4, "psnr": 30 + i * 0.1, "ssim": 0.9 + i * 0.001}
        for i in range(5)
    ]
    run_dir = _write_history(tmp_path, rows)
    out = plot_history(run_dir)
    assert out == run_dir / "training_curves.png"
    assert out.is_file()
    assert out.stat().st_size > 1000  # non-trivial PNG payload


def test_plot_history_custom_out_path(tmp_path):
    run_dir = _write_history(tmp_path, [{"epoch": 0, "train_loss": 0.5, "val_loss": 0.4}])
    destination = plot_history(run_dir, out=tmp_path / "custom" / "curve.png")
    assert destination.is_file()
