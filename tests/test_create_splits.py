import json
import subprocess
import sys
from pathlib import Path


def test_create_splits_uses_current_pair_root_instead_of_stored_absolute_paths(tmp_path):
    root = Path(__file__).parents[1]
    case_dir = tmp_path / "pairs" / "case-1"
    case_dir.mkdir(parents=True)
    for name in ("even.npy", "odd.npy", "full.npy"):
        (case_dir / name).write_bytes(b"fixture")
    record = {
        "case_id": "case-1",
        "even_path": r"\\server\share\pairs\case-1\even.npy",
        "odd_path": r"\\server\share\pairs\case-1\odd.npy",
        "full_path": r"\\server\share\pairs\case-1\full.npy",
    }
    (case_dir / "pair.json").write_text(json.dumps(record), encoding="utf-8")
    manifest_path = tmp_path / "splits.json"

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "create_splits.py"), str(case_dir.parent), str(manifest_path), "--ratios", "1", "0", "0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = manifest["records"][0]
    assert generated["even_path"] == str(case_dir / "even.npy")
    assert generated["odd_path"] == str(case_dir / "odd.npy")
    assert generated["full_path"] == str(case_dir / "full.npy")
