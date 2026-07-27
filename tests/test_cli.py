import subprocess
import sys
from pathlib import Path


def test_all_script_help_commands():
    root = Path(__file__).parents[1]
    for name in ("prepare_pairs.py", "create_splits.py", "train.py", "test.py"):
        result = subprocess.run([sys.executable, str(root / "scripts" / name), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        if name == "prepare_pairs.py":
            assert "--mode" in result.stdout
