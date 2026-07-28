from pathlib import Path


def test_linux_training_script_has_strict_mode_and_external_pair_default():
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "train_linux.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs" in script
    assert "scripts/create_splits.py" in script
    assert "scripts/train.py" in script


def test_linux_training_script_reexecs_bash_when_invoked_by_sh():
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "train_linux.sh").read_text(encoding="utf-8")
    assert 'if [ -z "${BASH_VERSION:-}" ]; then' in script
    assert 'exec bash "$0" "$@"' in script
