# Linux CBCT Training Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an executable Linux shell entry point that creates a local manifest from mounted pair data and runs training locally.

**Architecture:** The script resolves the checkout root, reads defaults from shell variables, validates the pair mount without writing to it, invokes existing Python CLIs, and supports checkpoint resume.

**Tech Stack:** Bash, Python 3, PyTorch training CLI.

## Global Constraints

- Pair input root is `/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs` by default.
- The script does not write under `PAIR_ROOT`.
- Local defaults are `outputs/splits-linux.json` and `runs/linux-full`.

---

### Task 1: Linux Training Entry Point

**Files:**
- Create: `scripts/train_linux.sh`
- Modify: `README.md`
- Test: `tests/test_linux_script.py`

- [ ] **Step 1: Write the failing static test**

```python
def test_linux_training_script_has_strict_mode_and_external_pair_default():
    script = (root / "scripts" / "train_linux.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs" in script
    assert "scripts/create_splits.py" in script
    assert "scripts/train.py" in script
```

- [ ] **Step 2: Run focused test**

Run: `& $py -m pytest -q tests/test_linux_script.py`

Expected: FAIL because the script is absent.

- [ ] **Step 3: Create the Bash script**

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR_ROOT="${PAIR_ROOT:-/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/splits-linux.json}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.yaml}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/linux-full}"
```

Validate `PAIR_ROOT` and one `pair.json`, create local parent directories,
generate the manifest, build optional `--resume "$RESUME"`, and invoke train.

- [ ] **Step 4: Run static test and Bash syntax validation**

Run:

```powershell
& $py -m pytest -q tests/test_linux_script.py
bash -n scripts/train_linux.sh
```

Expected: PASS and exit code 0.

- [ ] **Step 5: Document invocation and commit**

```powershell
git add scripts/train_linux.sh tests/test_linux_script.py README.md
git commit -m "feat: add linux cbct training script"
```
