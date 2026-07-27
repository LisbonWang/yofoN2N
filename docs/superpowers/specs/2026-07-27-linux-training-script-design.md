# Linux CBCT Training Script Design

## Objective

Provide a shell entry point for training on Linux with reconstructed pair data
mounted at:

```text
/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs
```

## Responsibilities

`scripts/train_linux.sh` uses the external pair directory as read-only input.
It creates a local case-level manifest, then runs the existing Python training
CLI. Manifests, checkpoints, and history remain local to the Linux checkout.

## Defaults

```text
PYTHON_BIN=python3
PAIR_ROOT=/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs
MANIFEST=outputs/splits-linux.json
CONFIG=configs/default.yaml
RUN_DIR=runs/linux-full
```

`RESUME`, when set, is passed as `--resume` to the training CLI.

## Safety And Validation

The script uses `set -euo pipefail`, resolves its project root from its own
path, checks that the pair root exists and contains at least one `pair.json`,
and creates only local parent directories for the manifest and run directory.
It does not write to the mounted pair directory.

## Invocation

```bash
chmod +x scripts/train_linux.sh
./scripts/train_linux.sh
```

Environment variables can override any default path or the Python executable.
