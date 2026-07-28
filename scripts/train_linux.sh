#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PAIR_ROOT="${PAIR_ROOT:-/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/splits-linux.json}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.yaml}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/linux-full}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$PAIR_ROOT" ]]; then
    echo "Pair root does not exist: $PAIR_ROOT" >&2
    exit 1
fi
if ! find "$PAIR_ROOT" -mindepth 2 -maxdepth 2 -name pair.json -print -quit | grep -q .; then
    echo "No pair.json files found below: $PAIR_ROOT" >&2
    exit 1
fi

mkdir -p "$(dirname "$MANIFEST")" "$RUN_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_splits.py" "$PAIR_ROOT" "$MANIFEST"

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
    resume_args=(--resume "$RESUME")
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train.py" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --run-dir "$RUN_DIR" \
    "${resume_args[@]}"
