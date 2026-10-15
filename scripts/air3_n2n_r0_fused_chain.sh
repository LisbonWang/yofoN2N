#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/air3_n2n_r0_fused_chain.sh MANIFEST N2N_CONFIG R0_CONFIG FUSED_CONFIG RUN_ROOT

Relative input and RUN_ROOT paths are resolved from the caller's current directory.

Environment:
  PYTHON                Python executable (default: python)
  CUDA_VISIBLE_DEVICES  GPU selection (optional; may be set by a scheduler)
  DEVICE                Teacher inference device (default: cuda:0; set cpu for CPU)

The manifest and configs must describe compatible records and supervision:
  N2N:   even->odd and odd->even
  R0:    even->full and odd->full
  fused: full->full.by_r0_by_n2n

This launcher creates RUN_ROOT/{n2n,r0,fused} and RUN_ROOT/chain.log. It
generates teacher targets only for train and val; the test split is excluded.
It refuses an existing RUN_ROOT. Target RAW files are written beside the
manifest-referenced full volumes by prepare_full_dn_targets.py.
EOF
}

if [[ $# -ne 5 ]]; then
  usage >&2
  exit 2
fi

CALLER_ROOT="$PWD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MANIFEST="$1"
N2N_CONFIG="$2"
R0_CONFIG="$3"
FUSED_CONFIG="$4"
RUN_ROOT="$5"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$CALLER_ROOT" "$1" ;;
  esac
}
MANIFEST="$(resolve_path "$MANIFEST")"
N2N_CONFIG="$(resolve_path "$N2N_CONFIG")"
R0_CONFIG="$(resolve_path "$R0_CONFIG")"
FUSED_CONFIG="$(resolve_path "$FUSED_CONFIG")"
RUN_ROOT="$(resolve_path "$RUN_ROOT")"

for path in "$MANIFEST" "$N2N_CONFIG" "$R0_CONFIG" "$FUSED_CONFIG"; do
  if [[ ! -r "$path" ]]; then
    echo "Required input is missing or unreadable: $path" >&2
    exit 2
  fi
done
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing run root: $RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/chain.log"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] N2N -> R0 -> fused chain started"
echo "[$(date -Is)] manifest: $MANIFEST"
echo "[$(date -Is)] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[$(date -Is)] teacher inference device: $DEVICE"

echo "[$(date -Is)] stage 1/5: train half->half N2N"
"$PYTHON" scripts/train_voxel.py \
  --config "$N2N_CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_ROOT/n2n"

echo "[$(date -Is)] stage 2/5: train half->full R0"
"$PYTHON" scripts/train_voxel.py \
  --config "$R0_CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_ROOT/r0"

echo "[$(date -Is)] stage 3/5: generate R0 targets for train,val"
"$PYTHON" scripts/prepare_full_dn_targets.py \
  --config "$R0_CONFIG" --manifest "$MANIFEST" \
  --checkpoint "$RUN_ROOT/r0/best.pt" --target-suffix by_r0 \
  --splits train,val --batch-size 1 --device "$DEVICE" --tiled

echo "[$(date -Is)] stage 4/5: generate N2N(R0(full)) targets for train,val"
"$PYTHON" scripts/prepare_full_dn_targets.py \
  --config "$N2N_CONFIG" --manifest "$MANIFEST" \
  --checkpoint "$RUN_ROOT/n2n/best.pt" --input-suffix by_r0 \
  --target-suffix by_r0_by_n2n --splits train,val \
  --batch-size 1 --device "$DEVICE" --tiled

echo "[$(date -Is)] stage 5/5: train fused student"
"$PYTHON" scripts/train_voxel.py \
  --config "$FUSED_CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_ROOT/fused"

echo "[$(date -Is)] N2N -> R0 -> fused chain completed"
