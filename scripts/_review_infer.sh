#!/usr/bin/env bash
# Expert-review package, phase 2/3: AI inference (r1b..r4b) on the chain-A volume.
#
# Each case directory already holds volume_<X>X<Y>X<Z>_f32.raw (PWLS 0.01 + BHC +
# RCoor + MARFI).  Every checkpoint is run over the identical in-memory array, so
# the four outputs differ only by model.  Names get the "ai" tag the review package
# asks for, joined with "_" (--suffix-sep) so the folder stays flat:
#   volume_640X640X360_f32.raw -> volume_640X640X360_f32_ai_r1b.raw ... _ai_r4b.raw
#
# Idempotent: denoise_multi_models.py skips an existing <stem><sep><model>.raw.
#
# ONLY="02 03" limits the run to those case indices (default: all).  A cached case is
# cheap to re-scan but still re-reads the volumes, so the filter saves real time when
# only one reconstruction changed.
set -u
export PATH="/usr/bin:/bin:$PATH"
export PYTHONIOENCODING=utf-8
export KMP_DUPLICATE_LIB_OK=TRUE

ONLY="${ONLY:-}"

PY=/c/Users/admin/anaconda3/envs/torch/python.exe
REPO=/g/workspace/N2N
REPO_W=G:/workspace/N2N
OUTROOT=$REPO/outputs/expert_review_ai_vs_conv
OUTROOT_W=$REPO_W/outputs/expert_review_ai_vs_conv
CK=$REPO_W/outputs/air3_ypdl0027_bhc_5models/checkpoints
LOGS=$OUTROOT/_logs
mkdir -p "$LOGS"

CONFIG=$REPO_W/configs/air3_mode2_1609_unet3d_h2f_128b_r4.yaml
SEP=_

MODELS=(--model "ai_r1b=$CK/r1b.best.pt"
        --model "ai_r2b=$CK/r2b.best.pt"
        --model "ai_r3b=$CK/r3b.best.pt"
        --model "ai_r4b=$CK/r4b.best.pt")

# idx | group | shape(z y x);  DIMS for the file name is the reverse (x X y X z)
SHAPES=(
"01|Mode0_1615|600 640 640"
"02|Mode1_1615|600 640 640"
"03|Mode1_1615|600 640 640"
"04|Mode2_1609|360 640 640"
"05|Mode2_1609|360 640 640"
"06|Mode2_1609|360 640 640"
"07|Mode3_1609|360 640 640"
"08|Mode3_1609|360 640 640"
"09|Mode3_1609|360 640 640"
)

for entry in "${SHAPES[@]}"; do
  IFS='|' read -r idx group shape <<< "$entry"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $idx "* ]]; then continue; fi
  set -- $shape                       # z y x
  BASE="volume_${3}X${2}X${1}_f32"
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  if [ -z "$DIR" ]; then echo "[miss] c${idx}: no directory"; continue; fi
  DIR_W="$OUTROOT_W/$(basename "$DIR")"
  log="$LOGS/infer_c${idx}.log"
  if [ ! -f "$DIR/${BASE}.raw" ]; then
    echo "[miss] c${idx}: ${BASE}.raw absent (recon phase not finished?)"
    continue
  fi
  echo "=== c${idx} $(basename "$DIR")  shape=$shape ==="
  t0=$(date +%s)
  "$PY" "$REPO_W/scripts/denoise_multi_models.py" \
    --input "$DIR_W/${BASE}.raw" \
    --shape $shape \
    --config "$CONFIG" \
    "${MODELS[@]}" \
    --suffix-sep "$SEP" \
    --tiled --voxel-depth 128 --patch-hw 256 --overlap-d 16 --overlap-hw 16 \
    --batch-size 2 --band-key ai_r1b --lo 0.2 --hi 0.4 \
    --summary "$DIR_W/metrics_ai.json" >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "[ok] c${idx}  $((t1-t0))s"
    sed -n '/\[bands\]/,$p' "$log" | head -12
  else
    echo "[FAIL] c${idx} rc=$rc (see $log)"; tail -8 "$log"
  fi
done

echo
echo "===================== phase 2 summary ====================="
for entry in "${SHAPES[@]}"; do
  IFS='|' read -r idx group shape <<< "$entry"
  set -- $shape
  BASE="volume_${3}X${2}X${1}_f32"
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  printf 'c%s %s\n' "$idx" "$(basename "${DIR:-?}")"
  ls -la "$DIR"/${BASE}_ai_*.raw 2>/dev/null | awk '{printf "      %14d  %s\n", $5, $9}'
done
echo "inference phase done"
