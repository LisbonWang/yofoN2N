#!/usr/bin/env bash
# Expert-review package, phase 3/3: guided-filter sharpening of the r4b output.
#
# strength 0.3, wiener mode, r=2, eps-k=8, sigma taken from the auto-picked flat
# ROI of each volume (the calibrated option: the whole-tissue Immerkaer sigma
# overestimates the noise floor 1.31-5.29x depending on the case and would make k
# meaningless).  Bone gain is ~94% x strength (measured linearity), so s=0.3 ->
# about +28% bone contrast at a flat-region noise gain of 1 + 0.3/(1+64) ~= 1.005.
#
# Case folders stay raw-only: figures / json / markdown of this step go to _logs/.
#
# Names (flat, one extension): in  volume_<X>X<Y>X<Z>_f32_ai_r4b.raw
#                              out volume_<X>X<Y>X<Z>_f32_ai_r4b_sharp.raw
#
# Re-runs unconditionally (guided_sharpen_volume.py has no cache check), so use
# ONLY="02 03" to redo a subset instead of sharpening all nine again.
set -u
export PATH="/usr/bin:/bin:$PATH"
export PYTHONIOENCODING=utf-8

# Space-separated case indices to run; empty = every case.
ONLY="${ONLY:-}"

PY=/c/Users/admin/anaconda3/envs/torch/python.exe
REPO=/g/workspace/N2N
REPO_W=G:/workspace/N2N
OUTROOT=$REPO/outputs/expert_review_ai_vs_conv
OUTROOT_W=$REPO_W/outputs/expert_review_ai_vs_conv
LOGS=$OUTROOT/_logs

# idx | shape(z y x);  DIMS for the file name is the reverse (x X y X z)
CASES=(
"01|600 640 640"
"02|600 640 640"
"03|600 640 640"
"04|360 640 640"
"05|360 640 640"
"06|360 640 640"
"07|360 640 640"
"08|360 640 640"
"09|360 640 640"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r idx shape <<< "$entry"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $idx "* ]]; then continue; fi
  set -- $shape                       # z y x
  BASE="volume_${3}X${2}X${1}_f32"
  IN_NAME=${BASE}_ai_r4b.raw
  OUT_NAME=${BASE}_ai_r4b_sharp.raw
  NOISY=${BASE}.raw
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  if [ -z "$DIR" ]; then echo "[miss] c${idx}: no directory"; continue; fi
  base=$(basename "$DIR")
  DIR_W="$OUTROOT_W/$base"
  work="$LOGS/gfsharp/c${idx}"
  mkdir -p "$work/figures"
  log="$LOGS/gfsharp_c${idx}.log"
  if [ ! -f "$DIR/$IN_NAME" ]; then echo "[miss] c${idx}: $IN_NAME absent"; continue; fi
  echo "=== c${idx} $base  shape=$shape ==="
  t0=$(date +%s)
  "$PY" "$REPO_W/scripts/guided_sharpen_volume.py" \
    --dir "$DIR_W" \
    --input "$IN_NAME" \
    --shape $shape \
    --reference "noisy=$NOISY" \
    --radius 2 --eps-k 8 --sigma-source flat --strength 0.3 --mode wiener --slab 96 \
    --out "$DIR_W/$OUT_NAME" \
    --figures "G:/workspace/N2N/outputs/expert_review_ai_vs_conv/_logs/gfsharp/c${idx}/figures" \
    --json    "G:/workspace/N2N/outputs/expert_review_ai_vs_conv/_logs/gfsharp/c${idx}/gfsharp_s0.3.json" \
    --markdown "G:/workspace/N2N/outputs/expert_review_ai_vs_conv/_logs/gfsharp/c${idx}/gfsharp_s0.3.md" \
    >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "[ok] c${idx}  $((t1-t0))s"
    grep -E '^\[noise\]|^\[cfg\]|^\[raw\]' "$log"
  else
    echo "[FAIL] c${idx} rc=$rc (see $log)"; tail -8 "$log"
  fi
done

echo
echo "===================== phase 3 summary ====================="
for entry in "${CASES[@]}"; do
  IFS='|' read -r idx shape <<< "$entry"
  set -- $shape
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  f="$DIR/volume_${3}X${2}X${1}_f32_ai_r4b_sharp.raw"
  if [ -f "$f" ]; then printf 'c%s  %14d  %s\n' "$idx" "$(stat -c%s "$f")" "$f"
  else printf 'c%s  MISSING %s\n' "$idx" "$f"; fi
done
echo "sharpen phase done"
