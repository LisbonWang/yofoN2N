#!/usr/bin/env bash
# Expert-review package, phase 3/3: guided-filter sharpening of the AI rounds.
#
# Two targets, same parameters for both:
#   volume_..._f32_ai_r4b.raw -> volume_..._f32_ai_r4b_sharp.raw
#   volume_..._f32_ai_r3b.raw -> volume_..._f32_ai_r3b_sharp.raw
#
# strength 0.3, wiener mode, r=2, eps-k=8, sigma taken from the auto-picked flat
# ROI of the volume being sharpened (the calibrated option: the whole-tissue
# Immerkaer sigma overestimates the noise floor 1.31-5.29x depending on the case
# and would make k meaningless).  Bone gain is ~94% x strength (measured
# linearity), so s=0.3 -> about +28% bone contrast at a flat-region noise gain of
# 1 + 0.3/(1+64) ~= 1.005.
#
# The ROI is picked per volume on purpose: eps is meant to follow each volume's
# own noise floor, so the lighter-smoothing r3b gets a slightly larger eps than
# r4b at the same strength.  Within a run "before" and "after" are always the
# same voxels, so the gain columns are comparable; the absolute sigma is not
# (that is already the project-wide rule for Immerkaer numbers).
#
# Case folders stay raw-only: figures / json / markdown of this step go to _logs/.
#
# Re-runs unconditionally (guided_sharpen_volume.py has no cache check), so use
# ONLY="02 03" to redo a subset of cases and ONLY_TARGETS="ai_r3b" to redo a
# subset of targets instead of sharpening all eighteen volumes again.
set -u
export PATH="/usr/bin:/bin:$PATH"
export PYTHONIOENCODING=utf-8

# Space-separated case indices to run; empty = every case.
ONLY="${ONLY:-}"
# Space-separated input tags to run; empty = every target.
ONLY_TARGETS="${ONLY_TARGETS:-}"

PY=/c/Users/admin/anaconda3/envs/torch/python.exe
REPO=/g/workspace/N2N
REPO_W=G:/workspace/N2N
OUTROOT=$REPO/outputs/expert_review_ai_vs_conv
OUTROOT_W=$REPO_W/outputs/expert_review_ai_vs_conv
LOGS=$OUTROOT/_logs
LOGS_W=$OUTROOT_W/_logs

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

# in-tag | out-tag | artefact stem in _logs/gfsharp/cNN | figures dir name.
# The r4b pair keeps the original flat names so artefacts from the first build
# are not orphaned.
TARGETS=(
"ai_r4b|ai_r4b_sharp|gfsharp_s0.3|figures"
"ai_r3b|ai_r3b_sharp|gfsharp_r3b_s0.3|figures_r3b"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r idx shape <<< "$entry"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $idx "* ]]; then continue; fi
  set -- $shape                       # z y x
  BASE="volume_${3}X${2}X${1}_f32"
  NOISY=${BASE}.raw
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  if [ -z "$DIR" ]; then echo "[miss] c${idx}: no directory"; continue; fi
  base=$(basename "$DIR")
  DIR_W="$OUTROOT_W/$base"
  for tentry in "${TARGETS[@]}"; do
    IFS='|' read -r in_tag out_tag stem figdir <<< "$tentry"
    if [ -n "$ONLY_TARGETS" ] && [[ " $ONLY_TARGETS " != *" $in_tag "* ]]; then continue; fi
    IN_NAME=${BASE}_${in_tag}.raw
    OUT_NAME=${BASE}_${out_tag}.raw
    work="$LOGS/gfsharp/c${idx}"
    work_w="$LOGS_W/gfsharp/c${idx}"
    mkdir -p "$work/$figdir"
    log="$LOGS/gfsharp_c${idx}_${in_tag}.log"
    if [ ! -f "$DIR/$IN_NAME" ]; then echo "[miss] c${idx} $in_tag: $IN_NAME absent"; continue; fi
    echo "=== c${idx} $base  ${in_tag} -> ${out_tag}  shape=$shape ==="
    t0=$(date +%s)
    "$PY" "$REPO_W/scripts/guided_sharpen_volume.py" \
      --dir "$DIR_W" \
      --input "$IN_NAME" \
      --shape $shape \
      --reference "noisy=$NOISY" \
      --radius 2 --eps-k 8 --sigma-source flat --strength 0.3 --mode wiener --slab 96 \
      --out "$DIR_W/$OUT_NAME" \
      --figures "$work_w/$figdir" \
      --json    "$work_w/${stem}.json" \
      --markdown "$work_w/${stem}.md" \
      >"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -eq 0 ]; then
      echo "[ok] c${idx} ${out_tag}  $((t1-t0))s"
      grep -E '^\[noise\]|^\[cfg\]|^\[raw\]|^\[roi\] (flat|bone)' "$log"
    else
      echo "[FAIL] c${idx} ${out_tag} rc=$rc (see $log)"; tail -8 "$log"
    fi
  done
done

echo
echo "===================== phase 3 summary ====================="
for entry in "${CASES[@]}"; do
  IFS='|' read -r idx shape <<< "$entry"
  set -- $shape
  DIR=$(ls -d "$OUTROOT/c${idx}_"* 2>/dev/null | head -1)
  for tentry in "${TARGETS[@]}"; do
    IFS='|' read -r _in_tag out_tag _stem _figdir <<< "$tentry"
    f="$DIR/volume_${3}X${2}X${1}_f32_${out_tag}.raw"
    if [ -f "$f" ]; then printf 'c%s  %-16s %14d  %s\n' "$idx" "$out_tag" "$(stat -c%s "$f")" "$f"
    else printf 'c%s  %-16s MISSING %s\n' "$idx" "$out_tag" "$f"; fi
  done
done
echo "sharpen phase done"
