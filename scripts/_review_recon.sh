#!/usr/bin/env bash
# Historical private-data review helper for one PDL network share.
# Expert-review package, phase 1/3: reconstruction.
#
# Builds, for each of 9 held-out (test11) studies, two volumes out of the *same*
# scanner configuration:
#   chain A  --post-procs "BHC RCoor MARFI"                              -> AI input
#   chain B  --post-procs "BHC ReduceNoise Anisotropic RCoor MARFI"      -> conventional denoise reference
# both with --pwls-scale 0.01 (beta = 0.01 x brightness mean) + the Hamming window
# that the patch applies together with PWLS.
#
# Delivery names are flat, one extension: volume_<X>X<Y>X<Z>_f32.raw (chain A) and
# volume_<X>X<Y>X<Z>_f32_traditional.raw (chain B), X/Y/Z = fdkPara xNum/yNum/zNum.
#
# Idempotent: recon_conventional_denoise.py skips an --out that already exists.
# ONLY="02 03" limits the run to those case indices (default: all).
set -u
export PATH="/usr/bin:/bin:$PATH"
export XDAY_DIR="D:/XDay Release/XDay 617.2_polyc"
export PYTHONIOENCODING=utf-8

ONLY="${ONLY:-}"

PY=/c/Users/admin/anaconda3/python.exe
REPO=/g/workspace/N2N
# Windows-style repo root for arguments handed to python.exe: git-bash rewrites a
# bare /g/... into g:\g\..., so the script path and --out must be given as G:/...
REPO_W=G:/workspace/N2N
OUTROOT=$REPO/outputs/expert_review_ai_vs_conv
OUTROOT_W=$REPO_W/outputs/expert_review_ai_vs_conv
LOGS=$OUTROOT/_logs
mkdir -p "$LOGS"

SOURCE_ROOT='//192.168.36.235/团队文件-牙科ct/data_raw/PDL'

CHAIN_A="BHC RCoor MARFI"
CHAIN_B="BHC ReduceNoise Anisotropic RCoor MARFI"

# idx | group | case_id | study_uid | dims (XxYxZ) | extra --filter-file (empty = group standard)
CASES=(
"01|Mode0_1615|YPDL-25B-0006|1.2.156.89797079.788644816.1217986243.4033654926|640X640X600|$SOURCE_ROOT/YPDL-25B-0006/1.2.156.89797079.788644816.1217986243.4033654926/config/recon/PDI1313/Mode0/filter.xml"
"02|Mode1_1615|YPDL-25B-0005|1.2.156.89797079.3354951403.1258185007.652073354|640X640X600|"
"03|Mode1_1615|YPDL-26A-0026|1.2.826.0.1.3680043.2.461.16122501.776344465|640X640X600|"
"04|Mode2_1609|YPDL-00A-0000|1.3.6.1.4.1.30071.8.53625245697634.8415609725064887.26|640X640X360|"
"05|Mode2_1609|YPDL-26A-0009|1.2.156.89797079.2335316951.1203213444.1392639632|640X640X360|"
"06|Mode2_1609|YPDL-26A-0027|1.2.156.89797079.1182197106.1232956091.447186561|640X640X360|"
"07|Mode3_1609|YPDL-26A-0026|1.2.156.89797079.835519821.1189836059.1130746797|640X640X360|"
"08|Mode3_1609|YPDL-26A-0026|1.2.826.0.1.3680043.2.461.16167459.1916027858|640X640X360|"
"09|Mode3_1609|YPDL-26A-0026|1.2.826.0.1.3680043.2.461.16142894.3829726010|640X640X360|"
)

recon () {  # tag chain out_win  [filter-file]
  local tag="$1" chain="$2" out="$3" ff="${4:-}"
  local log="$LOGS/recon_${tag}.log"
  if [ -f "$(printf '%s' "$out" | sed 's|^G:/|/g/|')" ]; then
    echo "[skip] $tag (already exists)"
    return 0
  fi
  local args=(--case "$CASE" --study "$STUDY" --group "$GROUP"
              --post-procs "$chain" --pwls-scale 0.01 --out "$out")
  [ -n "$ff" ] && args+=(--filter-file "$ff")
  local t0 t1
  t0=$(date +%s)
  if "$PY" "$REPO_W/scripts/recon_conventional_denoise.py" "${args[@]}" >"$log" 2>&1; then
    t1=$(date +%s)
    echo "[ok]   $tag  $((t1-t0))s  $(grep -o 'wrote .*' "$log" | tail -1)"
  else
    echo "[FAIL] $tag  (see $log)"
    tail -5 "$log"
  fi
}

for entry in "${CASES[@]}"; do
  IFS='|' read -r idx GROUP CASE STUDY DIMS FF <<< "$entry"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $idx "* ]]; then continue; fi
  BASE="volume_${DIMS}_f32"
  TAIL="${STUDY##*.}"
  DIR="$OUTROOT/c${idx}_${GROUP}__${CASE}__${TAIL}"
  DIR_W="$OUTROOT_W/c${idx}_${GROUP}__${CASE}__${TAIL}"
  mkdir -p "$DIR"
  echo "##############################################################"
  echo "### c${idx}  ${GROUP}  ${CASE}  ${STUDY}"
  echo "### dir: $DIR"
  echo "##############################################################"
  recon "c${idx}_A" "$CHAIN_A" "$DIR_W/${BASE}.raw" "$FF"
  recon "c${idx}_B" "$CHAIN_B" "$DIR_W/${BASE}_traditional.raw" "$FF"
done

echo
echo "===================== phase 1 summary ====================="
for entry in "${CASES[@]}"; do
  IFS='|' read -r idx GROUP CASE STUDY DIMS FF <<< "$entry"
  TAIL="${STUDY##*.}"
  DIR="$OUTROOT/c${idx}_${GROUP}__${CASE}__${TAIL}"
  printf 'c%s %s\n' "$idx" "$DIR"
  ls -la "$DIR"/*.raw 2>/dev/null | awk '{printf "      %14d  %s\n", $5, $9}'
done
echo "recon phase done"
