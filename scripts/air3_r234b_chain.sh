#!/bin/bash
# Historical AIReduceNoise3 experiment launcher; paths and stages are tied to
# that archived run. Use README.md and caller-owned configs for new workflows.
# AIReduceNoise3 · 方案B 续跑链：r1b 收敛 -> r2b -> r3b -> r4b
#
# 每一轮严格串行、三步：
#   P) 会话目标（朴素递归）full.dn_3d_128b_rN.raw = M_{N-1}b(full)，teacher = 上一轮 best.pt
#   G) 门控（仅目标层；r2+ 没有混合层）：覆盖率 200 + immerkaer delta>0 + 相对削减 >=5%
#   T) 训练 rNb，--init-weights 上一轮 best.pt（热启动，opt/sched 冷启动）
#
# 用法（服务器 192.168.36.102，必须在 tmux 内跑，日志 /tmp/air3_r234b.log）：
#   tmux new-session -d -s air3_r234b \
#     'cd /home/yyc/workspace/N2N && bash scripts/air3_r234b_chain.sh 2>&1 | tee /tmp/air3_r234b.log'
#
# 可调：ROUNDS="2 3 4"（默认）可只跑部分轮；EXPECT_STUDIES=200。
#
# 幂等与隔离（铁律）：
#   * 先等 /tmp/air3_r1b.log 出现 `chain done` 且 EXIT_CODE=0；若 r1b 链失败/消失则立刻停止，不越轮。
#   * 锁文件 /tmp/air3_r234b.lock + PID 存活判断，防止同一条链被启动两次。
#   * 目标生成幂等（已存在的 full.<suffix>.raw 直接跳过）；每轮训练成功后写
#     <run_dir>/.chain_done，重跑时整轮跳过，不会重复训练。
#   * 任一阶段非 0 退出立即停链并打印 FAIL_*，绝不跳过门控进入训练。
#   * test 集永不进入教师目标生成（--splits train,val，脚本内部另有硬拦截）。
set -u

cd /home/yyc/workspace/N2N || exit 1
ulimit -n 8192   # 180 例 × 6 卷 memmap > 默认 1024

PY=/opt/anaconda3/envs/pytorch201/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MANIFEST=/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json
AIR3=/data/raid0/yyc/AIReduceNoise3
R1_LOG=/tmp/air3_r1b.log
LOCK=/tmp/air3_r234b.lock
ROUNDS=${ROUNDS:-"2 3 4"}
EXPECT=${EXPECT_STUDIES:-200}
WAIT_POLL=${WAIT_POLL:-300}

# ---- 单实例锁 ----------------------------------------------------------------
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=$old (lock $LOCK) -- refusing to start a second chain"; exit 1
  fi
  echo "[lock] stale lock pid=${old:-?} -> taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run_dir_for() { echo "runs/air3_mode2_1609_unet3d_h2f_128b_r$1"; }

echo "=== air3 r2..r4 chain start $(date '+%F %T') (pid $$) ==="
echo "manifest=$MANIFEST"
echo "rounds=$ROUNDS"

# ---- 阶段 0：等 r1b 收敛 ------------------------------------------------------
echo
echo "=== WAIT r1b convergence $(date '+%F %T') ==="
while true; do
  if grep -q '^chain done' "$R1_LOG" 2>/dev/null; then
    if grep -q '^EXIT_CODE=0' "$R1_LOG"; then
      echo "[wait] r1b converged -- $(grep '^EXIT_CODE=' "$R1_LOG" | tail -1) $(date '+%F %T')"
      break
    fi
    echo "FAIL_R1 $(date '+%F %T')"
    grep '^EXIT_CODE=' "$R1_LOG" | tail -1
    tail -20 "$R1_LOG"
    exit 1
  fi
  if ! tmux has-session -t air3_r1b 2>/dev/null; then
    echo "FAIL_R1_CHAIN_DISAPPEARED $(date '+%F %T')"
    tail -20 "$R1_LOG" 2>/dev/null
    exit 1
  fi
  echo "[wait] r1b training $(date '+%F %T') epochs=$(grep -c '^epoch=' "$R1_LOG" 2>/dev/null) | $(grep ' best$' "$R1_LOG" | tail -1)"
  sleep "$WAIT_POLL"
done

# ---- 逐轮 ---------------------------------------------------------------------
for R in $ROUNDS; do
  P=$((R - 1))
  PREV_RUN=$(run_dir_for "$P")
  RUN=$(run_dir_for "$R")
  CFG="configs/air3_mode2_1609_unet3d_h2f_128b_r${R}.yaml"
  SUFFIX="dn_3d_128b_r${R}"
  OVERVIEW="$AIR3/prepare_full_${SUFFIX}_overview.json"

  echo
  echo "=== ROUND r${R}b START $(date '+%F %T') teacher=$PREV_RUN/best.pt run=$RUN ==="

  if [ -f "$RUN/.chain_done" ]; then
    echo "[skip] r${R}b already finished ($RUN/.chain_done); moving on"
    continue
  fi
  if [ ! -f "$CFG" ]; then echo "FAIL_MISSING_CONFIG $CFG $(date '+%F %T')"; exit 1; fi
  if [ ! -f "$PREV_RUN/best.pt" ]; then echo "FAIL_MISSING_TEACHER $PREV_RUN/best.pt $(date '+%F %T')"; exit 1; fi

  echo "--- STAGE P: teacher target full.${SUFFIX}.raw $(date '+%F %T')"
  $PY scripts/prepare_full_dn_targets.py \
    --config "$CFG" \
    --manifest "$MANIFEST" \
    --checkpoint "$PREV_RUN/best.pt" --arch unet3d \
    --target-suffix "$SUFFIX" --splits train,val \
    --tiled --patch-hw 256 --overlap-d 16 --overlap-hw 16 \
    --batch-size 2 --device cuda:0
  code=$?
  echo "STAGE_P_EXIT=$code"
  if [ $code -ne 0 ]; then echo "FAIL_STAGE_P r${R}b $(date '+%F %T')"; exit 1; fi

  echo "--- GATE r${R}b $(date '+%F %T')"
  $PY scripts/check_r1b_target_gates.py \
    --prep-overview "$OVERVIEW" \
    --expect-studies "$EXPECT" \
    --label "r${R}b"
  code=$?
  echo "GATE_EXIT=$code"
  if [ $code -ne 0 ]; then echo "FAIL_GATE r${R}b $(date '+%F %T')"; exit 1; fi

  echo "--- STAGE T: r${R}b training $(date '+%F %T') init=$PREV_RUN/best.pt"
  $PY scripts/train_voxel.py --arch unet3d \
    --config "$CFG" \
    --manifest "$MANIFEST" \
    --run-dir "$RUN" \
    --init-weights "$PREV_RUN/best.pt"
  code=$?
  echo "TRAIN_EXIT=$code"
  if [ $code -ne 0 ]; then echo "FAIL_TRAIN r${R}b $(date '+%F %T')"; exit 1; fi

  touch "$RUN/.chain_done"
  echo "=== ROUND r${R}b DONE $(date '+%F %T') last_epoch=$(tail -1 "$RUN/history.jsonl" 2>/dev/null) ==="
done

echo
echo "ALL_ROUNDS_DONE $(date '+%F %T') last_run=$RUN"
echo "next: banded QC -> scripts/eval_rounds_banded.py (per density band; decide r2b/r3b/r4b delivery)"
echo "chain done $(date '+%F %T')"
