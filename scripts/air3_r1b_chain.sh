#!/bin/bash
# AIReduceNoise3 · 方案B r1b（blend-at-round-1）自动链：M0 目标 -> r1 混合目标 -> 门控 -> r1b 训练
#
# 用法（服务器 192.168.36.102，必须在 tmux 内跑，日志 /tmp/air3_r1b.log）：
#   tmux new-session -d -s air3_r1b \
#     'cd /home/yyc/workspace/N2N && bash scripts/air3_r1b_chain.sh 2>&1 | tee /tmp/air3_r1b.log'
#
# 幂等：A/B 两步都跳过已存在的输出文件，可安全重跑；门控失败即停止，不启动训练。
set -u

cd /home/yyc/workspace/N2N || exit 1
ulimit -n 8192   # 180 例 × 6 卷 memmap > 默认 1024

PY=/opt/anaconda3/envs/pytorch201/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MANIFEST=/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json
R0_RUN=runs/air3_mode2_1609_unet3d_h2f_128_r0
R1_RUN=runs/air3_mode2_1609_unet3d_h2f_128b_r1
AIR3=/data/raid0/yyc/AIReduceNoise3

echo "=== chain start $(date '+%F %T') ==="
echo "manifest=$MANIFEST"
echo "teacher=$R0_RUN/best.pt  run_dir=$R1_RUN"

echo
echo "=== STAGE A: M0(full) teacher target -> full.dn_3d_128b_r0.raw $(date '+%F %T') ==="
$PY scripts/prepare_full_dn_targets.py \
  --config configs/air3_mode2_1609_unet3d_h2f_128_r0.yaml \
  --manifest "$MANIFEST" \
  --checkpoint "$R0_RUN/best.pt" --arch unet3d \
  --target-suffix dn_3d_128b_r0 --splits train,val \
  --tiled --patch-hw 256 --overlap-d 16 --overlap-hw 16 \
  --batch-size 2 --device cuda:0
code=$?
echo "STAGE_A_EXIT=$code"
if [ $code -ne 0 ]; then echo "STAGE_A_FAILED $(date '+%F %T')"; exit 1; fi

echo
echo "=== STAGE B: r1 blend target -> full.dn_3d_128b_r1.raw $(date '+%F %T') ==="
$PY scripts/blend_dn_targets.py \
  --manifest "$MANIFEST" \
  --r1-suffix full --r2-suffix dn_3d_128b_r0 \
  --target-suffix dn_3d_128b_r1 \
  --lo 0.2 --hi 0.4 --kind smoothstep --splits train,val
code=$?
echo "STAGE_B_EXIT=$code"
if [ $code -ne 0 ]; then echo "STAGE_B_FAILED $(date '+%F %T')"; exit 1; fi

echo
echo "=== GATE $(date '+%F %T') ==="
$PY scripts/check_r1b_target_gates.py \
  --prep-overview "$AIR3/prepare_full_dn_3d_128b_r0_overview.json" \
  --blend-overview "$AIR3/blend_full_dn_3d_128b_r1_overview.json" \
  --expect-studies 200
code=$?
echo "GATE_EXIT=$code"
if [ $code -ne 0 ]; then echo "GATE_FAILED $(date '+%F %T')"; exit 1; fi

echo
echo "=== STAGE C: r1b training (init-weights=$R0_RUN/best.pt) $(date '+%F %T') ==="
$PY scripts/train_voxel.py --arch unet3d \
  --config configs/air3_mode2_1609_unet3d_h2f_128b_r1.yaml \
  --manifest "$MANIFEST" \
  --run-dir "$R1_RUN" \
  --init-weights "$R0_RUN/best.pt"
code=$?
echo "EXIT_CODE=$code"
echo "chain done $(date '+%F %T')"
exit $code
