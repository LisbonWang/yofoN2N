# 递归训练方案：half → full_denoised（自蒸馏监督目标）

> 状态：已定义 + 已落地为脚本。方案落地于  
> `scripts/prepare_full_dn_targets.py`（教师降噪目标生成）与  
> `scripts/run_recursive_round.py`（单轮编排）。

## 1. 定义

**递归训练**（recursive / self-distillation on the supervision target）把  
`half → full` 有监督去噪拆成多轮，每一轮都用上一轮收敛模型的输出作为**更干净  
的监督目标**：

```
round 0:  M0 = train(even/odd → full)                    # 普通 half→full
round k:  教师 = M_{k-1}
          full.dn{k}.raw = M_{k-1}(full)                 # 教师对 full 降噪
          M_k = train(even/odd → full.dn{k}, 从 M_{k-1} 热启动)
```

- **输入不变**：始终是半视图 `even` / `odd`。
- **只有监督目标升级**：round 0 目标是 `full`，round k 目标是 `full.dn{k}`  
  （= 教师对 `full` 域推理出的"噪声更低"的全视图）。
- **关键认知**：`full` 本身含噪（不是干净真值），所以 `half → full` 里天然有  
  Noise2Noise 成分——模型对含噪目标拟合时会把目标里与输入无关的噪声平均掉，  
  输出可以比 `full` 更干净。递归的收益**不是** PSNR 更高，而是把监督目标的  
  残留噪声再压低一层；学生 `M_k` 学 `half → 噪声更低的目标`，去噪下限随之降低。

## 2. 关键规则（不可违反）

1. **热启动用 `--init-weights`，禁用 `--resume`**。新目标 `full.dn{k} ≈ full`，  
   本质是"上一收敛解附近的微调式目标升级"。`--init-weights` 只载入  
   `model.state_dict`，optimizer/scheduler 冷启动、lr 回 1e-4、epoch 从 0；  
   `--resume` 会错误带入上一轮的 Adam 动量 + 已衰减 lr + epoch 计数。
2. **test 永不进教师**。教师只对 `train`(+`val`) 的 `full` 降噪；`test` 保持  
   完全隔离，只用于最终评测（噪声水平为主，PSNR vs `full` 仅参考），否则会泄漏。
3. **轮次索引后缀**。round k 的目标卷写 `full.dn{k}.raw`（round 1 → `dn1`，  
   round 2 → `dn2` …），注册进 manifest `outputs` 的 `full.dn{k}.raw` 键，避免  
   相邻轮次互相覆盖。
4. **控制变量只换目标**。派生 config 只改 `data.mappings` 与 `training.run_dir`，  
   其余（arch/base_channels/depth/batch/lr/seed）与基线 `half→full` 完全一致。

## 3. 单轮流水线

```
python scripts/run_recursive_round.py \
  --config configs/1609v2_unet2d_ds_h2f.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise2/splits_1609v2.json \
  --teacher runs/1609v2_n2n_2d_ds_h2f/best.pt \
  --round 1
```

内部三步：

1. **目标生成**（`prepare_full_dn_targets.py`）：教师对 train+val 每个 study 的  
   `full.raw` 逐卷推理 → 写 `full.dn1.raw`（无头 LE float32，与 `full.raw` 同  
   目录）→ 幂等注册 manifest → 输出 `full.dn1` 的噪声水平诊断  
   （`prepare_full_dn1_overview.json`，含 noise_median/noise_immerkaer 对比）。  
   断点续跑安全（已存在的卷跳过）。
2. **派生 config**：写 `configs/1609v2_unet2d_ds_h2f_dn1.yaml`，mappings 变  
   `[even->full.dn1, odd->full.dn1]`，run_dir 变 `…_dn1`。
3. **打印训练命令**（在服务器 tmux 里执行）：
   ```
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/anaconda3/envs/pytorch201/bin/python \
     scripts/train_sparse_dense.py --arch unet2d_ds \
     --config configs/1609v2_unet2d_ds_h2f_dn1.yaml \
     --manifest /data/raid0/yyc/AIReduceNoise2/splits_1609v2.json \
     --run-dir runs/1609v2_n2n_2d_ds_h2f_dn1 \
     --init-weights runs/1609v2_n2n_2d_ds_h2f/best.pt
   ```

训练收敛后，`best.pt` 做 `--input-kind full` 评测（噪声水平为主）；若诊断显示  
`full.dn1` 的噪声水平确实低于 `full`（delta > 0），则用 `--round 2 --teacher <M1 best.pt>` 继续递归。

## 4. 诊断门控（决定要不要继续递归）

**判据是噪声水平，不是 PSNR。** `full` 本身含噪，`half→full` 天然带 Noise2Noise  
成分，所以相对含噪 `full` 的 PSNR 无法衡量去噪质量——目标变干净后，相对 `full`  
的 PSNR 甚至可能略降，这恰恰可能是有效的表现。

`prepare_full_dn_targets.py` 对每个 study 用两个稳健无参考估计器量化噪声水平：

- `estimate_noise_std`：局部去趋势残差的 `median(|r|)/0.6745`（只统计软组织，排除空气）。
- `estimate_noise_std_immerkaer`：Immerkaer 拉普拉斯核 RMS（独立交叉验证）。

输出到 `prepare_full_dn{k}_overview.json` 的 `noise_median` / `noise_immerkaer` 字段：

- `delta = noise(full) − noise(full.dn{k}) > 0` → 教师产出**噪声更低**的目标，本轮值得训练。
- `delta ≈ 0`（甚至 < 0）→ 教师没能把 `full` 压得更干净，本轮退化，不应投入训练。
- PSNR/SSIM vs `full` 仅作参考，**不作为门控依据**。

## 5. 与既有 E4 的关系（需用噪声水平重新评估，勿沿用 PSNR 结论）

本方案在数据 1（AIReduceNoise）上曾以 **E4** 名义试过，当时的结论是"负例"：

| 方案                    | half→full (E2) | half→full.dn (E4)   |
| --------------------- | -------------- | ------------------- |
| test PSNR（参考=含噪 full） | 39.68          | **38.62（-1.06 dB）** |

**但这个结论用了错误的判据**：它用"相对含噪 `full` 的 PSNR"衡量去噪，而 PSNR  
在这里本来就没有意义。E4 的 PSNR 比 E2 低 1.06dB，完全可能是 E4 输出**噪声更低**  
导致的（更干净 ≠ 更接近含噪 `full`）。当时"教师对 full 域外推理近似恒等  
（MAE 0.024）"也只是体素级改动量的观察，同样不能说明"没降噪"。

**正确做法**：用第 4 节的噪声水平估计重新评估 E4（以及当前数据 2 的教师），
看 `full.dn` 的噪声 std 是否真的低于 `full`，而不是看 PSNR/MAE delta。在拿到
噪声水平数据前，E4 的"负例"结论应视为**存疑、待重测**。

> 备选方向（记忆已记）：若"单遍教师降噪 full"确实无法压低噪声（噪声 delta ≈ 0），  
> 可改二阶蒸馏 `y2 = M(M(x))`（x 为 half，M(x) 落在模型训练域内、被真正降噪），  
> 其噪声下降才可能更显著。

## 6. 落地文件

- `scripts/prepare_full_dn_targets.py` — 教师降噪目标生成，支持  
  `unet2d / unet2d_ds / unet3d` 教师 + `--target-suffix`（轮次）+ `--splits`。
- `scripts/run_recursive_round.py` — 单轮编排（目标生成 → 派生 config → 打印命令）。
- `src/cbct_n2n/training/inference.py` — `infer_volume_3d` 迁入，供 3D 教师整卷推理复用。
- `src/cbct_n2n/training/metrics.py` — `estimate_noise_std` / `estimate_noise_std_immerkaer`  
  稳健无参考噪声水平估计器（递归收益的正确门控）。
- 数据集 `full.*` 目标白名单（`n2n_dataset.py` `_is_target_allowed` / `_resolve_output`）  
  为方案既有基础，本轮未改。

