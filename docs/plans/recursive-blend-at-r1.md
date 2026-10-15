# 最终方案更新：N2N 降噪与 R0 抗混叠蒸馏融合

> **AIReduceNoise3 实验快照（2026-10-14 整理）**：以下数据、服务器和路径只适用于原始实验环境，具体值为历史记录。通用训练步骤见仓库根目录 `README.md`。

> 状态：**新最终方案已确认**。本方案取代下方历史方案 B 的 r0→r4b 递归链；旧链结果保留作对照，不再作为当前训练计划。

## 1. 目标

训练两个同参数规模的教师模型，分别学习 Noise2Noise 降噪和 half-view 到 full-view 的抗混叠映射；再将它们串联生成教师目标，蒸馏到一个同参数规模的学生模型，使单模型同时具备抗混叠与强 N2N 降噪能力。

## 2. 三阶段训练流程

| 阶段 | 模型 | 输入→监督目标 | 学习目标 |
| --- | --- | --- | --- |
| 1 | `N2N` | `even→odd` 与 `odd→even` | half→half Noise2Noise，学习强去噪 |
| 2 | `R0` | `even→full` 与 `odd→full` | half→full，学习抗混叠 |
| 3 | `Fused`（蒸馏学生） | `full→N2N(R0(full))` | 将 R0 的抗混叠结果与 N2N 的强去噪融合到单模型 |

第三阶段的教师目标按顺序生成：先对全视图重建运行 `R0`，再把 `R0(full)` 送入 `N2N`，即 `target = N2N(R0(full))`。学生直接以原始 `full` 为输入，拟合该组合教师的输出。

三个模型使用相同架构和参数规模。沿用当前 AIReduceNoise3 `Mode2_1609` 训练计划的模型规格：UNet3D、`base_channels=16`、`depth=4`；输入窗口 D=128、stride=112、H/W patch=256。N2N、R0 与学生各有独立 checkpoint。除监督映射和 run 目录外，训练超参沿用同一基准配置，保证参数规模与优化设置可比较。

## 3. 数据与隔离

- 训练范围沿用 AIReduceNoise3 `Mode2_1609` 的 200 个 study，train/val=180/20；跨模式 11 例独立测试集只用于最终评估。
- 沿用四域训练：`full`、`full.bhc`、`full.marfi`、`full.marfi.bhc`。half-view 对及 full-view 目标必须在同一 study 内构造，study 级 split 不变。
- 第三阶段教师目标只为 train/val 生成；test 不运行教师，也不参与蒸馏。
- N2N 教师必须以训练时相同的 half-view 输入分布训练。第三阶段将 `R0(full)` 作为 N2N 输入，是方案指定的串联推理步骤；生成目标时记录输入/输出范围并做抽样视觉检查。

## 4. 训练和验收

1. 训练 `N2N` 与 `R0` 两个教师模型，分别保存最佳 checkpoint，并确认验证集收敛。
2. 用 `R0` 处理 train/val 的 `full`，再将结果输入 `N2N`，生成并登记 `full_by_R0_by_N2N` 目标。
3. 训练 `Fused` 学生：输入 `full`，目标为 `full_by_R0_by_N2N`，参数规模与两个教师一致。
4. 在独立测试集上并列评估 half 输入下的 N2N、R0，以及 full 输入下的蒸馏学生。除高频/分带指标外，必须做同层视觉质检，检查抗混叠伪影、纹理保留和过平滑；不得把教师目标或 full 重建视为干净真值。

### 可复现命令

以下命令在 AIR3 训练服务器执行。先训练两个教师：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_voxel.py \
  --config configs/air3_mode2_1609_unet3d_n2n.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_voxel.py \
  --config configs/air3_mode2_1609_unet3d_r0.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json
```

再按顺序生成 `R0(full)` 和 `N2N(R0(full))`。目标生成脚本默认只处理 train/val，禁止将 test 加入 `--splits`：

```bash
python scripts/prepare_full_dn_targets.py \
  --config configs/air3_mode2_1609_unet3d_r0.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json \
  --checkpoint runs/air3_mode2_1609_unet3d_r0/best.pt \
  --target-suffix by_r0 --splits train,val --tiled --patch-hw 256 \
  --overlap-d 16 --overlap-hw 16 --batch-size 2 --device cuda:0

python scripts/prepare_full_dn_targets.py \
  --config configs/air3_mode2_1609_unet3d_n2n.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json \
  --checkpoint runs/air3_mode2_1609_unet3d_n2n/best.pt \
  --input-suffix by_r0 --target-suffix by_r0_by_n2n \
  --splits train,val --tiled --patch-hw 256 --overlap-d 16 \
  --overlap-hw 16 --batch-size 2 --device cuda:0
```

最后从随机初始化训练蒸馏学生，目标键为 `full.by_r0_by_n2n.raw`：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_voxel.py \
  --config configs/air3_mode2_1609_unet3d_fused.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json
```

## 5. 实现约束

`N2NDataset` 已支持 `full` 映射源和 `full.*` 监督目标；`prepare_full_dn_targets.py --input-suffix` 可从已登记的中间体继续运行教师。第三阶段可直接使用 `train_voxel.py` 和 `configs/air3_mode2_1609_unet3d_fused.yaml`。

旧方案 B 的计划与执行记录从下一节起保留，仅供追溯，不代表当前方案。

---

## 历史方案：方案 B（r1 混合递归，blend-at-round-1）

> 已被上面的三阶段 N2N/R0 蒸馏方案取代。其训练、验收及结论仅作为历史记录。

### 历史执行背景

原状态：AIReduceNoise3 r0 于 2026-09-17 在服务器 GPU 0 以既有 r0 best.pt 热启动；方案 A（`r3-density-blend.md`）曾降级为对照。

## 1. 最终训练数据范围

训练根目录：服务器 `/data/raid0/yyc/AIReduceNoise3/PDL`（Windows 映射
`Z:\raid0\yyc\AIReduceNoise3\PDL`）。最终训练数据进一步收缩为**仅 `Mode2_1609`**，体素尺度0.25 mm、体尺寸360×640×640；Mode0、Mode1、Mode3、0808及全部1615均不进入本轮训练。

训练/验证清单固定为 `splits_mode2_1609_balanced200_trainval.json`：不限制设备名称前缀，
`Mode2_1609` 目录下所有设备均可入选，但每例必须六卷（even/odd + 四域full）齐全。
共13台设备、529例完整study；采用容量受限的均衡分配，优先向当前入选数最少且仍有余量的
设备补样。两台设备仅有15例和1例，故全取；其余11台分担184例，最终为8台×17例、
3台×16例，合计200个不重复study。固定随机种子42，train/val=180/20，内部test为空。

独立测试清单为 `splits_cross_mode_test11.json`，固定11例：`Mode0_1615` 2例、
`Mode1_1615` 3例、`Mode2_1609` 3例、`Mode3_1609` 3例。Mode2测试3例与200例训练/验证
清单按复合key严格互斥；Mode0/1仅使用四域full作跨模式部署评估，不作为Noise2Noise训练输入。

每条链均做**四域联合**，单个基础训练对保留原始域并附加三个后处理残差域：

1. `full`；
2. `full.bhc`；
3. `full.marfi`；
4. `full.marfi.bhc`（用户口径 `full.bhc.marfi`）。

配置写法统一为 `residuals: [bhc, marfi]`：当前 Dataset 会保留 base pair，并自动
追加 `bhc`、`marfi`、`marfibhc` 三个 residual pair，恰好得到四域；不拆四套模型。
训练/验证按 study 划分，四域及同一 study 的不同重建组必须绑定在同一 split，禁止泄漏。

### 1.1 1615 排除规则

Mode1 双源拆分的单例冒烟结果存在伪影：拆分后每支 510 帧虽然索引与矩阵一致，
但不满足 XDay 内部对单源投影数量/排列的隐含要求。因此原先的程序级验收不能作为
图像正确性的依据，`Mode1_1615` 不进入训练、验证或测试 manifest；Mode0 同样排除。
在根因与 SDK 约束完全查清并重新通过图像级验收前，不恢复任何 1615 训练计划。

训练/验证数据抽取白名单仅为 `Mode2_1609`。生成后必须断言其 `records.group` 全部
严格等于 `Mode2_1609`，200个key不重复，train/val=180/20且精确覆盖全部key，1200个所需
RAW文件（200×6）均真实存在。独立测试manifest必须满足2/3/3/3组配额、11个test key完整覆盖，
且其中3个Mode2 key不出现在训练/验证清单中。

## 2. 方案定义（Mode2_1609 单链执行）

| 轮 | 监督目标 | 说明 |
| -- | -------- | ---- |
| r0 | `full` | 标准 half→full；须在 AIReduceNoise3 的 Mode2_1609 200例清单上重新训练，AIReduceNoise2 权重只可作为初始化候选，不作为最终 r0 |
| r1 | `full·(1−w) + full_by_r0·w` | 混合只在这一轮做；`w = smoothstep(μ)`，μ∈[0.2,0.4] 对应 0→1，**键 = full_by_r0 的强度**（去噪后，无抖动） |
| r2 | `full_by_r1` | 回归朴素递归（教师 = r1 模型对 full 的输出） |
| r3 | `full_by_r2` | 同上，教师 = r2 模型 |
| r4 | `full_by_r3` | 同上，教师 = r3 模型（2026-09-18 追加：本轮一路跑到 r4b，见 §11） |

软组织带（μ≤0.2）w=0 → 目标=原始 `full`；骨带（μ≥0.4）w=1 → 目标=`full_by_r0`。

## 3. 机制：混合在 r1 做，密度分级深度自动向后传播

r1 之后每轮都是朴素递归（half → 上一轮教师输出），而教师输出**本身就携带
分带噪声差**：r1 输出软组织 ≈ full 压一层、骨 ≈ M0 输出再压一层。后续每轮
N2N 压缩是逐带近似乘性的，所以带间差距会一直保持并放大——**不需要再混合目标**。

按"每带有效压缩层数"计（N2N 每轮把目标噪声独立分量平均掉一层）：

| 轮 | 软组织 | 骨 |
| -- | ------ | -- |
| r1b | 1（从 full 起） | 2（从 M0 输出起） |
| r2b | 2 | 3 |
| r3b | 3 | 4 |
| r4b | 4 | 5 |

## 4. 与方案 A（已训完）的关系——诚实的等价性分析

方案 A 的 r3：软组织目标 = M1 输出（2 层）→ 学生 3 层；骨目标 = M2 输出（3 层）
→ 学生 4 层。**终点深度结构与方案 B 的 r3 完全相同（软 3 / 骨 4）**，两条链是
同一终点的两种路径：

- 方案 A：先全局递归两层（软组织也跟着深了），最后在 r3 用混合把软组织"拉回"；
- 方案 B：一开始就让软组织浅一轮，之后朴素递归自然追赶。

因此**预期终点数值与现链接近**（软 ~0.0040、骨 ~0.0039 量级；骨区第 4 层压缩的
实测收益在方案 A 里只有 −10%，方案 B 不会魔法般更大）。方案 B 的真实收益：

1. **流程简化**：目标混合只做一次（纯 numpy、分钟级、无模型），之后每轮都是
   标准递归，没有 r3 那套"目标层/输出层两层门控 + off-by-one"的心智负担；
2. **中间模型即分级模型库**：r1b = 软组织浅/骨深的轻量档，r2b/r3b 逐档加深，
   临床可按剂量/部位选档，而不必只依赖终点模型；
3. **可随时提前停靠**：若骨区在第 4 层出现过平滑，停在 r2b（软 2/骨 3）即可，
   损失可控。

若目标是"骨区显著深于现链 r3"，正确杠杆是**在方案 B 上再加 r4**（朴素递归第 4
轮，软 4/骨 5），而不是换混合位置——递归收益递减，需用分带 σ 实测决定停在哪轮。
**r4b 已纳入执行**（2026-09-18）：链一路跑到 r4b 收敛，交付哪一轮由 r4b 收敛后的
分带验收裁定，默认交付档位仍是 r1b/r2b/r3b。

## 5. 命名（避免与已训完的旧链冲突）

旧链（方案 A）目标文件 `full.dn_3d_128_rN.raw`、run `..._128_dn_3d_128_rN` **全部保留不动**。
新链统一加链标签 **`128b`**：

| 产物 | 文件/目录 |
| ---- | --------- |
| r1 目标 | `full.dn_3d_128b_r1.raw`（= blend(full, M0(full))） |
| r2 目标 | `full.dn_3d_128b_r2.raw`（= M1b(full)） |
| r3 目标 | `full.dn_3d_128b_r3.raw`（= M2b(full)） |
| r4 目标 | `full.dn_3d_128b_r4.raw`（= M3b(full)） |
| run 目录 | `runs/air3_mode2_1609_unet3d_h2f_128b_r{1,2,3,4}`（AIR3；旧链为 `1609v2_*_128b_r{1,2,3}`） |
| config | `configs/air3_mode2_1609_unet3d_h2f_128b_r{1,2,3,4}.yaml`、`configs/1609v2_unet3d_h2f_128b_r{1,2,3}.yaml` |

AIReduceNoise2 旧链的 `full.dn_3d_128_r1.raw` 仅作为机制验证产物，不能复用于
AIReduceNoise3。Mode2_1609 单链完成新 r0 后，须用该 r0 对train+val的四域full生成
`full_by_r0`，再构造r1混合目标；不同数据集之间禁止交叉复用目标。

## 6. 门控

1. **r1 目标层**（`blend_dn_targets.py --r1-suffix full --r2-suffix dn_3d_128_r1
   --target-suffix dn_3d_128b_r1` 自动输出）：low 带 target ≈ full（≈不去噪）、
   high 带 target ≈ M0 输出、每带 target ∈ [full, M0 输出]。
2. **r1 输出层**：low 带 σ 介于 full 与 M0 输出之间（≈ full×0.6），high 带 < M0 输出。
3. **r2b/r3b**：全局 immerkaer delta>0 照旧；终点用 `scripts/eval_rounds_banded.py`
   （`--checkpoint M1b=... M2b=... M3b=... --band-key M2b`）看分带比值是否随密度
   单调下降、每轮 delta 逐带核对。
4. 铁律不变：`--init-weights` 热启动（r1b←r0、r2b←r1b、r3b←r2b）、test 永不进教师、
   控制变量只换 mappings+run_dir。

## 7. 执行流水线（服务器 192.168.36.102，AIReduceNoise3 最终版）

整条 r1b 链由 `scripts/air3_r1b_chain.sh` 驱动：**A 目标生成 → B 混合 → 门控 → C 训练**
串行；A/B 幂等（跳过已存在的输出文件），门控不过即停、不启动训练。2026-09-18 10:17 在
tmux 会话 `air3_r1b` 启动，日志 `/tmp/air3_r1b.log`：

```bash
tmux new-session -d -s air3_r1b \
  'cd /home/yyc/workspace/N2N && bash scripts/air3_r1b_chain.sh 2>&1 | tee /tmp/air3_r1b.log'
```

```bash
# A) M0(full) 教师目标（~2 h / 200 卷）：教师 = AIR3 r0 best.pt（epoch 21, val_l1 0.016562）
python scripts/prepare_full_dn_targets.py --config configs/air3_mode2_1609_unet3d_h2f_128_r0.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json \
  --checkpoint runs/air3_mode2_1609_unet3d_h2f_128_r0/best.pt --arch unet3d \
  --target-suffix dn_3d_128b_r0 --splits train,val \
  --tiled --patch-hw 256 --overlap-d 16 --overlap-hw 16 --batch-size 2 --device cuda:0

# B) r1 混合目标（~1-2 h，纯 numpy）
python scripts/blend_dn_targets.py --manifest <同上> \
  --r1-suffix full --r2-suffix dn_3d_128b_r0 --target-suffix dn_3d_128b_r1 \
  --lo 0.2 --hi 0.4 --kind smoothstep --splits train,val

# GATE：A 要求 immerkaer delta>0 且相对削减 ≥5%；B 要求 low 带 target==full、high 带
#       target==M0、mid 带 target∈[full,M0]、键必须是 M0 输出
python scripts/check_r1b_target_gates.py \
  --prep-overview  /data/raid0/yyc/AIReduceNoise3/prepare_full_dn_3d_128b_r0_overview.json \
  --blend-overview /data/raid0/yyc/AIReduceNoise3/blend_full_dn_3d_128b_r1_overview.json \
  --expect-studies 200

# C) r1b 训练（~10 h，热启动 r0 best.pt，patience 5）
python scripts/train_voxel.py --arch unet3d \
  --config configs/air3_mode2_1609_unet3d_h2f_128b_r1.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json \
  --run-dir runs/air3_mode2_1609_unet3d_h2f_128b_r1 \
  --init-weights runs/air3_mode2_1609_unet3d_h2f_128_r0/best.pt
```

r2b/r3b/r4b 与 A~C 同构但**省掉 B 步**（朴素递归，目标就是教师输出），并且全部由
`scripts/air3_r234b_chain.sh`（tmux `air3_r234b`，日志 `/tmp/air3_r234b.log`）自动串行：
先等 `/tmp/air3_r1b.log` 出现 `chain done` 且 `EXIT_CODE=0`，然后逐轮执行
「P) 教师目标 → G) 门控（仅目标层）→ T) 训练（热启动上一轮 best.pt）」。教师换成上一轮
best.pt、`--target-suffix dn_3d_128b_r2|r3|r4`、config 换 `air3_..._128b_r2|r3|r4.yaml`。

- **不要用 `scripts/run_recursive_round.py` 跑这条链**：它是早期工具，会重写 config
  （并强制 `stride = voxel_depth`，破坏 128 窗口 stride=112 的控制变量），
  且用 `dn{k}` 后缀，与 `dn_3d_128b_rN` 命名冲突。链脚本显式传 config，不改写。
- **门控模式**：r2b+ 没有混合层，用 `check_r1b_target_gates.py --prep-overview <overview>`
  （不带 `--blend-overview`）即进入"仅目标层"模式：覆盖率 200 + `delta>0` + 相对削减 ≥5%；
  相对削减 >70% 打印 `WARN`（疑似过平滑，需分带 QC），但不阻断。
- **overview 跨次合并**：`prepare_full_dn_targets.py` 现在把每次运行的诊断行按 study
  合并写回 `prepare_full_<suffix>_overview.json` 的 `rows`，断点续跑（只补缺失文件）时
  覆盖率仍是 200，不会因覆盖写导致假门控失败。
- **磁盘**：每轮新增 200 × 589.8 MB ≈ 118 GB；r2b~r4b 合计约 354 GB（启动前余 3.6 TiB）。

AIReduceNoise2 旧链
（`--manifest /data/raid0/yyc/AIReduceNoise2/splits_1609v2.json`、`1609v2_*` config、
`full.dn_3d_128_rN.raw`）的命令已作废，不再执行。

预计总时长：r1b 约 12 h（A ~2 h + B ~1-2 h + 训练 ~10 h，实测 r0 约 22 min/epoch）；
r2b/r3b/r4b 每轮约 11 h（目标生成实测 46 min + 门控分钟级 + 训练）。四轮合计约 45 h
（机器争用时 epoch 会从 22 min 拉长到 40-60 min，需按 2 倍余量估）。r4b 收敛后按 §8
用分带验收裁定交付档位。

## 8. 风险与可调项

- **骨区过平滑**（第 4 层压缩的历史收益只有 −10%）：若 r3b 骨带纹理 QC 变糊，
  停在 r2b 或把 `--hi` 从 0.4 降到 0.35（骨区更早切到深档）后仅重跑混合（分钟级）。
- **软组织终点可能略浅于方案 A**：B 的软组织在 r1/r2 训练时监督更噪（权重被噪
  目标拉过），终点若有差异以 `eval_rounds_banded.py` 实测为准。
- **键源**：默认键 = full_by_r0（方案 A 同款理由：去噪后强度做键不抖动）。想换
  `--mask-source low`（键=full，含噪，不推荐）。
- 代码支持：`blend_dn_targets.py` 已扩展 `--r1-suffix full`（低侧源直接用原始
  full，方案 B 专用路径），改动已通过数值等价检查的同一估计器输出门控。

## 9. 执行检查单（最终版，服务器 192.168.36.102）

> **执行状态（2026-09-18 10:17 起）**：步骤 1~2 由 `scripts/air3_r1b_chain.sh`
> 自动串行执行（tmux `air3_r1b`，日志 `/tmp/air3_r1b.log`）。步骤 4~9 由
> `scripts/air3_r234b_chain.sh` 自动串行（tmux `air3_r234b`，日志 `/tmp/air3_r234b.log`，
> 2026-09-18 19:14 启动，先等 r1b 收敛再逐轮 P→G→T，一路到 r4b）。预检通过：200 例 /
> train-val-test=180/20/0 / 13 台设备六卷齐全（1290 个 RAW 大小全部匹配）/ 磁盘余 3.6 TiB
> （r2b~r4b 三个新 suffix 约需 354 GiB）。门控脚本 `scripts/check_r1b_target_gates.py`
> 支持「目标层 + 混合层」与「仅目标层」两种模式（17 项单测覆盖）。

开始前使用 `/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced200_trainval.json`，并确认
200例、仅Mode2_1609、13台设备全部参与、六卷齐全、train/val/test=180/20/0；独立评估使用
`/data/raid0/yyc/AIReduceNoise3/splits_cross_mode_test11.json`。随后为该单链新建r0~r3 config；
现有 `1609v2_*` config 的数据路径仍指向 AIReduceNoise2，禁止直接启动；新建 AIReduceNoise3
专用 config 时，r0、r1b、r2b、r3b 均设置 `training.patience: 5`，即验证指标连续5个epoch
无改善便早停。所有步骤在 tmux 中执行，日志 `/tmp/<session>.log`；训练与目标生成错开 GPU。

| # | 步骤 | tmux 会话 | 依赖 | 预计 | 通过条件 |
| - | ---- | --------- | ---- | ---- | -------- |
| 1 | M0(full) 目标生成（AIR3 新增的 A 步） | `air3_r1b` | 无 | ~2 h | 全局 immerkaer full→M0 相对削减 ≥5%；manifest 注册 `full.dn_3d_128b_r0.raw` |
| 2 | r1 混合目标生成 | `air3_r1b` | 1 | ~1-2 h | 门控：low 带 target==full（≈不去噪）、high 带 target==M0 输出、mid 带 ∈ [full, M0]；键=M0 输出；manifest 注册 `full.dn_3d_128b_r1.raw` |
| 3 | r1b 训练 | `air3_r1b` | 2 | ~10 h | 热启动 r0 best.pt；`--init-weights` 校验通过；epoch1 val_l1 低于 r0 epoch1（0.01672） |
| 4 | r2b 目标生成 | `air3_r234b` | 3 | ~46 min | 门控 `check_r1b_target_gates.py --prep-overview`（仅目标层）：immerkaer delta>0（full → M1b(full)）、相对削减 ≥5%、覆盖 200；manifest 注册 `full.dn_3d_128b_r2.raw` |
| 5 | r2b 训练 | `air3_r234b` | 4 | ~10 h | 热启动 r1b best.pt；全局 delta>0 |
| 6 | r3b 目标生成 | `air3_r234b` | 5 | ~46 min | 同 4（教师 r2b best.pt，suffix `dn_3d_128b_r3`） |
| 7 | r3b 训练 | `air3_r234b` | 6 | ~10 h | 热启动 r2b best.pt |
| 8 | r4b 目标生成 | `air3_r234b` | 7 | ~46 min | 同 4（教师 r3b best.pt，suffix `dn_3d_128b_r4`） |
| 9 | r4b 训练 | `air3_r234b` | 8 | ~10 h | 热启动 r3b best.pt |
| 10 | 终点分带评估 | `eval_b` | 9 | ~30 s/例 ×11 | `eval_rounds_banded.py --checkpoint M1b/M2b/M3b/M4b --band-key M2b`：比值随密度单调下降（low>mid>high）、每带 σ < full；据此裁定交付档位（默认 r2b/r3b，过平滑则不上 r4b） |

- **自动续跑编排**：r1b 的 A→B→门控→训练内建在 `scripts/air3_r1b_chain.sh`（tmux
  `air3_r1b`）；r2b→r3b→r4b 由 `scripts/air3_r234b_chain.sh`（tmux `air3_r234b`）
  自动串行——先等 r1b 日志出现 `chain done` 且 `EXIT_CODE=0`，再逐轮
  「P 目标 → G 门控 → T 训练」，每轮训练成功写 `<run_dir>/.chain_done` 以便重跑跳过整轮，
  `FAIL_*` 立即停链。两个脚本都带单实例锁（PID 存活判断），防止被启动两次。
  进度用定时自动化每 2 h 只读检查两个日志（tmux 存活 / `[done]` 计数 / epoch 行 /
  `ROUND rNb DONE` / `ALL_ROUNDS_DONE`），不自动重启、不修改文件。
- **提前停靠点**：若终点评估显示 r3b/r4b 骨带过平滑（纹理 QC 变糊 / 分带比值不再
  下降甚至回升），可裁定停在 r2b（软 2/骨 3），或将 `--hi` 降至 0.35 仅重跑混合
  （分钟级）后从 r1b 起重训——r1b~r4b 的训练成本重付，故停靠优先于调参。
  链内建的部分保护：每轮的 teacher 目标都要过「相对削减 ≥5%」门控，收益递减到
  退化时链会自己停在下一轮之前。
- **test 隔离**：所有目标生成 `--splits train,val`；test 集只出现在步骤 10 及
  之后的交付推理，永不进教师。
- **总时长预算** ≈ 11 h/轮（目标生成实测 46 min + 训练；r0 实测 22 min/epoch，
  机器争用时按 40-60 min/epoch 估，四轮合计 45-90 h）。

## 10. 交付物清单（终点轮收敛后）

| 交付 | 位置 | 格式 |
| ---- | ---- | ---- |
| 分带验证报告 | `docs/reports/air3-128b-banded-validation.md` + `outputs/eval_air3_128b_banded_test11.json` | 由 `report_banded_rounds.py` 生成，评估对象是 `splits_cross_mode_test11.json` |
| test 集 11 例降噪体数据 | `outputs/eval_air3_128b_r*b_best/<group>__<study_uid>/{full.raw, full_denoised.raw}` | 360×640×640 headerless `<f4`，589,824,000 B/卷 |
| 分级模型库 | `runs/air3_mode2_1609_unet3d_h2f_128b_r{1,2,3,4}/best.pt` | 临床按剂量/部位选档：r1b 轻量档（软 1/骨 2）→ r4b 重档（软 4/骨 5）；实际交付档位由分带验收裁定 |

## 11. 决策记录

- 2026-09-17：AIReduceNoise3 r0 在 GPU 0 启动，使用旧 r0 `best.pt`（epoch 78）通过
  `--init-weights` 热启动，optimizer/scheduler冷启动；四域联合、patience=5。180例×6 memmap
  超过默认nofile=1024，首次启动在epoch前失败；确认hard limit=1048576后仅对tmux会话设置
  `ulimit -n 8192`，重启成功进入训练，显存约14.96GiB。
- 2026-09-17：取消YPDL设备名前缀限制，Mode2_1609下全部13台设备均可入选；200例按
  容量受限的均衡策略分配（8台×17、3台×16、容量不足两台分别15和1），seed=42。
  用户随后将200例改为纯train/val=180/20，并另设11例跨模式test（Mode0/1/2/3=2/3/3/3）；
  其中3个Mode2测试key与训练/验证互斥。最终链早停为连续5个epoch无改善即停止。
- 2026-09-17：Mode1 冒烟重建发现伪影，双源拆分后的投影不满足 XDay 单源偶数张
  图像的内部要求；暂时取消全部 1615 训练计划，manifest生成器增加不可绕过的硬排除。
- 2026-09-17：方案 B 定为最终方案。理由见 §3（与方案 A 终点深度等价、流程
  简化、中间模型即分级模型库、可提前停靠）；方案 A 链（旧命名
  `dn_3d_128_rN`）及其产物、报告、分带实测全部保留作对照，不再扩展。
- 2026-09-18：AIR3 r0 收敛（27 epoch 早停，best = epoch 21，train_l1 0.015335、
  val_l1 0.016562，22 min/epoch）。r1 目标无法复用 AIR2 的 `full.dn_3d_128_r1.raw`，
  故在 §7 中把 A 步显式化：先用 r0 best.pt 生成 `full.dn_3d_128b_r0.raw`，再混合出
  `full.dn_3d_128b_r1.raw`。混合目标只有 `full` 一份文件——四域由 Dataset 的
  `residuals:[bhc,marfi]` 在 source/target 上同时加 `variant−full` 实现，与 r0 完全一致。
- 2026-09-18 10:17：`scripts/air3_r1b_chain.sh` 在 tmux `air3_r1b` 启动（GPU0 空闲，
  另一项目 projection3d 的作业已结束）。运行时参数 `ulimit -n 8192` 写在脚本内。
  预检脚本 `scripts/_check_air3_r1_readiness.py` 确认 200 例六卷齐全、1290 个文件
  大小与 shape×4 一致、磁盘余 3.76 TiB（两个新 suffix 约需 220 GiB）。
  控制变量核验：r1b config 与 r0 仅 `data.mappings` 与 `training.run_dir` 不同。
  新增单测 12 项（门控脚本）+ 1 项（r1b config 与 r0 的对照），全量 186 passed。
- 2026-09-18 19:14：**用户决定链一路跑到 r4b**（"r1 收敛后接 r2，r2 收敛后接 r3，
  直到 r4 结束"）。据此新增 `configs/air3_mode2_1609_unet3d_h2f_128b_r{2,3,4}.yaml`
  （与 r0 仅 mappings/run_dir 不同，已用测试锁死）与 `scripts/air3_r234b_chain.sh`
  （tmux `air3_r234b`，日志 `/tmp/air3_r234b.log`，先等 r1b `EXIT_CODE=0` 再逐轮
  P→G→T；单实例锁 + 每轮 `.chain_done` 幂等标记）。门控脚本扩展为支持"仅目标层"模式
  （r2+ 无混合层）并在相对削减 >70% 时 WARN（疑似过平滑）；`prepare_full_dn_targets.py`
  的 overview 改为跨运行按 study 合并 `rows`，避免断点续跑造成假门控失败。
  预检在真实 r0 overview 上跑通（prep-only 门控 PASS，43.64% 削减），
  本地全量 196 passed。r4b 收敛后需做分带验收裁定交付档位，r4 本身不一定是终点。
- **结论未定**：r4b 是否优于 r3b 必须由 `eval_rounds_banded.py` 的分带 σ 实测决定；
  递归收益递减，历史数据显示骨区第 4 层压缩收益仅约 −10%，第 5 层可能更小甚至过平滑。
