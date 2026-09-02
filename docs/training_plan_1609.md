# 1609 视野去噪训练计划

> 数据源：全量重建产物 `/data/raid0/yyc/AIReduceNoise/PDL/`（Windows 侧 `Z:\raid0\yyc\AIReduceNoise\PDL\`；`reconstruction-index.json`，308 项全 success）
> 训练目标：Noise2Noise 自监督（even ↔ odd 独立噪声对），在 1609 视野（FOV 16cm）数据上训练 2D UNet 去噪模型
> 状态：计划稿 v1.3（2026-09-02，`create_recon_splits.py` 落地；修正 case 语义为设备号；E1 manifest 已生成：train 83 / val 10 / test 17）

---

## 0. 训练环境

| 项 | 值 |
|---|---|
| 训练服务器 | `192.168.36.102`（Linux，GPU 机；同机即 git 服务器 `http://192.168.36.102/imaging/N2N.git`） |
| GPU | **卡 0**（`CUDA_VISIBLE_DEVICES=0`） |
| 数据挂载点 | `/data/raid0/`（与 Windows `Z:/raid0/` 同源，重建产物在其下） |
| 数据根目录 | `/data/raid0/yyc/AIReduceNoise/`（`reconstruction-index.json` + `PDL/`） |
| 代码同步 | 经 git 拉取 `N2N` 仓库（与本地同源，已推送 origin/main） |
| Python 环境 | 服务器 conda 环境需含 `torch` + `numpy`（沿用项目依赖，环境名待确认） |

**执行模型**：代码在本地 `G:\workspace\N2N` 开发 → git 推送 → 服务器拉取；manifest/配置生成后在服务器执行训练命令。数据无需跨机拷贝（服务器直接读本地挂载）。

**路径映射**：Windows `Z:/raid0/yyc/AIReduceNoise/...` ≡ 服务器 `/data/raid0/yyc/AIReduceNoise/...`。训练命令一律使用服务器路径。

---

## 1. 目标

1. 用 **Mode2_1609 + Mode3_1609** 重建卷训练 CBCT 去噪 UNet，验证 Noise2Noise 自监督（half-view 互监督）在 1609 视野上的效果。
2. 与既有 `full_view`（even/odd → full，相关监督）基线对比，量化独立噪声对监督的增益。
3. 产出可用于推理/ONNX 部署的去噪模型（延续 base16 轻量化方向，7.76M → 1.94M 参数）。

## 2. 数据盘点

### 2.1 1609 组（本次训练主数据）

| 分组 | study 数 | case 数 | shape (z,y,x) | nviews | 卷类型（每 study） |
|---|---|---|---|---|---|
| Mode2_1609 | 104 | 4 | (360, 640, 640) | 640 | full / full.bhc / full.marfi / full.marfi.bhc / even / odd |
| Mode3_1609 | 6 | 1 | (360, 640, 640) | 1080 | 同上 |

- 全部 float32（`<f4`）、体素间距 0.25 mm、`split: true`（even/odd 各 320/540 views）。
- **术语澄清（2026-09-02 与数据方确认）**：`case_id` 是**设备号**（YPDL-26A-0009/-0020/-0026、YPDL-26B-0026 为 4 台设备），**不是患者**。每台设备下每条 study（一次扫描）对应**一位不同患者**。同设备多 study 无解剖重叠；数据集无患者级标识，故按 study 独立处理。Mode2_1609 各设备 study 数：A-0009 42、A-0020 46、A-0026 15、B-0026 1。

### 2.2 对照组（可选消融）

| 分组 | study | shape | 说明 |
|---|---|---|---|
| Mode2_0808 | 104 | (640, 640, 640) | 同模式、8cm FOV，FOV 泛化对比 |
| Mode3_0808 | 6 | (640, 640, 640) | 8cm + 1080 views |
| Mode1_1615 | 88 | (600, 640, 640) | 无 even/odd（未拆分），仅作 reference |

### 2.3 监督语义

- `full.raw`：全视角重建（目标，含源子集的贡献 → 相关监督）
- `even.raw` / `odd.raw`：奇偶子视角重建（互相独立噪声 → Noise2Noise 独立对）

## 3. 关键约束与决策

### 3.1 case 是设备号，无需 case 级划分

Mode2_1609 有 4 台设备、共 104 条 study，每条 study 是独立患者 → **不存在"同患者多次扫描跨 split 泄漏"**。8:1:1 直接按 study（患者）随机切分即可，seed 固定可复现；不需要（也不应该）按设备整块聚合——否则 val/test 会被锁死在单台设备上，指标受设备特性左右。

### 3.2 manifest key 需跨 group 唯一

`N2NDataset`（及其前身 `RawPairedSliceDataset`）以记录级 key 建索引（`records[_manifest_key]`，key 优先取 `item['key']`、缺省回退 `case_id`）、`splits[split]` 存 key 列表；**`case_id`（设备号）在同一设备跨组重建时必然重复**（如 A-0009 同时有 Mode1_1615/Mode2_0808/Mode2_1609 记录），按 case_id 索引会互相覆盖。实测 **study_uid 也不唯一**：308 条索引记录仅 198 个唯一 study_uid（110 条重复），重复源于**同一物理扫描重建到多个 FOV group**（同 study_uid 同时出现在 Mode2_1609 与 Mode2_0808）。E1 只含 1609 组时 110 条恰好全唯一，但混入 0808 等组时必然冲突。

**决策：key 用复合键 `f"{study_uid}::{group}"`（全库 308 条验证唯一），case_id 仅作辅助字段。** manifest 生成脚本见 §4，不复用 `create_splits.py`（其按 case 聚合且读 `pair.json`）。

### 3.3 划分单位：study（患者单次扫描），不做跨组聚合

- 同设备下不同 study = 不同患者 → 无解剖泄漏，可独立随机划分；
- 同一 study 重建到多个 FOV 组（1609/0808）的卷**图像内容不同**（FOV/视野裁剪差异），实际训练也不会把两版放在一起用，故不强制"同 study_uid 同 split"；
- 脚本按记录（study）独立随机分配 train/val/test；`--test-groups` 仅把整组（如 Mode3_1609）定向进 test，用于跨模式评估。

### 3.4 映射方向与残差增强（N2NDataset）

映射方向 = source → target 显式列表，可多项并存、同 epoch 混合：

| 映射 | 语义 | 用途 |
|---|---|---|
| `even->odd` / `odd->even` | 独立噪声对，Noise2Noise 核心 | **主实验**（两者同开 = 旧 `half_view_n2n`） |
| `even->full` / `odd->full` | 相关监督（基线，source 含 target 贡献） | 对照（两者同开 = 旧 `full_view`） |

- 配置键 `mappings: [even->odd, odd->even]`（`src/cbct_n2n/data/n2n_dataset.py`）；旧 `supervision_mode` 仅作兼容翻译，`mappings` 存在时优先。
- 可选残差增强 `residuals: [bhc, marfi]`：残差 `R = variant − full`（逐层，variant ∈ `full.bhc`/`full.marfi`/`full.marfi.bhc`），同时加到 source 与 target → **base 样本保留**、每启用 tag 追加 1 样本/层；`bhc`+`marfi` 同开自动追加 `marfibhc`。
- 缺卷降级：缺基础卷（even/odd/full）→ 跳过整个 study；缺变体卷 → 仅禁用该 study 上对应残差 tag，base 样本保留。

### 3.5 数据量控制

104 study × 360 层 × 2 source = 约 7.5 万样本/epoch（batch 2 时 3.7 万 iter）。**决策：默认 `max_slices_per_case: 120`（每 study 取前 120 层），兼顾训练速度与覆盖；如需更高精度可放开。**

## 4. 数据准备（manifest 生成）

`scripts/create_recon_splits.py`（已实现，2026-09-02；study 级随机划分 + `--test-groups` 定向）。**E1 manifest 已生成并验证**；如需复现（或重新切分），在服务器执行：

```
cd <N2N 仓库目录> && git pull
python scripts/create_recon_splits.py \
  --index /data/raid0/yyc/AIReduceNoise/reconstruction-index.json \
  --output-root /data/raid0/yyc/AIReduceNoise \
  --groups Mode2_1609,Mode3_1609 \
  --test-groups Mode3_1609 \
  --out /data/raid0/yyc/AIReduceNoise/splits_1609.json \
  --seed 42
```

- 输入：重建索引（308 记录，全 success）；按 `--groups` 过滤。
- 记录字段映射：**key = `f"{study_uid}::{group}"`（复合键，跨 group 唯一）**；`shape`/`dtype`/`outputs`（相对路径）直取；补充 `case_id`、`study_uid`、`group`、`mode`、`kv`、`ma`、`nviews`、`scan_id`、`status`。
- **`--out` 必须与 `--output-root` 同目录**（脚本硬校验）：数据集类以 manifest 文件所在目录为 base 解析相对路径，manifest 须与 `reconstruction-index.json` 同层。
- 划分：`--test-groups Mode3_1609` → 该组 6 条**全部进 test**（跨模式）；其余 Mode2_1609 104 条按 `--ratios 0.8/0.1/0.1` 随机切（seed 42，study 级独立，见 §3.3）。
- **E1 实际落位（seed 42）**：train 83（Mode2_1609）、val 10（Mode2_1609）、test 17（Mode2_1609×11 + Mode3_1609×6）。缺基础卷（even/odd/full）的记录会保留但计入告警，N2NDataset 侧自行跳过。
- 命令行可调：`--ratios`、`--seed`、`--groups`、`--test-groups`（`--out` 与 `--output-root` 同目录限制不变）。

## 5. 训练方案

### 5.1 入口

`N2NDataset` 直接 memmap 消费 `.raw` 卷（相对路径 outputs，兼容 `.raw` 与裸键两种记录风格），**无需重建成对 .npy**；方向映射与 MARFI/BHC 残差增强全由 config `data` 段开关。训练入口 `train_sparse_dense.py`（已接 N2NDataset），**在服务器上执行、显式指定卡 0**：

```
cd <N2N 仓库目录> && git pull
CUDA_VISIBLE_DEVICES=0 python scripts/train_sparse_dense.py \
  --config configs/1609.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise/splits_1609.json \
  --run-dir runs/1609_n2n
```

（`device: cuda:0` 已在 config 中声明，`CUDA_VISIBLE_DEVICES=0` 双保险。）

### 5.2 配置 `configs/1609.yaml`（见配套文件）

- 模型：`base_channels: 16, depth: 3, residual: true`（轻量主线，延续 base16 方向）
- 映射：`mappings: [even->odd, odd->even]`（= 旧 `half_view_n2n`；`residuals:` 默认关闭，E6 时打开）
- 训练：`epochs: 100, batch_size: 2, lr: 1e-4, amp: true, patience: 15, device: cuda:0`
- 数据：`manifest: /data/raid0/yyc/AIReduceNoise/splits_1609.json`、`metric_data_range: 4.0`、`max_slices_per_case: 120`

## 6. 实验矩阵

| # | 实验 | 数据 | 监督模式 | 模型 | 目的 |
|---|---|---|---|---|---|
| E1 | 1609 N2N（主） | Mode2_1609 | half_view_n2n | base16 | Noise2Noise 主结果 |
| E2 | 1609 full 基线 | Mode2_1609 | full_view | base16 | 对照：独立对 vs 相关监督 |
| E3 | 1609 base32 | Mode2_1609 | half_view_n2n | base32 | 容量消融（7.76M vs 1.94M） |
| E4 | 0808 N2N | Mode2_0808 | half_view_n2n | base16 | FOV 泛化对比（同模式 8cm） |
| E5 | 跨模式测试 | 训练 E1，测试 Mode3_1609 | — | — | 1080-view 跨模式泛化 |
| E6 | 残差增强消融（可选） | Mode2_1609 | even↔odd + `residuals: [bhc, marfi]` | base16 | 后端伪影链对去噪的影响 |

- 主结论口径：E1 vs E2（监督模式）、E1 vs E3（容量）、E1→E5（跨模式）。
- E4 需同法生成 `splits_0808.json`；E6 通过 N2NDataset `residuals: [bhc, marfi]` 直接启用（依赖 1609 组已有的 `full.bhc`/`full.marfi`/`full.marfi.bhc` 卷）。

## 7. 评估方案

- 指标：PSNR / SSIM / MAE / RMSE（`metric_data_range=4.0` 仅归一化 PSNR/SSIM），与既有 `compute_metrics` 一致。
- **注意**：现 `scripts/test.py` 读 `full_path/even_path/odd_path`（.npy 风格），**不兼容 raw 记录**。评估需扩展：新增 raw 版推理（memmap 读 `outputs`，`infer_volume` 复用）或改造 test.py 支持两种 manifest。计划为待办项。
- 测试集构成（seed 42 实际）：Mode2_1609 随机留出 11 条（同模式）+ Mode3_1609 全部 6 条（跨模式），报告分开与合并两组指标。
- 可选：中心层切片可视化 + 与 full（相关参考）的残差图，交训练曲线 `training_curves.png` 一并归档。

## 8. 执行步骤

```mermaid
graph LR
  A[新建 create_recon_splits.py] --> B[生成 splits_1609.json]
  B --> C[新增 configs/1609.yaml]
  C --> D[E1 训练 runs/1609_n2n]
  D --> E[raw 版评估/改造 test.py]
  E --> F[E1 测试集指标]
  F --> G[跑 E2-E4 矩阵]
  G --> H[汇总对比 + 选型]
  H --> I[ONNX 导出/量化 1609 模型]
```

1. **数据准备**：`create_recon_splits.py` 已实现（含单测），本地已对真实索引生成 `Z:\raid0\yyc\AIReduceNoise\splits_1609.json`（train 83 / val 10 / test 17），并用 N2NDataset 端到端验证读真实 `.raw` 卷 + 残差增强数值一致。服务器侧 `git pull` 后即可用 §4 命令复现。
2. **配置**：落地 `configs/1609.yaml`（`device: cuda:0`、服务器 manifest 路径）。
3. **smoke 验证**：用 `smoke` 思路（epochs=2、max_slices 16）在服务器跑通 E1 全链路（manifest → 训练 → 评估，卡 0）。
4. **正式训练 E1**（服务器后台 `CUDA_VISIBLE_DEVICES=0`，预计 6–12h，见 §9）。
5. **评估**：扩展 raw 版 test，输出测试集 PSNR/SSIM 等。
6. **实验矩阵**：E2–E4 顺序训练（E3 可复用 E1 权重 warm-start）。
7. **选型与部署**：对比表选最优，`export_onnx.py` 导出（fp16 约 3.9MB）。

## 9. 里程碑与时间预估

| 阶段 | 内容 | 预估 |
|---|---|---|
| M1 | manifest + 配置 + smoke | 0.5–1 天 |
| M2 | E1 正式训练 | 6–12h（GPU 后台） |
| M3 | E2–E4 矩阵 | 1–2 天（并行/顺次） |
| M4 | 评估、选型、ONNX | 0.5 天 |

粗估依据：640×640 2D 切片、base16、batch2、AMP；max_slices=120 时约 1.4 万样本/epoch、100 epoch。实际按 GPU 占用调整 `max_slices_per_case` 与 `epochs`。

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| case_id=设备号、study=患者，无患者级标识 | 若同一患者确有多次扫描无法识别 | 按 study 独立随机划分（§3.3）；报告注明按扫描而非患者分层 |
| 同 study 跨 FOV 组重复记录（1609/0808） | 记录重复计数、双份进 train | key=复合键唯一标识记录（§3.2）；两版图像不同、不并训，不强制聚合 |
| 仅 4 台设备、样本集中于少数设备 | 泛化到新设备存疑 | 随机划分保证各 split 混有全部设备；后续并入 0808/1615 扩集 |
| test.py 不兼容 raw manifest | 评估阻塞 | §7 待办：raw 版评估（优先于矩阵训练） |
| 360 层×104 study 训练量大 | 单 epoch 过慢 | max_slices_per_case 控制；amp；必要时 depth=3 保持 |
| Mode3 仅 6 条 study | 跨模式测试样本少 | 与 Mode2 留出集合并解读；标注置信度 |

---

## 待办清单（按序）

- [x] `RawPairedSliceDataset` 索引从 case_id 改为 manifest key（兼容回退）
- [x] 实现 `N2NDataset`（`src/cbct_n2n/data/n2n_dataset.py`）：`mappings` 显式方向（多项并存）、legacy `supervision_mode` 翻译、`residuals` 残差增强（bhc/marfi/自动 marfibhc）、缺卷降级语义
- [x] `N2NDataset` 回归测试 `tests/data/test_n2n_dataset.py`（23 用例）+ 训练入口 `train_sparse_dense.py` 接线 + `configs/1609.yaml` 迁移至 `mappings` schema
- [x] 新建 `scripts/create_recon_splits.py`（study 级随机划分 + 复合 key `study_uid::group` + `--test-groups`；单测 `tests/data/test_create_recon_splits.py`）并生成验证 E1 manifest（train 83/val 10/test 17）
- [ ] raw 版评估：改造 `scripts/test.py` 或新增 `scripts/test_raw.py`
- [ ] smoke 跑通 E1 链路（含 N2NDataset + DataLoader 元数据 collate）
- [ ] E1 正式训练 → E2/E3/E4 矩阵 → E5 跨模式测试 → E6 残差增强消融（可选）
- [ ] 汇总指标对比、选型、ONNX 导出
