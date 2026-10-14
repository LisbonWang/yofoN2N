# CBCT 图像域去噪

本工程包含 CBCT 奇偶投影重建、2D/3D UNet 图像域去噪、Noise2Noise 与 half-to-full 训练，以及 RAW 体数据推理和 TensorRT 导出工具。训练数据、checkpoint 与输出目录由配置或命令行指定；运行前确认目标存储路径和 manifest 与当前任务匹配。

## 环境

Windows 工作站 RTX 5060 Ti 的 CUDA 环境使用独立 Conda 环境 `torch`：

```powershell
conda activate torch
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

该工作站曾验证 `torch 2.12.1+cu130` 可在 RTX 5060 Ti 上运行。`environment.yml` 定义 Python 和通用依赖，不安装 PyTorch CUDA wheel；按本机驱动选择 PyTorch 构建。CUDA 12.6 的 PyTorch 构建不支持 RTX 5060 Ti 的 `sm_120`。

创建环境并安装已验证的 CUDA 构建：

```powershell
conda env create -f environment.yml
conda activate torch
python -m pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -e PATH_TO_XDAY_KITS
pip install -e . --no-deps
$env:XDAY_DIR = "D:\XDay Release\XDay617.2"
```

## 传统配对数据流程

以下 `prepare_pairs.py` / `train.py` 命令属于较早的二维配对数据流程。它需要本机存在 XDay SDK 和有效的项目配置；当前仓库不包含 `configs/default.yaml`。

默认数据源是 `Z:\raid5\共享数据\典型图像\长安管口-PDL`。先只扫描并检查目录：

```powershell
python scripts\prepare_pairs.py `
  "Z:\raid5\共享数据\典型图像\长安管口-PDL" `
  outputs\pairs --mode Mode2 --dry-run
```

执行奇偶拆分和 XDay 重建：

```powershell
python scripts\prepare_pairs.py `
  "Z:\raid5\共享数据\典型图像\长安管口-PDL" `
  outputs\pairs --mode Mode2 --xday-dir $env:XDAY_DIR --link-mode auto
```

扫描到的病例数和模式分布取决于当前源目录；先查看 dry-run 输出。上述真实重建命令只处理 Mode2。每个入选病例生成 `even.npy`、`odd.npy`、`pair.json`。投影文件按自然数字顺序拆分，配置副本的 `fdkPara.xml/noView` 和 `projectionMatrix.bin` 同步切片；源盘不会被修改。

若某个配置二进制文件已确认是逐视角布局，需要显式提供每个视角的字节数，例如 `--per-view-file fitData.bin=32`。不要仅凭文件名切分 `weightR.bin`、`gx.errors` 等文件：当前扫描样本中的 `weightR.bin` 长度不能被 640 个视角整除，默认会作为共享校准数据原样复制。显式声明的文件若长度与投影视角数不匹配，准备过程会立即报错。

## 划分、训练、测试

```powershell
python scripts\create_splits.py outputs\pairs outputs\splits.json
python scripts\train.py --config YOUR_2D_CONFIG.yaml --manifest outputs\splits.json --run-dir runs\default
python scripts\test.py --config YOUR_2D_CONFIG.yaml `
  --manifest outputs\splits.json --checkpoint runs\default\best.pt `
  --output-dir outputs\test --split test --input-kind both
```

The config must match the selected training entry point. This checkout does not include the historical `configs/default.yaml`, `configs/smoke.yaml`, or `configs/unet3d_smoke.yaml` defaults; pass an existing config explicitly.

### Full-target supervision

Each Mode2 case produces `even.npy`, `odd.npy`, `full.npy`, and `pair.json`.
The full reconstruction uses all original views. Training uses raw-value L1 for
the correlated pairs `even -> full` and `odd -> full`; this is not independent
Noise2Noise supervision because the target contains the input subset views.

The test command writes `even_denoised.npy` and `odd_denoised.npy` with matching
metrics files. The reference label is `correlated full-view reconstruction`.

The older 2D path trains full axial slices. The active AIR3 volumetric runs use `train_voxel.py` instead.

### AIR3 3D 训练

#### 102 服务器与数据

训练主机为 `yyc@192.168.36.102`。代码和运行目录位于 `/home/yyc/workspace/N2N`，AIR3 体数据和 manifest 位于 RAID 卷：

| 内容 | 服务器路径 |
| --- | --- |
| AIR3 数据根目录 | `/data/raid0/yyc/AIReduceNoise3` |
| 重建体数据 | `/data/raid0/yyc/AIReduceNoise3/PDL/Mode2_1609/<case>/<study_uid>/` |
| 400 份训练 manifest | `/data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced400_trainval.json` |
| 训练运行、checkpoint、history | `/home/yyc/workspace/N2N/runs/` |

核验于 2026-10-14 的 400 份 manifest 中，`records` 有 400 条，`splits` 为 360 train、40 val、0 test；但顶层 `target_total` 仍为 200，与记录数不一致。重建 manifest 后重新核对 `records`、`splits` 和 `target_total`，并检查每条记录的 `outputs` 路径确实存在。manifest 中的输出路径相对 `/data/raid0/yyc/AIReduceNoise3` 解析。原始体数据是 headerless float32 RAW，shape/dtype 和每个体数据的路径记录在 manifest。

使用 102 上的训练环境：

```bash
ssh yyc@192.168.36.102
cd /home/yyc/workspace/N2N
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch251
git status --short
ls configs/air3_mode2_1609_unet3d_n2n_s64_400.yaml configs/air3_mode2_1609_unet3d_r0_s64_400.yaml
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
```

服务器环境当前报告 PyTorch 2.5.1、CUDA runtime 12.1、CUDA 可用，并有 4 张 RTX 4090。Windows 工作区不会自动同步到服务器；运行前确认服务器 checkout 中包含所需配置和代码。核验时 Python 曾显示 `_distutils_hack` 的 `.pth` 告警，但 PyTorch 导入和 CUDA 可用性检查成功；若告警仍出现，单独检查环境，不要把它直接当成训练失败。训练 GPU 编号按 `nvidia-smi` 的当前空闲情况设置。

长时间训练建议在 tmux 中启动，SSH 断开后进程仍会继续：

```bash
tmux new -s air3-train
# 在 tmux 会话中执行下面的训练命令
# Ctrl-b 后按 d 可 detach；重新登录后用：
tmux attach -t air3-train
```

#### 训练入口与数据流

AIR3 体数据训练通过 `scripts/train_voxel.py` 进入：它读取 YAML 配置和 study-level manifest，构造 `N2N3DDataset` 的 train/val 划分，经 `GroupedShuffleSampler` 组织训练样本，再调用 `build_model` 创建 2D/3D UNet，并用共享的 `training.trainer.fit` 训练、验证和保存 checkpoint。`data.mappings` 决定监督方向，`data.residuals` 可加入 BHC/MARFI 残差增强；3D 输入为 `[B,C,D,H,W]` 窗口，参数由 `voxel_depth`、`stride` 和 `patch_size` 控制。验证不做随机增强，模型选择和早停依据验证 loss。

AIR3 方案定义为：N2N 用 `even->odd`、`odd->even`；R0 用 `even->full`、`odd->full`；fused 用 `full->full.by_r0_by_n2n`，目标依次由 R0 和 N2N teacher 生成。注意，当前 400 份、stride-64 配置只覆盖 N2N 和 R0；`configs/air3_mode2_1609_unet3d_fused.yaml` 与 `scripts/air3_n2n_r0_fused_chain.sh` 仍是 200 份、stride-112 的旧流程。不要直接用旧链脚本跑 400 份训练。若扩展 fused 到 400 份，应先基于 400 份 manifest 生成 teacher RAW、登记输出，再创建 fused 的 400 份配置和独立 run-dir。具体目标生成命令见 [`docs/superpowers/plans/2026-10-08-n2n-r0-distillation.md`](docs/superpowers/plans/2026-10-08-n2n-r0-distillation.md)。

#### 从服务器启动 stride-64 继续训练

下面两项从原 stride-64 checkpoint 初始化模型权重，在 400 份数据上继续训练。`--init-weights` 只加载模型参数，优化器和调度器从头初始化；`patience: 5` 表示验证 loss 连续 5 轮未改善时早停。配置、manifest、初始 checkpoint 和新 run 目录需在服务器上存在：

```bash
cd /home/yyc/workspace/N2N
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch251
ulimit -n 65536
RUN_ID=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/train_voxel.py \
  --config configs/air3_mode2_1609_unet3d_n2n_s64_400.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced400_trainval.json \
  --run-dir "runs/air3_mode2_1609_unet3d_n2n_s64_400_${RUN_ID}" \
  --init-weights runs/air3_mode2_1609_unet3d_n2n_s64_ft/best.pt

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/train_voxel.py \
  --config configs/air3_mode2_1609_unet3d_r0_s64_400.yaml \
  --manifest /data/raid0/yyc/AIReduceNoise3/splits_mode2_1609_balanced400_trainval.json \
  --run-dir "runs/air3_mode2_1609_unet3d_r0_s64_400_${RUN_ID}" \
  --init-weights runs/air3_mode2_1609_unet3d_r0_s64_ft/best.pt
```

GPU 编号以 `nvidia-smi` 为准；上例假定物理 GPU 0、1 都空闲，且两个阶段依次执行。若只有一张 GPU，按空闲卡调整 `CUDA_VISIBLE_DEVICES` 并逐个运行。训练会在 `--run-dir` 下写入 `best.pt`、`last.pt`、`history.jsonl` 和训练图；每次新训练使用唯一 run-dir，避免覆盖旧结果。中断后用 `--resume runs/<run-dir>/last.pt` 恢复模型、优化器和调度器状态；`--init-weights` 则只导入模型权重，作为新阶段冷启动优化器。初始 200 份配置 `air3_mode2_1609_unet3d_{n2n,r0}.yaml` 对应另一份 manifest，不要与 400 份清单混用。

#### 设计新的训练任务

1. 明确输入、目标和数据划分。例如独立半视图监督用 `even->odd` 与 `odd->even`；half-to-full 用 `even->full` 与 `odd->full`。避免把包含输入视图的 full 重建描述为独立干净真值。
2. 准备并检查 manifest：记录稳定的 study key、`shape`、`dtype`、相对或绝对输出路径，以及 train/val/test split。先验证所需 RAW 文件存在、字节数与 shape/dtype 一致，并确认患者/病例划分没有泄漏。
3. 复制最接近的 YAML 配置，修改 `data.mappings`、`residuals`、`voxel_depth`、`stride`、`patch_size`、模型结构和训练超参数；把 `training.run_dir` 指向全新的目录。窗口深度应满足 UNet 下采样层数要求，显存不足时先减小 patch 或 batch。
4. 在 manifest 有独立验证集时运行 `scripts/train_voxel.py --config ... --manifest ... --run-dir ...`。从头训练不传初始化参数；接续一个新阶段用 `--init-weights`；从中断点完整恢复用 `--resume`，两者不能同时传。
5. 检查 `history.jsonl` 中的 train/val loss、早停轮次和最佳 checkpoint，再用保留的 test split 或独立扫描评估。需要 teacher/student 链时，先定义并产出 teacher 目标、登记回 manifest，再训练下游配置；记录 teacher checkpoint 和目标生成参数，确保结果可复现。

目录职责和 AIR3 三阶段方案另见下文“工程结构”和计划文档。

### 单个 RAW 体数据推理

`denoise_single_raw.py` 读取 headerless little-endian float32 RAW。必须给出 `(Z,Y,X)` 尺寸和匹配的 3D checkpoint；切块推理可降低显存占用，并用重叠 ramp 融合块边界：

```bash
python scripts/denoise_single_raw.py \
  --input input.raw --output outputs/input_denoised.raw \
  --shape 240 400 400 --arch unet3d --checkpoint runs/RUN_NAME/best.pt \
  --base-channels 16 --depth 4 --voxel-depth 128 --batch-size 1 \
  --device cuda:0 --tiled --patch-hw 256 \
  --overlap-d 16 --overlap-hw 16
```

批量测试清单推理使用 `scripts/denoise_test_set.py`；原始输入和降噪输出分别写入 `inputs/`、`denoised/` 子目录。TensorRT 工具包括 `export_onnx.py`、`quantize_onnx.py` 和 `build_trt_engine.py`，需在安装了兼容 TensorRT 的环境中运行。

### Pair Reconstruction Storage

Use the configuration-driven Mode2 reconstruction command to write only
`even.npy`, `odd.npy`, `full.npy`, and `pair.json` under
`Z:\raid5\共享数据\典型图像\长安管口-PDL-N2N-Pairs`:

```powershell
python scripts\reconstruct_mode2_pairs.py --config YOUR_RECONSTRUCTION_CONFIG.yaml --dry-run
python scripts\reconstruct_mode2_pairs.py --config YOUR_RECONSTRUCTION_CONFIG.yaml
```

The example config must define `reconstruction.source_root`, `output_root`, and `xday_dir`; the historical `configs/default.yaml` is not present in this checkout.

### 批量 PDL 重建

`scripts\reconstruct_pdl_variants.py` 按 `configs\recon_groups.json` 中匹配的重建组处理 PDL study。每个支持奇偶拆分的 Mode2/Mode3 重建组写 6 个 float32 RAW：`even.raw`、`odd.raw`、`full.raw`、`full.bhc.raw`、`full.marfi.raw`、`full.marfi.bhc.raw`。一个 Mode2 study 默认匹配 `Mode2_1609` 和 `Mode2_0808` 两组，合计最多 12 个输出；Mode1 默认不做奇偶拆分。输出写到默认根目录 `Z:\raid0\yyc\AIReduceNoise\PDL\<group>\<病例>\<study_uid>\`，另写 `metadata.json` 和根目录 `reconstruction-index.json`。完整输出会在续跑时跳过。

输入布局为 `\\192.168.36.235\团队文件-牙科ct\data_raw\PDL\<病例>\<study_uid>\<序列>_ModeX_*kV*mA*/`，
配置位于同 study 下 `config/recon/<dev>/<ModeX>/`（含 `filter.xml`、`fdkPara.xml`）。
没有配置或无法解析 kV/mA 的 study 会被跳过并记录原因。study 数会随输入和分组配置变化；以运行生成的 `reconstruction-index.json` 和 `skipped` 列表为准，不在 README 固定记录数据集计数。

重建前 raw 与 config 先拷贝到临时工作区（默认系统临时目录，`--workspace-root` 可改）；
偶/奇子集由 `build_subset_workspace` 切出（同步更新 `fdkPara.xml/noView` 并切片逐视角二进制）。
逐视角布局按模式区分：Mode2/Mode3 的 `projectionMatrix.bin` 为 84 字节/视角（640×84、1080×84），
Mode1 的 `projectionMatrix.bin`（85680 字节）与视角数不匹配，按共享校准数据整份复制；
`weightR.bin`、`defective.bin`、`offset.bin` 等始终整份复制。声明为逐视角但尺寸校验不匹配的文件
会自动降级为整份复制并在 metadata 中记录 `per_view_warnings`，不会中断批处理。
降采样子集按规则修正 MARFI/BHC 的 `step`：若 `step > 1` 且为奇数则向上取偶，full 不变；
`--filter-set MODULE.FIELD=VALUE` 可对每个临时配置额外覆盖任意 filter.xml 参数。

```powershell
# dry-run：只扫描并打印计划（不重建、不写输出目录）
python scripts\reconstruct_pdl_variants.py --dry-run

# 真实批量重建（输出量取决于扫描结果和匹配的重建组；先跑 --limit 1）
python scripts\reconstruct_pdl_variants.py --limit 1 --keep-workspace
python scripts\reconstruct_pdl_variants.py

# 只跑 Mode2，并覆盖 MARFI 参数
python scripts\reconstruct_pdl_variants.py --mode Mode2 --filter-set MARFI.step=2
```

Training manifests, checkpoints, history, metrics, and denoised outputs should
be written to the configured `outputs/` and `runs/` directories. Large datasets
and run artifacts may live on a mounted data volume or a remote training host.

### Legacy 2D Linux training

This path requires a compatible user-provided 2D config. With the pair share mounted at
`/data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs`, create a local manifest and
start training directly with Python:

```bash
python3 scripts/create_splits.py \
  /data/raid5/共享数据/典型图像/长安管口-PDL-N2N-Pairs \
  outputs/splits-linux.json \
  --source-root /data/raid5/共享数据/典型图像/长安管口-PDL

python3 scripts/train.py \
  --config YOUR_2D_CONFIG.yaml \
  --manifest outputs/splits-linux.json \
  --run-dir runs/linux-full
```

The pair mount is read-only input. The manifest, checkpoints, and all training
artifacts are written locally. Resume a stopped run with
`--resume runs/linux-full/last.pt`.

旧 2D full-target 配置的训练输入是完整轴向切片，右侧和底部补齐到 32 的倍数；每个切片提供 `even -> full` 和 `odd -> full` 两个样本。全视图包含输入子集，因此是相关监督目标，不是独立 Noise2Noise 目标或干净真值。

## 工程结构

| 路径 | 职责 |
| --- | --- |
| `scripts/` | 命令行入口：数据准备、manifest 划分、2D/3D 训练、RAW 推理、teacher 目标生成、评估和 TensorRT 导出。 |
| `configs/` | YAML 任务配置，以及重建组/filter 配置。训练配置组合数据映射、窗口参数、模型和优化参数。 |
| `src/cbct_n2n/reconstruction/` | 扫描发现、奇偶投影拆分、XDay 重建编排和输出索引。 |
| `src/cbct_n2n/data/` | 配对切片和 study-level RAW Dataset、manifest 读取、3D 窗口采样及按 study 分组的 sampler。 |
| `src/cbct_n2n/models/` | 2D UNet、depthwise-separable 2D UNet 和 3D UNet。 |
| `src/cbct_n2n/training/` | 训练循环、loss/指标、scheduler、checkpoint、绘图、体数据推理和融合工具。 |
| `runs/` | 被忽略的运行产物位置约定；服务器上的实际 checkpoint、history 和日志位于 `/home/yyc/workspace/N2N/runs/`。 |

2D 配对训练入口是 `scripts/train.py`；AIR3 体数据训练入口是 `scripts/train_voxel.py`。训练脚本复用 Dataset、模型构造和 `training.trainer.fit`，任务差异主要通过 YAML 配置中的映射、数据窗口和模型/训练参数表达。新增任务通常新增配置和独立 run-dir；只有数据格式、监督语义或模型结构超出现有 Dataset/模型接口时，才需要扩展对应模块。

## 验证

```powershell
$py = "C:\Users\admin\anaconda3\envs\torch\python.exe"
& $py -m pytest -q
& $py -m compileall src scripts
& $py scripts\prepare_pairs.py --help
& $py scripts\create_splits.py --help
& $py scripts\train.py --help
& $py scripts\test.py --help
```

没有设置有效 XDay SDK 时，重建只能执行 dry-run 或 mock 测试；不会把 mock 结果当作真实扫描重建结果。AIR3 训练命令还要求训练服务器可访问配置指定的 RAW 文件和 manifest。
