# CBCT 图像域去噪

本工程包含 CBCT 奇偶投影重建、2D/3D UNet 图像域去噪、Noise2Noise 与 half-to-full 训练，以及 RAW 体数据推理和 TensorRT 导出工具。训练数据、checkpoint 与输出目录由配置或命令行指定；运行前确认目标存储路径和 manifest 与当前任务匹配。

## 环境与安装

要求 Python 3.10 或更新版本。CPU-only 和 CUDA PyTorch 均可使用；安装与本机 PyTorch 官方支持的命令匹配的 PyTorch 构建。基础安装不需要 XDay SDK：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

重建原始投影数据需要可用的 XDay SDK 和 `xday-kits`；只训练、评估或推理 RAW 体数据不需要它。按平台安装 XDay SDK，再安装项目可选依赖 `python -m pip install -e '.[reconstruction]'`，并通过 `--xday-dir` 或 `XDAY_DIR` 指定 SDK 位置。

## 数据与 manifest

训练数据不随代码仓库分发。准备 headerless little-endian float32 RAW 体数据和 JSON manifest；数据布局、字段、相对路径规则、划分隔离与校验方法见 [数据格式说明](docs/data-format.md)。运行时由用户提供 manifest；manifest 中 RAW 相对路径以 manifest 所在目录为基准。

```bash
python scripts/train_voxel.py \
  --config configs/examples/unet3d_n2n.yaml \
  --manifest /path/to/your/manifest.json \
  --run-dir /path/to/your/runs/n2n-example
```

`configs/examples/unet3d_n2n.yaml` 是可移植模板，不含机器、账号或数据集路径。配置的 `data.mappings` 决定 source/target 监督关系，例如互补半视图 N2N 可用 `even->odd` 和 `odd->even`；half-to-full 可用 `even->full` 和 `odd->full`。数据应使用独立 train/validation/test 划分，full 重建包含输入视图时不能当作独立干净真值。

## 使用重建数据

`scripts/prepare_pairs.py` 和 `scripts/reconstruct_pdl_variants.py` 可将投影数据重建为训练体数据。传入自己的源目录、输出目录和 XDay SDK 路径；先 dry-run 检查扫描结果，再执行：

```bash
python scripts/prepare_pairs.py /path/to/raw-scans /path/to/pairs \
  --mode Mode2 --dry-run
python scripts/prepare_pairs.py /path/to/raw-scans /path/to/pairs \
  --mode Mode2 --xday-dir "$XDAY_DIR" --link-mode auto
```

重建布局、config/filter 输入、奇偶投影切分与输出索引见各脚本 `--help` 和其配置文件。源投影和配置按只读方式处理，产物写入用户指定的输出根目录。XDay 和扫描格式因站点而异，不把某个内部共享盘或病例数量作为默认值。

## 本地或远程训练

训练命令在任何能访问代码、manifest、RAW 文件和 checkpoint 的主机上运行。SSH、tmux 或作业调度器只负责会话与资源管理；命令行接口相同。长任务可在 tmux/scheduler allocation 内启动，`--run-dir` 下保存 `best.pt`、`last.pt`、`history.jsonl` 和图表。中断续训用 `--resume <run-dir>/last.pt` 恢复训练状态；从已有权重开始新阶段用 `--init-weights <checkpoint>`，仅加载模型参数。

通用代码架构与新训练设计步骤见 [工程结构](#工程结构) 和 [设计新训练](#设计新训练)。AIReduceNoise 专用配置、服务器路径和训练结果是实验快照，清单见 [`configs/README.md`](configs/README.md)；它们需要对应数据集和路径，不作为通用默认配置。

### 本机与远程服务器

本机开发和远程训练使用相同的 Python 入口。将代码、配置和 manifest 放在计算节点可访问的位置；RAW 文件可以留在该节点可读的数据存储中，不要求复制进仓库。下面使用占位变量，按本机目录和集群规则替换：

```bash
export PROJECT_ROOT=/path/to/N2N
export DATA_ROOT=/path/to/dataset
export MANIFEST="$DATA_ROOT/manifest.json"
export RUN_ROOT=/path/to/runs
cd "$PROJECT_ROOT"
python scripts/train_voxel.py --config configs/examples/unet3d_n2n.yaml \
  --manifest "$MANIFEST" --run-dir "$RUN_ROOT/n2n-example"
```

Linux 服务器可通过 SSH 登录；长任务可放在 tmux 或集群调度器分配的作业中。先按所在机构的认证、队列和 GPU 资源规则申请节点，再在远程 shell 中设置该节点上的目录并运行同一条训练命令。`CUDA_VISIBLE_DEVICES` 由用户或调度器设置。不要把密码、私钥、个人账号、服务器地址或数据挂载路径写入仓库。

```bash
export SSH_USER=your-account
export SSH_HOST=your-compute-host
ssh "${SSH_USER}@${SSH_HOST}"
export PROJECT_ROOT=/path/to/N2N
export DATA_ROOT=/path/to/dataset
export MANIFEST="$DATA_ROOT/manifest.json"
export RUN_ROOT=/path/to/runs
cd "$PROJECT_ROOT"
tmux new -s n2n-train
# 在 tmux 中运行上面的训练命令；退出终端后可重新连接该会话。
```

### 设计新训练

一般不需要复制或修改训练器：

1. 准备满足 [数据格式说明](docs/data-format.md) 的训练数据和 manifest，按病例/独立采集对象划分 train、val、test。
2. 复制 [`configs/examples/unet3d_n2n.yaml`](configs/examples/unet3d_n2n.yaml)，调整 `data.mappings`、模型容量、3D 窗口、优化器、早停和输出路径。
3. 用显式的 `--manifest` 和全新 `--run-dir` 启动 `scripts/train_voxel.py`。训练会写入 `best.pt`、`last.pt`、`history.jsonl` 和图表。
4. 在验证集上选择 checkpoint；最终只在 test split 上报告一次留出集指标。
5. 若新增监督语义，先判断现有 `N2N3DDataset` 的 mapping/residual 接口能否表达；若数据格式或采样行为不同，再扩展 Dataset。只有模型接口发生变化时才增加模型实现。

优化目标和数据处理应作为单独实验记录：记录数据版本与 split seed、manifest 实际记录数、映射关系、归一化/单位、模型和窗口参数、优化器/学习率、停止规则、checkpoint 初始化方式和代码版本。继续训练用 `--resume` 恢复优化器状态；从已有最佳权重开始新的微调则用 `--init-weights` 并创建新 run 目录。

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

### 重建结果存储

配置驱动的 Mode2 重建命令只写入
`even.npy`、`odd.npy`、`full.npy` 和 `pair.json`，写入配置指定的 output root。配置必须指向运行机器可访问的数据和 SDK：

```bash
python scripts/reconstruct_mode2_pairs.py --config /path/to/reconstruction.yaml --dry-run
python scripts/reconstruct_mode2_pairs.py --config /path/to/reconstruction.yaml
```

配置定义 `reconstruction.source_root`、`output_root` 和 `xday_dir`。XDay SDK 是可选重建依赖；它不参与 RAW 训练和推理。

### 批量 PDL 重建

`scripts/reconstruct_pdl_variants.py` processes studies according to the
user-supplied reconstruction-group config. Input layout and filter semantics
are site-specific; pass explicit source, output, and XDay SDK paths. Use
`--dry-run` first. Output count depends on discovered studies and matching
groups; see the generated index and skipped-study records for the run summary.

```bash
python scripts/reconstruct_pdl_variants.py \
  --source-root /path/to/raw-scans \
  --output-root /path/to/reconstruction-output \
  --xday-dir "$XDAY_DIR" --dry-run
```

### 旧版 2D 配对训练

This path requires a compatible 2D config and a user-provided pair dataset:

```bash
python scripts/create_splits.py /path/to/pairs /path/to/manifest.json
python scripts/train.py --config /path/to/task-2d.yaml \
  --manifest /path/to/manifest.json --run-dir /path/to/runs/example
```

`even -> full` and `odd -> full` are correlated full-target supervision when
the full reconstruction contains the input subset views; it is not a clean
ground-truth target.

## 工程结构

| 路径 | 职责 |
| --- | --- |
| `scripts/` | 命令行入口：数据准备、manifest 划分、2D/3D 训练、RAW 推理、teacher 目标生成、评估和 TensorRT 导出。 |
| `configs/` | YAML 任务配置，以及重建组/filter 配置。训练配置组合数据映射、窗口参数、模型和优化参数。 |
| `src/cbct_n2n/reconstruction/` | 扫描发现、奇偶投影拆分、XDay 重建编排和输出索引。 |
| `src/cbct_n2n/data/` | 配对切片和 study-level RAW Dataset、manifest 读取、3D 窗口采样及按 study 分组的 sampler。 |
| `src/cbct_n2n/models/` | 2D UNet、depthwise-separable 2D UNet 和 3D UNet。 |
| `src/cbct_n2n/training/` | 训练循环、loss/指标、scheduler、checkpoint、绘图、体数据推理和融合工具。 |
| `runs/` | 被忽略的运行产物位置约定；checkpoint、history 和训练图由 `--run-dir` 指定。 |

2D 配对训练入口是 `scripts/train.py`；AIR3 体数据训练入口是 `scripts/train_voxel.py`。训练脚本复用 Dataset、模型构造和 `training.trainer.fit`，任务差异主要通过 YAML 配置中的映射、数据窗口和模型/训练参数表达。新增任务通常新增配置和独立 run-dir；只有数据格式、监督语义或模型结构超出现有 Dataset/模型接口时，才需要扩展对应模块。

文档按用途索引见 [`docs/README.md`](docs/README.md)。

## 静态检查

```bash
python -m compileall -q src scripts
python scripts/train_voxel.py --help
python scripts/denoise_single_raw.py --help
```

XDay 未安装时，重建只能执行 dry-run 或 mock 测试；不会把 mock 结果当作真实扫描重建结果。
