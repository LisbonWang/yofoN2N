# TensorRT FP16 整卷部署基准：360×640×640（数据全程驻留 GPU）

**问题**：把一整卷 640×640×360 的 float32 数据放在 GPU 上，用 ONNX FP16 + TensorRT 引擎处理，
整体需要多少显存、跑多久？

**答案**：**约 3.2 GiB 显存增量（含进程基线约 4.4 GiB）、约 1.96 秒/卷**。
前提是走**重叠分块**（引擎本身只有 128×256×256）——"整卷一次前向"在这张卡上不可行，见 §6。

日期 2026-09-20，本机 RTX 5060 Ti。

## 1. 结论速览

| 项 | 值 |
| --- | --- |
| 整卷形状 | 360×640×640 = 147.5 M 体素，float32 = 562.5 MiB |
| 引擎 | `unet3d_256x256x128_fp16_io32.engine`（fp16 权重 / fp32 I/O，57.5 MB） |
| 分块 | 1×1×128×256×256，D stride 112、HW stride 240 → **4×3×3 = 36 块** |
| 每块 kernel | **54.45 ms**（分块级单独跑：54.29 ms） |
| **整卷 GPU 时间** | **1960 ms**（中位；5 次 1952.9 – 1970.3 ms） |
| 整卷壁钟 | 1960.4 ms —— 与 GPU 时间差 **0.2 ms**，全 GPU 路径无宿主瓶颈 |
| **显存增量（峰值）** | **3252 MiB**（引擎 1400 + 卷缓冲 1808 + 分块缓冲，拆解见 §4） |
| 进程 CUDA 基线 | 1153 MiB |
| **整卡占用上限** | **约 4.4 GiB**（16 GiB 卡余量充足） |
| 数值一致性 | vs PyTorch 整卷推理：mean \|Δ\| 3.7e-5（= 参考 std 的 0.028%），r = 0.99999977 |

## 2. 环境

- GPU：NVIDIA GeForce RTX 5060 Ti，16311 MiB，sm_120
- Python：`C:/Users/admin/anaconda3/envs/torch/python.exe`（torch 2.12.1+cu130，`KMP_DUPLICATE_LIB_OK=TRUE`）
- tensorrt 11.3.0.99 / onnx 1.22.0
- 权重：`outputs/air3_ypdl0027_bhc_5models/checkpoints/r0.best.pt`（AIR3 r0，epoch 21）

## 3. 为什么是"分块"而不是"整卷一次"

两个约束决定了只能分块，与显存无关：

1. **引擎输入是静态形状**。已发布的引擎按 1×1×128×256×256 构建（与训练 patch 一致），
   换形状必须重建引擎。
2. **UNet3D depth=4 要求每个轴能被 16 整除**。360 不行（360/16 = 22.5），整卷必须先 pad 到 368；
   640 可以。这是架构约束，不是配置问题。

分块采用与训练/推理一致的方案：块 128×256×256，D overlap 16（stride 112）、HW overlap 16（stride 240），
相邻块用三维可分离梯形窗加权融合。360 轴上得到 4 个 D 位置（0/112/224/232）——最后一块与前一块
只错开 8 体素，是为覆盖尾部而不得不加的重叠块。

## 4. 运行期显存拆解（cudaMemGetInfo 差值实测）

| 组成 | MiB | 说明 |
| --- | --- | --- |
| 引擎权重 + 执行上下文 | **1400.0** | 其中引擎自报激活池 1328.0；差值 ≈ 72，大部分是 fp16 权重（引擎文件 57.5 MB） |
| 卷缓冲 输入 f32 | 562.5 | 整卷常驻 |
| 卷缓冲 融合输出 f32 | 562.5 | 加权累加和 |
| 卷缓冲 权重累加 f32 | 562.5 | 梯形窗累加和（融合用） |
| 分块 I/O + 权重块 | 96.0 | 两块 f32 128×256×256 = 64 + 权重块 32 |
| 小计（PyTorch 分配） | 1808.0 | 实测差值 1808.0；torch peak reserved 1820 |
| **实测峰值增量** | **3252.0** | 引擎 + 缓冲全部 |
| 进程 CUDA 上下文基线 | 1152.6 | 进程启动即有，与本模型无关 |

**整卡占用 ≈ 1153 + 3252 = 4405 MiB ≈ 4.4 GiB。**

三点注意：

- **激活池占大头，且它与"块"绑定而不是与"卷"绑定**。分块方案下无论整卷多大，激活池恒为 1328 MiB。
- **卷缓冲才是随卷增长的部分**：3 份 f32 整卷（输入 / 融合输出 / 权重累加）。若输入在 GPU 上以 fp16
  常驻可省 281 MiB，权重累加可用 fp16 再省 281 MiB，但**融合输出必须是 f32**（累加精度）。
- 若数据**不**驻留 GPU（从 CPU 流式喂），模型侧只需 1400 + 96 ≈ 1.5 GiB，代价是每块要过一遍 PCIe。
  本基准按"数据已在 GPU 上、不拷贝"的口径测，故 §1 的 3.2 GiB 是这一口径下的数字。

## 5. 时间拆解

| 项 | 值 |
| --- | --- |
| 每块纯 kernel（分块级基准） | 54.29 ms |
| 每块含切块 + 加权融合（本基准） | 54.45 ms |
| 36 块合计 | 1960 ms |
| 壁钟 − GPU 时间 | 0.4 – 0.2 ms（可忽略） |

切块（GPU 内不连续拷贝）+ 两次加权累加只让每块多花 **0.16 ms（0.3%）**，因为都是访存受限的
小操作，而 54 ms 的卷积核把带宽和 SM 都占满了，这些开销被重叠掉。**结论：全 GPU 流水线的开销
可以忽略，时间 ≈ 分块数 × 单块延迟。**

吞吐换算：约 **0.51 卷/秒**；200 卷训练集 ≈ 6.5 分钟；1000 卷 ≈ 33 分钟（单卡、无批处理）。

## 6. "整卷一次前向"需要多少显存（外推 + 实测尝试）

激活池与**体素数精确线性**——两点实测，系数完全一致：

| 引擎输入 | 体素数 | 引擎自报激活池 | B/体素 |
| --- | --- | --- | --- |
| 128×256×256 | 8,388,608 | 1328 MiB (1,392,458,752 B) | **166.0** |
| 128×384×384 | 18,874,368 | 2988 MiB (3,133,147,136 B) | **166.0** |

据此外推整卷（必须 pad 到 368×640×640 = 150,732,800 体素）：

| 组成 | 大小 |
| --- | --- |
| 激活池 | 166.0 B × 150,732,800 = **23.3 GiB** |
| fp32 I/O（输入 + 输出） | 1.12 GiB |
| 权重 | 0.11 GiB |
| **合计** | **约 24.5 GiB** |

**16 GiB 卡装不下，差了一倍还多。** 实测尝试也印证了这一点（本条为实测，非外推）：

- 以 1×1×368×640×640 构建引擎，跑了 **13 分 41 秒**后失败，过程里显存被吃到
  **15621 / 16311 MiB**，最终报
  `UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory on requested size of 4296315608`
  →
  `Could not find any implementation for node {ForeignNode[...]}`。
  注意这条 "Could not find any implementation" **不是**算子不支持，仍是显存不足的老面孔。
- 甚至导出整卷 ONNX 这一步在本机就不成立：`--input-size 368x640x640` 时 `torch.onnx.export`
  直接 **Segmentation fault**（同一脚本导出 128×384×384 正常，1.4 MB）——图的规模超出了导出路径的承受范围。

要真正一次过整卷，需要 ≥32 GiB 的卡（且导出路径也得换新 exporter），收益是省掉块间重叠
（36 块 → 1 次，重叠部分的算力浪费从 ~18% 降到 0）。**在 16 GiB 级设备上，分块是唯一可行路径。**

## 7. 精度

与已有 PyTorch 整卷推理产物 `full.bhc.r0.raw`（同一权重、同一分块参数）对比：

| 指标 | 值 |
| --- | --- |
| mean \|Δ\| | 3.699e-5 |
| p99 \|Δ\| | 2.102e-4 |
| max \|Δ\| | 1.197e-2 |
| 参考数据 std | 0.13094 |
| 相对（mean \|Δ\| / std） | 2.8e-4（0.028%） |
| Pearson r | 0.99999977 |

差异来源是 fp16 权重/激活（引擎）对 fp32 推理，符合 fp16 ULP 量级；max 值出现在强边缘的少量体素。
**融合权重口径一致（同一梯形窗），因此这里的差就是精度差，不含实现差。**

## 8. 部署建议

1. **一律用分块**（128×256×256，D/HW overlap 16）。激活池是常数 1328 MiB，与卷大小无关。
2. **按"每卷 1.7 GiB 卷缓冲 + 1.4 GiB 模型"预留显存**。若要并行 N 卷，缓冲按 N 线性增长
   （N=4 时约 8.2 GiB，16 GiB 卡仍可行）。
3. **别把卷缓冲全放在 f32 上除非必要**：权重累加和输入都可以 f16，融合输出保持 f32。
   全 f32 是本次实测的口径（因为输入数据就是 f32）。
4. 吞吐 ~2 s/卷/卡。若目标是 200 卷批处理，单卡 6.5 分钟；多卡按卷切分即可线性扩展
   （引擎不可跨架构移植，每台机器首次运行自行构建，约 40 s）。
5. **不要把"整卷单次"列入方案**：显存差 1.5 倍以上，且 ONNX 导出路径本身不成立。

## 9. 复现命令

```bash
export PATH="/usr/bin:/bin:$PATH"; cd /g/workspace/N2N
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
PY=/c/Users/admin/anaconda3/envs/torch/python.exe

# 整卷常驻 GPU 基准（显存拆解 + 时间 + 可选落盘做精度比对）
$PY scripts/bench_trt_volume_resident.py \
  --engine outputs/trt_bench/unet3d_256x256x128_fp16_io32.engine \
  --shape 360 640 640 --warmup 3 --repeats 5 \
  --input outputs/air3_ypdl0027_bhc_5models/full.bhc.raw \
  --save /tmp/trt_fused.raw \
  --json outputs/trt_bench/resident_360x640x640_fp16_io32.json

# 激活池线性验证（第二个数据点）
$PY scripts/export_onnx.py --checkpoint outputs/air3_ypdl0027_bhc_5models/checkpoints/r0.best.pt \
  --input-size 128x384x384 --fp16 --no-validate --output /tmp/scale.onnx
$PY scripts/build_trt_engine.py --onnx /tmp/scale.onnx --fp16 \
  --input-shape 1x1x128x384x384 --workspace-gb 4 --no-verify --output /tmp/scale.engine
# 构建日志里的 "Total Activation Memory" 即为该点的激活池

# 分块级参照（既有基准）
$PY scripts/bench_trt_engine.py --stage bench --engine fp16io32 --iters 300
```

产物：`outputs/trt_bench/resident_360x640x640_fp16_io32.json`。
