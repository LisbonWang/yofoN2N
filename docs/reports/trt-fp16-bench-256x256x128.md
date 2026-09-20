# TensorRT FP16 基准：UNet3D on 256×256×128（本机实测）

**日期**：2026-09-18
**问题**：当前模型处理 256×256×128 数据，ONNX FP16 量化后构建 TensorRT 引擎，显存占用多大？本机运行时间多长？（输入已在 GPU，不计数据拷贝）

**实测脚本**：`scripts/bench_trt_engine.py`（patch 级显存/延迟/精度）、
`scripts/bench_trt_build_peak.py`（构建期显存峰值）
**原始数据**：`outputs/trt_bench/trt_bench_report.json`
**权重**：`runs/air3_mode2_1609_unet3d_h2f_128_r0/best.pt`（AIR3 当前生产 r0）

---

## 1. 结论速览

| 问题 | 答案 |
| --- | --- |
| **运行期显存占用** | **1428 MiB ≈ 1.40 GiB**（fp16 输入输出）/ **1476 MiB ≈ 1.44 GiB**（fp32 输入输出） |
| **单 patch 纯 GPU 前向时间** | **中位数 53.9 ms**（fp16 I/O）、**54.3 ms**（fp32 I/O）；均值 54.4 ms，p99 57.2 ms |
| **吞吐** | 18.4 patch/s（单张 patch，batch=1，不含任何主机↔设备拷贝） |
| **引擎构建时间** | 冷构建 **43 s**，命中 timing cache **37 s**；构建期额外显存峰值 **+1.73 GiB** |
| **引擎文件大小** | 54.8 MiB（ONNX FP16 仅 1.36 MiB） |

对比参照：同一 patch 在 PyTorch eager FP16 下 140.5 ms，FP32 下 195.8 ms ——
**TensorRT FP16 快 2.6×（对 FP16）/ 3.6×（对 FP32）**。

---

## 2. 测试环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti，15.93 GiB，sm_120（Blackwell） |
| 驱动 / CUDA | torch 2.12.1+cu130，CUDA 13.0 |
| TensorRT | **11.3.0.99** |
| ONNX / 转换器 | onnx 1.22.0 + `onnxruntime.transformers.float16.convert_float_to_float16` |
| 模型 | UNet3D `base_channels=16, depth=4, residual=True`，**673,868 参数** |
| patch | `1×1×128×256×256`（与训练/推理所用分块一致），静态 shape |

---

## 3. 模型与引擎

| 产物 | 大小 |
| --- | --- |
| ONNX FP32 | 2.62 MiB |
| ONNX FP16 | **1.36 MiB**（转换后图中无残留 FP32 计算节点） |
| ONNX FP16 + FP32 I/O | 1.36 MiB（仅 I/O 边界保持 fp32） |
| TensorRT engine（序列化） | **54.8 MiB** |

引擎文件 54.8 MiB 远大于权重本身（fp16 权重约 1.35 MiB），因为其中固化了选定的
kernel 实现与 tactic 信息。这是正常现象，不是模型变大。

### 引擎构建

| 项 | fp16 I/O | fp32 I/O |
| --- | --- | --- |
| 网络层数 | 734 | 736 |
| 冷构建 | **43.2 s** | 40.6 s |
| timing cache 命中重建 | 36.9 s | 36.4 s |
| 构建期显存峰值（额外 scratch） | **+1.73 GiB** | 同 |
| 构建后驻留增量 | ~20–38 MiB | 同 |

**坑（已踩）**：tactic workspace 上限设 1 GiB 会直接构建失败——
level-0 的一个候选 tactic 就需要 `1,075,090,136 B`（1.0013 GiB），
日志为 `Skipping tactic 0 due to insufficient memory` 进而
`Could not find any implementation for node {ForeignNode[/encoders.0/conv1/depthwise/Conv.../Add]}`。
**必须给到 4 GiB**（`set_memory_pool_limit(WORKSPACE, 4<<30)`）。

**另一个坑**：TensorRT 11 已删除全部精度 flag（`BuilderFlag` 只剩 DEBUG / TF32 /
DIRECT_IO / …，没有 FP16 / BF16 / INT8）。精度完全由 ONNX 图自身的 dtype 决定，
所以"ONNX 侧先转 FP16 再建引擎"是唯一路径，`export_onnx.py --fp16` 的做法正确。

---

## 4. 运行期显存占用

用 `execute_async_v3` + `set_tensor_address` 直接把 GPU 上的输入/输出缓冲区地址交给
引擎执行，全程零 H2D/D2H。

| 组成 | fp16 I/O | fp32 I/O |
| --- | --- | --- |
| 引擎权重（反序列化后设备驻留） | 64 MiB | 64 MiB |
| Context 激活池（`device_memory_size_v2`） | 1312 MiB | 1328 MiB |
| 输入缓冲区 | 16 MiB | 32 MiB |
| 输出缓冲区 | 16 MiB | 32 MiB |
| **合计** | **1428 MiB** | **1476 MiB** |

激活池约 1.3 GiB 与手算量级一致：level-0 特征图 `128×256×256×16` 在 fp16 下
单张即 256 MiB，编码器/解码器需同时驻留若干张。

**交叉验证**：用 9-16 建好的旧引擎 `runs/1609v2_n2n_3d_h2f_128/best_fp16.engine`
（同架构同 patch、装的是 AIR2 r0 权重）复测 → **1476 MiB / 中位数 54.5 ms**，
与本次 **1476 MiB / 54.3 ms** 相差 <0.5%，说明该占用由架构与 patch 尺寸决定，
与具体权重无关。

---

## 5. 延迟

| 配置 | mean | median | min | p99 | max | 吞吐 |
| --- | --- | --- | --- | --- | --- | --- |
| TRT FP16（fp16 I/O） | 54.4 ms | **53.9 ms** | 52.8 ms | 57.2 ms | 76.5 ms | 18.4/s |
| TRT FP16（fp32 I/O） | 54.4 ms | 54.3 ms | 52.9 ms | 57.3 ms | 57.8 ms | 18.4/s |
| PyTorch FP16 eager | 140.5 ms | 139.6 ms | 138.0 ms | 152.6 ms | 152.6 ms | 7.1/s |
| PyTorch FP32 eager | 195.8 ms | 195.5 ms | 191.9 ms | 202.1 ms | 202.1 ms | 5.1/s |

- 两种 I/O 精度组合速度**没有差别**（54.4 vs 54.4 ms），所以生产上没有必要为了
  省一次 cast 而迁就 dtype——**保持 FP32 I/O 更省事**，管线里的 float32 数据
  可以直接喂。
- 300 次迭代，20 次预热；事件与计算记录在**同一条 CUDA 流**上。
  （第一版把事件记在默认流、计算在另一条流，量出来 0.83 ms 的假值——已修正。）

---

## 6. 精度

| 指标 | 值 |
| --- | --- |
| 输出幅值（`abs().mean()`，fp32 参考） | 0.419 |
| TRT FP16 vs PyTorch FP32：max\|Δ\| | 0.00399 |
| TRT FP16 vs PyTorch FP32：mean\|Δ\| | **2.15e-4**（相对 0.05%） |
| TRT FP16 vs PyTorch FP16：max\|Δ\| | **0.00391 = 1 个 fp16 ULP**（同量级舍入） |
| PyTorch FP16 vs PyTorch FP32：max\|Δ\| | 0.00295 |

TRT 输出与 PyTorch FP16 输出在 1 ULP 内一致，说明引擎确实在执行完整的 fp16 网络，
FP16 化带来的偏差在 3e-3 量级、平均 2e-4，对去噪输出可忽略。

---

## 7. 对整卷推理的影响（估算，乐观上界）

本机现有 33 卷测试集推理实测 **~14 s/卷**（`batch=2`，含 numpy 端 tiling / 梯形加权
融合 / 主机↔设备拷贝；`outputs/test11_cross_mode_run.log`）。按 `_tile_starts` 规则
（D tile 128/stride 112，HW tile 256/stride 240）分块数为：

| 卷尺寸 | 分块数（D×H×W） | batch=2 批次数 | 前向纯 GPU：TRT @54 ms | 前向纯 GPU：PyTorch @140 ms |
| --- | --- | --- | --- | --- |
| 360×640×640 | 4×3×3 = 36 | 18 | 0.97 s | 5.0 s |
| 640×640×640 | 6×3×3 = 54 | 27 | 1.46 s | 7.6 s |

把上表代入，整卷端到端的**乐观上界**（假设主机端开销完全不变）：

- 360³：14.4 s → 14.4 − 5.0 + 0.97 ≈ **10.4 s（1.4×）**
- 640³：14.5 s → 14.5 − 7.6 + 1.46 ≈ **8.4 s（1.7×）**

**注意两个不确定性**：
1. 这两个数字把"整卷耗时 − 前向估算"全部归给主机端，得到 360³ 的主机开销（9.4 s）
   反而大于 640³（6.9 s）——这与直觉矛盾，说明整卷耗时**并不随分块数线性变化**，
   存在与卷尺寸无关的固定开销（整卷 staging、写盘、加权融合的整卷数组操作等）。
   在跑 profiler 之前，这个 1.4×–1.7× 只能当上界看。
2. 2.6× 的速度比**只适用于纯 GPU 前向**这一段。若要在整卷上接近它，必须把
   tiling / 梯形窗融合也搬到 GPU（现在在 numpy 侧）。

建议下一步：对单卷跑一次 `torch.profiler` 或分段计时，量清"前向 / 切片+融合 / 拷贝"
三段占比，再决定 TRT 值不值得以及要不要顺手改融合的实现。

---

## 8. 部署建议

1. **只发布 ONNX（1.36 MiB），在目标机首次运行时构建引擎。** engine 不可跨 GPU
   架构与 TensorRT 版本移植，构建成本仅 ~43 s、+1.73 GiB 临时显存，不值得冒不兼容风险。
2. **workspace 必须 ≥4 GiB**，否则构建直接失败（见 §3）。
3. **保持 FP32 I/O**（`keep_io_types=True`）：速度与 fp16 I/O 相同，管线无需改 dtype。
4. **batch>1 需动态 profile**：现有整卷推理用 `batch=2`。若要让一个引擎覆盖
   batch 1–2，把 optimization profile 的 min/opt/max 设为 batch 1/2/2，
   激活池按 max 分配 → 显存约 2×1.32 GiB ≈ 2.7 GiB + I/O，16 GB 卡足够；
   本报告的数字均为 batch=1。
5. 与既有工具链一致：`scripts/export_onnx.py --fp16` → `scripts/build_trt_engine.py`
   → 用 `scripts/bench_trt_engine.py` 复核显存/延迟/精度。

---

## 附：复现命令

```bash
# ONNX 导出 + FP16 转换 + 构建两个引擎（fp16 I/O、fp32 I/O）
python scripts/bench_trt_engine.py --stage build

# 逐 patch 显存 / 延迟 / 精度（纯 GPU，输入已在显存）
python scripts/bench_trt_engine.py --stage bench --engine fp16    --iters 300
python scripts/bench_trt_engine.py --stage bench --engine fp16io32 --iters 300

# 构建期显存峰值
python scripts/bench_trt_build_peak.py

# 复测任意已有引擎
python scripts/bench_trt_engine.py --stage bench --engine fp16io32 \
    --engine-file runs/1609v2_n2n_3d_h2f_128/best_fp16.engine --iters 300

# PyTorch eager 参照
python scripts/bench_trt_engine.py --stage torch --iters 50
```
