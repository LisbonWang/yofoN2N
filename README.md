# CBCT 图像域 Noise2Noise

本工程从奇偶投影子集分别重建 CBCT 体数据，并使用完整轴向切片训练 2D UNet。原始扫描目录只读，所有子集、重建结果、manifest、checkpoint 和测试输出写入本地输出目录。

## 环境

当前机器 RTX 5060 Ti 使用独立 Conda 环境 `torch`：

```powershell
conda activate torch
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

已验证组合为 `torch 2.12.1+cu130`、CUDA runtime 13.0、NumPy 1.26.4，RTX 5060 Ti 上 `Conv2d` 前向和反向通过。CUDA 12.6 构建不包含 `sm_120` 内核，不能用于这张显卡。

创建环境并安装已验证的 CUDA 构建：

```powershell
conda env create -f environment.yml
conda activate torch
python -m pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -e D:\workspace2\xday_kits
pip install -e D:\workspace2\N2N\.worktrees\cbct-noise2noise --no-deps
$env:XDAY_DIR = "D:\XDay Release\XDay617.2"
```

## 数据准备

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

源目录的 50 个病例中有 47 个 Mode2 和 3 个 Mode1；上述命令会在重建前排除 Mode1。每个入选病例生成 `even.npy`、`odd.npy`、`pair.json`。投影文件按自然数字顺序拆分，配置副本的 `fdkPara.xml/noView` 和 `projectionMatrix.bin` 同步切片；源盘不会被修改。

若某个配置二进制文件已确认是逐视角布局，需要显式提供每个视角的字节数，例如 `--per-view-file fitData.bin=32`。不要仅凭文件名切分 `weightR.bin`、`gx.errors` 等文件：当前扫描样本中的 `weightR.bin` 长度不能被 640 个视角整除，默认会作为共享校准数据原样复制。显式声明的文件若长度与投影视角数不匹配，准备过程会立即报错。

## 划分、训练、测试

```powershell
python scripts\create_splits.py outputs\pairs outputs\splits.json
python scripts\train.py --config configs\default.yaml --manifest outputs\splits.json --run-dir runs\default
python scripts\test.py --config configs\default.yaml `
  --manifest outputs\splits.json --checkpoint runs\default\best.pt `
  --output-dir outputs\test --split test --input-kind both
```

### Full-target supervision

Each Mode2 case produces `even.npy`, `odd.npy`, `full.npy`, and `pair.json`.
The full reconstruction uses all original views. Training uses raw-value L1 for
the correlated pairs `even -> full` and `odd -> full`; this is not independent
Noise2Noise supervision because the target contains the input subset views.

The test command writes `even_denoised.npy` and `odd_denoised.npy` with matching
metrics files. The reference label is `correlated full-view reconstruction`.

For a case-isolated 10-epoch smoke run, use `configs/smoke.yaml`. It uses raw
float32 values, batch size 2, and the first 16 axial slices per train/validation
case; `metric_data_range: 4.0` applies only to PSNR and SSIM.

训练输入是完整二维轴向切片，右侧和底部只补齐到 32 的倍数；没有二维随机 patch 选项。每个切片提供 `even -> full` 和 `odd -> full` 两个样本，增强对输入和全视图目标同步。测试会分别将 even 和 odd 输入与相关的全视图重建进行比较；该参考不是独立 Noise2Noise 目标，也不应视为干净真值。

## 工程结构

`src/cbct_n2n/reconstruction` 负责病例发现、奇偶投影和 XDay 编排；`data` 负责病例级 manifest 和完整切片 Dataset；`models` 是残差 2D UNet；`training` 包含指标、checkpoint、训练和体推理；`scripts` 提供命令行入口。

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

没有设置有效 XDay SDK 时，只能执行 dry-run 和 mock 重建单元测试；不会把 mock 结果当作真实扫描重建结果。
