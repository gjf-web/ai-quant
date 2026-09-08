# AI Quant

A machine-learning-based quantitative trading research project.

## Goal

Predict next-day stock returns using historical market data.

## Pipeline

1. Data collection
2. Feature engineering
3. Model training
4. Signal generation
5. Backtesting

## 本地学习环境：D2L + PyTorch

此仓库同时提供《动手学深度学习》1.0.3 的基础学习环境。
沿用[教材官方测试组合](https://en.d2l.ai/chapter_installation/index.html)：
Python 3.9、PyTorch 2.0.0、torchvision 0.15.1、d2l 1.0.3。
另安装 Jupyter Notebook、ipykernel 和现有随机森林模型依赖 scikit-learn。
这是为旧版教材复现准备的环境；升级 Python 或框架时，应单独建立分支并重新验证。

### 日常使用

在终端执行：

```bash
cd /Users/gjf/ai-quant
source .venv/bin/activate
jupyter notebook
```

在 Jupyter 中选择 **Python (ai-quant D2L)** 内核。VS Code 也可以直接选择
仓库里的 `.venv/bin/python`。结束后在终端按 `Ctrl+C` 停止 Jupyter，再运行
`deactivate` 退出环境。此环境用 uv 创建，激活命令不是 `conda activate d2l`。

### 重新安装（macOS Apple Silicon）

需要已经安装 [uv](https://docs.astral.sh/uv/)。从仓库根目录执行：

```bash
uv venv --python 3.9.25 .venv
uv pip sync requirements-macos-arm64.lock --python .venv/bin/python
.venv/bin/python -m ipykernel install --user --name ai-quant-d2l --display-name "Python (ai-quant D2L)"
.venv/bin/python scripts/verify_environment.py
```

`requirements.txt` 记录直接依赖，`requirements-macos-arm64.lock` 固定传递依赖并包含
下载文件的哈希。锁文件针对 Python 3.9 / macOS arm64 生成；其他平台应重新解析：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

需要更新依赖时，在新分支调整版本，然后重新生成锁文件、同步环境并验证：

```bash
uv pip compile requirements.txt --python .venv/bin/python --generate-hashes --output-file requirements-macos-arm64.lock
uv pip sync requirements-macos-arm64.lock --python .venv/bin/python
.venv/bin/python scripts/verify_environment.py
```

PyTorch 2.0.0 的 ARM64 安装包内部 `WHEEL` 标签写为
`cp39-cp39-macosx_11_0_x86_64`，但实际 `_C` 扩展为 Mach-O arm64。
因此 `uv pip check` 和 pip 26 的 `python -m pip check` 均会报告 torch 平台不兼容，
并返回非零退出码。本环境保留原始安装包，不修改包元数据；验证脚本逐项检查已安装包的
必需依赖版本（不包含可选 extras），再实际检查二进制运行与 MPS。平台标签报警仍然存在，
不能把通用检查工具的结果描述为全部通过。

### 教材 Notebook

官方英文版 1.0.3 教材放在本地 `d2l-en/`。若重新克隆仓库，可在根目录下载：

```bash
mkdir -p work d2l-en
curl --fail --location https://d2l.ai/d2l-en-1.0.3.zip -o work/d2l-en-1.0.3.zip
unzip -n work/d2l-en-1.0.3.zip -d d2l-en
```

本次下载归档的 SHA-256：
`79d4f433712c2a99350a27878c9dafe076cb35f1a3c1b4593c59b1650f931e51`。

启动 Jupyter 后进入 `d2l-en/pytorch/chapter_preliminaries/` 开始第二章。
下载的整本教材不纳入 Git。需要保留自己的练习时，将相关 Notebook 复制到自建的
`notebooks/` 目录，提交前清除输出，避免把数据或敏感内容写入提交。
部分章节会在运行时另行下载数据集。

### 验证内容与计算设备

验证脚本不依赖行情接口或数据集下载，检查：

- 当前已安装包的必需依赖是否齐全，版本是否符合约束。
- D2L、PyTorch、torchvision 和 NumPy 之间的兼容性，以及 torchvision 的编译算子。
- 对 `sum(x**2)` 自动求导，核对解析结果 `2*x`。
- 现有随机森林模型能训练，预测概率有限且每行和为 1。
- 若 MPS 可用，比较 Apple GPU 与 CPU 的矩阵乘法及梯度，采用 `rtol=1e-4`、`atol=1e-5`。

Apple GPU 使用 MPS。D2L 1.0.3 的 `try_gpu()` 只考虑 CUDA，因此在 Mac 上返回 CPU
是正常的。前几章可以使用 CPU，后续代码可显式选择设备：

```python
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(100, 100, dtype=torch.float32, device=device)
```

模型和输入需要位于同一设备；MPS 的算子支持和精度限制需逐章验证。
小规模计算可能受调度开销影响，GPU 不一定更快。
验证中的矩阵乘法使用连续 float32 张量，n×n 稠密乘法的计算量为 O(n³)、
存储量为 O(n²)；实际线程调度和 GPU 并行由 PyTorch 后端处理。
此处只验证环境正确性，没有训练收益评估或性能基准，也不能据此判断模型预测能力。

2026-09-08 本机验证：Python 3.9.25 / arm64，CPU 数值检查通过；MPS 已构建且可用，
GPU 矩阵乘法及自动求导与 CPU 对照通过；专用 Jupyter 内核可执行 D2L 导入。

### Git 管理

版本控制包括代码、依赖文件、验证脚本和文档；`.venv/`、`work/`、下载教材、
`data/`、缓存和 `.env` 被忽略。新的研究改动应在独立分支完成，检查差异并验证后提交。
本次安装在 `setup/d2l-environment` 分支完成。
