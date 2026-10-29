# APR-PiT

简体中文 | [English](README.en.md)

**APR-PiT** 是一种残差自适应物理信息 Transformer，用于重建隧道火灾中浮力驱动的热量与烟气输运过程。本仓库提供其 PyTorch 实现，以下说明以隧道中心平面的二维重建流程为主。

本研究使用的所有大型数据文件均存放于以下网盘：[Google Drive](https://drive.google.com/drive/folders/1JGFw06BPNis-8mYIm7WMLB3A50Q7rRUH?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1mh7BpiarOyjJi_IX-wVYUw)（提取码：`d5ed`）。

代码包含 APR-PiT 网络、控制方程残差、自适应配点、两阶段优化及预测场评估流程。

## 项目说明

本项目支持 CPU 小规模流程测试和 GPU 论文规模训练。运行参考场对比前，请从上述网盘下载所需数据。

仓库另含三维配置 `configs/tunnel_3d.yaml` 和三维评估脚本 `scripts/evaluate_3d.py`；本说明中的快速验证、训练及绘图命令使用二维配置。

## 主要功能

- **纯物理信息训练**：不将 FDS 场快照作为优化损失中的监督标签。
- **输入与输出**：二维模式输入无量纲坐标 `(t, x, z)`，输出物理量 `(u, w, pi, T, C_s)`。
- **模型结构**：Fourier 位置编码、4 个 Transformer 模块及输出解码器。
- **物理引导注意力**：结合归一化偏微分方程（PDE）残差与竖直方向对齐关系构造注意力偏置。
- **稀疏上下文**：局部注意力窗口与全局高残差锚点相结合。
- **残差调制**：基于残差的特征调制及辅助潜在物理损失。
- **控制方程**：准二维、低马赫数条件下的质量、动量、温度与烟气输运方程。
- **物理模型**：Smagorinsky 涡黏性、高斯热源、简化辐射损失及壁面换热。
- **自适应采样**：在最多 300,000 个配点的预算内，自适应加密高残差区域并剪枝低贡献点。
- **两阶段优化**：先使用 Adam，再使用 L-BFGS 精修。
- **训练监测**：在固定评估点集上跟踪损失和残差，并应用论文设定的停止准则。
- **结果导出**：保存模型检查点、JSONL 损失历史、预测场、参考场误差指标及场分布图。

注意力模块采用坐标对角自动微分路径。相邻点的键/值（K/V）上下文仍参与参数训练，但其坐标计算图停止梯度传播，以避免计算物理信息神经网络（PINN）导数时误将跨点的 Jacobian 项求和。

## 目录结构

以下列出主要配置、脚本和源代码：

```text
ARP_PIT_Project/
├── configs/
│   ├── tunnel_2d.yaml           # 二维论文规模配置
│   └── tunnel_3d.yaml           # 三维配置
├── scripts/
│   ├── train.py                 # 训练入口：Adam、APR 与 L-BFGS
│   ├── evaluate.py              # 二维规则网格预测与场分布绘图
│   ├── evaluate_3d.py           # 三维场评估
│   ├── compare_reference.py     # 同网格参考场误差计算
│   └── plot_history.py          # 训练历史绘图
├── src/apr_pit/
│   ├── model.py                 # Fourier 编码、稀疏 Transformer 与解码器
│   ├── physics.py               # PDE、初始条件与边界条件残差
│   ├── sampling.py              # 拉丁超立方采样点池与 APR 控制器
│   ├── trainer.py               # 优化流程与检查点保存
│   ├── config.py                # 配置读取
│   ├── metrics.py               # 误差指标
│   └── utils.py                 # 通用工具
└── tests/test_core.py           # 核心单元测试
```

## 环境配置

论文规模训练建议使用支持 CUDA 的 PyTorch。CPU 可用于单元测试和小规模流程验证。

### 1. 获取代码

```bash
git clone https://github.com/AiRzT/ARP_PIT_Project.git
cd ARP_PIT_Project
```

以下命令均在仓库根目录执行。

### 2. 创建独立环境

以 Conda 为例：

```bash
conda create -n apr-pit python=3.10 -y
conda activate apr-pit
```

### 3. 安装依赖

```bash
pip install torch numpy pyyaml matplotlib
```

使用 GPU 时，应根据工作站的 CUDA 环境安装对应的 PyTorch 版本。

若需要运行三维评估脚本，还需安装：

```bash
pip install h5py
```

以下示例使用 Bash/Zsh 风格的命令换行。`MPLCONFIGDIR` 用于指定 Matplotlib 的配置与缓存目录。

## 快速验证

先运行单元测试：

```bash
python -m unittest discover -s tests -v
```

再使用少量采样点运行小规模流程测试（smoke test）：

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/train.py \
  --smoke-test --device cpu --output outputs/tunnel_2d
```

该模式执行 2 次 Adam 更新及 1 次 L-BFGS 迭代，用于快速检查训练流程。论文规模训练命令见下一节。

测试模式会自动为输出目录添加 `_smoke` 后缀，上述命令的检查点与训练记录保存在：

```text
outputs/tunnel_2d_smoke/
```

## 论文规模训练

在 CUDA 工作站上运行：

```bash
python scripts/train.py \
  --config configs/tunnel_2d.yaml \
  --device cuda \
  --output outputs/tunnel_2d_full
```

默认二维配置中的主要参数如下：

| 参数 | 配置值 |
| --- | ---: |
| 初始内部配点数 | 150,000 |
| 边界条件采样点数 | 25,000 |
| 初始条件采样点数 | 25,000 |
| 配点数量上限 | 300,000 |
| APR 更新间隔 | 每 10,000 个训练轮次 |
| Adam 训练轮次 | 30,000 |
| L-BFGS 最大迭代次数 | 20,000 |

具体配置见 `configs/tunnel_2d.yaml`；实际停止时点由训练流程及停止准则共同决定。

## 预测场评估与绘图

### 在规则网格上评估模型

使用训练完成的检查点，导出 10、60 和 120 秒时的预测场：

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/evaluate.py \
  outputs/tunnel_2d_full/checkpoint_final.pt \
  --times 10 60 120 --device cuda
```

对每个指定时刻，脚本导出压缩的 `.npz` 场数据，以及包含速度大小、温度和烟气质量分数的白底 PNG 图像。默认结果保存在检查点所在目录下的 `evaluation/` 子目录。

### 绘制训练历史

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/plot_history.py \
  outputs/tunnel_2d_full/history.jsonl
```

## 与参考场对比

下载参考数据并对齐预测结果的网格与时刻后，运行：

```bash
python scripts/compare_reference.py \
  outputs/tunnel_2d_full/evaluation/fields_t120s.npz \
  data/fds_fields_t120s.npz \
  --fields temperature smoke
```

`data/fds_fields_t120s.npz` 为示例路径，请替换为下载并整理后的参考文件实际路径。参考文件需满足：

- 包含 `--fields` 指定的同名字段；上述示例需要 `temperature` 和 `smoke`。不指定该参数时，默认比较 `u`、`w`、`temperature` 和 `smoke`。
- 所选字段的数组形状均与预测结果一致，为 `(nz, nx)`。
- 数据对应相同的时刻、空间网格和物理单位；请在比较前完成坐标检查和必要的网格插值。

上述命令默认将误差指标保存为预测文件同目录下的 `fields_t120s_metrics.json`。

## 复现说明

复现流程包括环境配置、物理信息训练、自适应配点更新、预测场导出及参考场误差计算。

运行时请记录硬件、PyTorch 和 CUDA 版本、随机种子及优化器配置，并保存模型检查点、训练历史和评估结果。

## 引用

使用本项目时，请引用配套论文。正式引用信息将在论文发表后补充。

## 许可证

原英文 README 声明本项目采用 MIT 许可证，但当前仓库尚未包含独立的 `LICENSE` 文件。具体许可条款待维护者补充该文件后确认。
