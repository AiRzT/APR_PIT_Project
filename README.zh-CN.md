# APR-PiT

[English](README.md) | 简体中文

**APR-PiT** 将物理条件化 Transformer 与残差自适应配点加密及剪枝相结合，用于重建隧道火灾纵向中心平面的速度、动态压力、温度和烟气质量分数，服务于通风方案评估。本仓库提供 PyTorch 代码及训练、预测和参考场对比流程。

本研究使用的所有大型数据文件均存放于以下网盘：[Google Drive](https://drive.google.com/drive/folders/1JGFw06BPNis-8mYIm7WMLB3A50Q7rRUH?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1mh7BpiarOyjJi_IX-wVYUw)（提取码：`d5ed`）。

## 论文方法

### 重建任务与变量

论文在纵向中心平面建立准二维模型，令横向速度 `v = 0`、横向导数 `∂/∂y = 0`。式 (24) 的映射为：

$$
\mathcal{G}_{\theta_c}:(t,x,z,V^*)\longmapsto(u,w,\pi,T,C_s).
$$

其中，`u`、`w` 为纵向与竖直速度，`pi` 为动态压力，`T` 为温度，`C_s` 为烟气质量分数。`V* = V / U_ref` 是强制纵向通风的工况条件变量，开放端口基准中取 0；自动微分仅作用于时空坐标。

基准工况使用 `200 m × 6 m` 的中心平面和 `0–120 s` 时间区间。时空坐标按固定参考尺度归一化，速度、温升和烟气质量分数的参考尺度分别为 `5 m/s`、`300 K` 和 `1.0e-3`。

### 物理约束与网络结构

- **控制方程**：低马赫数变密度质量、动量、温度与烟气输运方程，密度由 `rho = p0 / (R T)` 确定；采用 Smagorinsky 涡黏性、简化体积辐射损失和壁面换热。
- **给定火源**：高斯体积热源在三维物理域内归一化，再取中心平面值进入二维能量和烟气残差。
- **上下文编码**：Fourier 特征与输入嵌入共同进入 Transformer；稀疏注意力结合局部时空窗口和全局高残差锚点，注意力偏置包含归一化残差与重力轴对齐项。
- **物理条件化解码**：局部质量守恒、动量残差和浮力指标经调制网络生成仿射系数，调制上下文后由 MLP 输出物理场，并以辅助解码器构造潜在物理一致性损失。
- **条件缓存**：论文采用上一自适应周期的残差与局部物理指标作为当前周期的条件信息；训练损失由当前预测的控制方程残差计算。

论文表 5 的主要网络参数如下：

| 模块 | 设置 |
| --- | --- |
| Transformer | 4 层，特征宽度 256，8 个头（6 个特征头、2 个残差修正头），FFN 宽度 1024 |
| Fourier 编码 | 64 个频率、128 个正余弦特征，标准差 1.0 |
| 稀疏注意力 | 局部窗口 64，全局锚点 32，切换阈值 4096 |
| 物理调制 | 2 个隐藏层，每层 128，Tanh；残差/浮力偏置系数 0.5/0.1 |
| 场解码器 | 4 个隐藏层，每层 256，Tanh |

### 训练与工况评估

训练目标由控制方程、初始条件、边界条件及潜在物理一致性四项组成：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{PDE}}+100\mathcal{L}_{\mathrm{IC}}+100\mathcal{L}_{\mathrm{BC}}+0.1\mathcal{L}_{\mathrm{lat}},\qquad\mathcal{L}_{\mathrm{data}}=0.
$$

配点使用拉丁超立方采样初始化，再按残差加密和剪枝，采样概率与归一化残差的平方成正比，总数不超过 300,000。优化先执行 Adam，再用 L-BFGS 精修；固定评估点集用于比较不同自适应周期的损失和残差。FDS 与实验流场在检查点冻结后用于验证，不参与训练损失、配点选择或检查点选择。

论文分别研究开放端口单火源、纵向通风双火源，以及纵向通风耦合局部顶部排烟三种构型，每种火源布局与边界拓扑分别训练，每个强制通风检查点在一个固定入口风速下训练。强制通风采用式 (51) 的速度边界提升函数满足入口速度；非训练风速评估在同一构型内固定权重，改变 `V*` 和入口提升函数，并与相应参考数据比较。

## 当前代码入口

本说明中的运行命令使用 `configs/tunnel_2d.yaml`：当前二维接口接收无量纲坐标 `(t, x, z)`，输出 `(u, w, pi, T, C_s)`；入口边界通过零速度残差约束。论文中的 `V*` 条件输入及速度边界提升尚未接入该默认脚本。

仓库另含三维配置 `configs/tunnel_3d.yaml` 和评估脚本 `scripts/evaluate_3d.py`，用于三维计算扩展；论文的主要神经重建模型为上述二维中心平面模型。

代码包含物理残差、自适应配点、Adam–L-BFGS 优化、检查点保存、JSONL 训练历史、预测场导出及参考误差计算。支持 CPU 小规模流程测试和 GPU 训练；运行参考场对比前，请从上述网盘下载所需数据。

## 目录结构

以下列出主要配置、脚本和源代码：

```text
APR_PIT_Project/
├── README.md                   # 英文主页（默认）
├── README.zh-CN.md              # 中文说明
├── requirements.txt            # Python 依赖
├── configs/
│   ├── tunnel_2d.yaml           # 二维默认训练配置
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
git clone https://github.com/AiRzT/APR_PIT_Project.git
cd APR_PIT_Project
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
python -m pip install -r requirements.txt
```

依赖列表已包含三维评估脚本 `scripts/evaluate_3d.py` 导出 HDF5 所需的 `h5py`。

使用 GPU 时，应先根据工作站的 CUDA 环境安装对应的 PyTorch 版本，再执行上述命令安装其余依赖。

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

该模式执行 2 次 Adam 更新及 1 次 L-BFGS 迭代，用于快速检查训练流程。完整配置的训练命令见下一节。

测试模式会自动为输出目录添加 `_smoke` 后缀，上述命令的检查点与训练记录保存在：

```text
outputs/tunnel_2d_smoke/
```

## 训练配置

在 CUDA 工作站上运行：

```bash
python scripts/train.py \
  --config configs/tunnel_2d.yaml \
  --device cuda \
  --output outputs/tunnel_2d_full
```

默认二维配置采用以下采样和优化参数，对应论文表 5 的设置：

| 参数 | 配置值 |
| --- | ---: |
| 初始内部配点数 | 150,000 |
| 边界条件采样点数 | 25,000 |
| 初始条件采样点数 | 25,000 |
| 配点数量上限 | 300,000 |
| 高残差候选区域 | 前 10% |
| 采样指数 / 剪枝阈值 | 2.0 / 0.05 |
| APR 更新间隔 | 每 10,000 个训练轮次 |
| Adam 训练轮次 | 30,000 |
| Adam 初始学习率 | 1.0e-3，每 1,000 轮乘以 0.95 |
| PDE / BC / IC 批量大小 | 8,192 / 2,048 / 2,048 |
| L-BFGS 最大迭代次数 | 20,000 |
| 固定残差评估点数 | 50,000 |
| 收敛阈值 | 损失相对变化 < 1.0e-4 且平均残差 < 1.0e-3，连续 3 个周期 |

具体配置见 `configs/tunnel_2d.yaml`；实际停止时点由训练流程及停止准则共同决定。

仓库默认随机种子为 `2026`；论文 PINN 场图和剖面曲线使用种子 `42`。运行对应工况时请设置并记录所用种子。

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

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
