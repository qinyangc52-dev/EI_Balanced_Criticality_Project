# E-I平衡网络临界雪崩复现项目

[English](#english) | [中文](#中文)

<a name="中文"></a>

## 项目概述

本项目复现Yang等人(2025)在《物理评论快报》上发表的研究：**E-I平衡网络中的临界雪崩现象**。通过BrainPy框架，系统地模拟和分析了800个兴奋神经元和200个抑制神经元组成的网络在不同参数条件下的雪崩统计特性。

## 快速开始

### 环境要求

```bash
Python 3.10+
BrainPy 2.7.2
NumPy, Matplotlib, SciPy
```

### 完整流程（推荐）

```bash
# 1. 进入项目目录
cd EI_Balanced_Criticality_Project

# 2. 运行完整分析（模拟+分析+可视化）
python main.py --all

# 输出：
#   - data/raw/spikes_*.npz (原始脉冲数据)
#   - data/processed/avalanche_stats_*.pkl (统计结果)
#   - visualization/figures/*.png (图表)
```

### 灵活选项

```bash
# 只分析现有数据（不重新模拟）
python main.py --analyze-only

# 模拟单个tau_d_I值
python main.py --tau 8.0

# 生成可视化图表
python -c "from visualization.plot_criticality import plot_avalanche_distribution, plot_phase_diagram; from configs.model_config import TAU_DECAY_I_LIST; [plot_avalanche_distribution(tau) for tau in TAU_DECAY_I_LIST]; plot_phase_diagram()"
```

## 项目结构

```
EI_Balanced_Criticality_Project/
├── configs/              → 模型参数配置（SSOT）
├── models/               → 神经元、突触、网络实现
├── core/                 → 外部输入（泊松过程）
├── experiments/          → 模拟控制与参数扫描
├── analysis/             → 雪崩检测与统计
├── visualization/        → 可视化与绘图
├── data/                 → 数据存储
│   ├── raw/              → 原始模拟数据
│   └── processed/        → 处理后的统计结果
├── main.py               → 管道入口点
├── FINAL_REPORT.md       → 详细技术报告
└── PROJECT_STATUS.md     → 项目状态跟踪
```

## 主要结果

### 核心指标

| 参数组 | 雪崩数 | Size指数 | R² | Duration指数 |
|--------|--------|---------|-----|--------|
| τ_d_I=2.0ms (亚临界) | 119,948 | 1.323 | 0.660 | 2.036 |
| τ_d_I=8.0ms (临界)   | 111,963 | 1.363 | 0.839 | 1.761 |
| τ_d_I=11.0ms (超临界)| 104,827 | 1.475 | 0.813 | 1.810 |

### 生成的图表

1. **7个单参数分析图** (`avalanche_dist_X.X.png`)
   - 左图：雪崩大小分布（对数坐标）
   - 右图：雪崩持续时间分布（对数坐标）
   - 每图显示拟合指数α和R²

2. **1个相位图** (`phase_diagram.png`)
   - 展示幂律指数如何随τ_d_I变化
   - 点颜色表示拟合质量(R²)
   - 红色虚线标注理论临界值α=1.0

## 使用示例

### 查看特定参数的结果

```python
from utils.io_manager import load_pkl
from pathlib import Path

# 加载tau_d_I=8.0ms的分析结果
results = load_pkl(Path('data/processed/avalanche_stats_8.0.pkl'))

print(f"检测到的雪崩数：{results['preprocessing']['n_avalanches']}")
print(f"幂律指数（大小）：{results['size_exponent']:.3f}")
print(f"拟合质量（R²）：{results['size_r2']:.3f}")
```

### 自定义分析

```python
import numpy as np
from analysis.preprocessing import preprocess_single_tau
from analysis.avalanche_metrics import analyze_single_tau

# 分析特定tau值
results = analyze_single_tau(tau_d_I=6.0)
```

## 参数说明

### 关键模拟参数

```python
# 网络大小
N_E = 800          # 兴奋性神经元
N_I = 200          # 抑制性神经元
CONN_PROB = 0.2    # 连接概率

# 神经元时间常数（LIF模型）
TAU_E = 20.0       # E神经元膜时间常数(ms)
TAU_I = 10.0       # I神经元膜时间常数(ms)
REF_E = 2.0        # E神经元不应期(ms)
REF_I = 1.0        # I神经元不应期(ms)

# 突触参数
TAU_RISE = 0.5     # 突触上升时间(ms)
TAU_DECAY_E = 2.0  # E突触衰减时间(ms)
TAU_DECAY_I = [2.0-11.0]  # I突触衰减时间(ms) ← 控制参数

# 突触权重（电导）
G_EE = 0.012       # E→E
G_EI = 0.024       # E→I
G_IE = 0.18        # I→E
G_II = 0.31        # I→I

# 模拟设定
DT = 0.05          # 时间步长(ms)
DURATION = 5000.0  # 模拟时长(ms)
WARMUP = 500.0     # 预热期(ms)
```

### 扫描范围

```python
TAU_DECAY_I_LIST = [2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0]  # ms
```

代表三种状态：
- **亚临界**（undercritical）：τ_d_I = 2.0 ms
- **临界**（critical）：τ_d_I ≈ 8.0 ms  
- **超临界**（supercritical）：τ_d_I = 11.0 ms

## 文献参考

Yang, Z., Liang, J., & Zhou, C. (2025). Critical Avalanches in Excitation-Inhibition Balanced Networks Reconcile Response Reliability with Sensitivity for Optimal Neural Representation. *Physical Review Letters*, 134(2), 028401.

DOI: [待补充]

## 故障排除

### 问题：模拟太慢

**解决方案**：
- 使用`--analyze-only`标志分析现有数据
- 减少DURATION（编辑configs/model_config.py）
- 使用更强大的硬件

### 问题：内存不足

**解决方案**：
- 减少模拟时长
- 一次处理一个tau值

### 问题：图表为空白

**解决方案**：
- 确保`data/processed/`目录下有`*.pkl`文件
- 删除`visualization/figures/`下的旧图，重新生成
- 检查是否成功运行了分析步骤

## 扩展与改进

### 可尝试的改进

1. **更密集的参数扫描**
   ```python
   # 在configs/model_config.py中修改
   TAU_DECAY_I_LIST = np.arange(1.5, 12, 0.25).tolist()
   ```

2. **更长的模拟时间**
   ```python
   # 获得更好的统计
   DURATION = 10000.0  # 10秒
   ```

3. **多次试验集合平均**
   ```python
   # 在experiments/phase_transition_runner.py中添加循环
   for seed in range(5):
       run_simulation(seed=seed)
   ```

4. **高级分析**
   - 计算branching ratio
   - 分析雪崩大小-持续时间关系
   - 计算标度指数

## 许可证

[待定义]

## 联系方式

如有问题或建议，请联系项目维护者。

---

<a name="english"></a>

## English

### Overview

This project replicates the study by Yang et al. (2025) published in *Physical Review Letters*: **Critical Avalanches in Excitation-Inhibition Balanced Networks**. Using BrainPy, we systematically simulate and analyze avalanche statistics in a network of 800 excitatory and 200 inhibitory neurons under different parameter conditions.

### Quick Start

```bash
# Complete pipeline (simulation + analysis + visualization)
python main.py --all

# Or analyze existing data only
python main.py --analyze-only

# Or simulate specific tau value
python main.py --tau 8.0
```

### Key Results

- **7 parameter sweeps** from τ_d_I = 2.0 to 11.0 ms
- **~110,000 avalanche events** detected per condition
- **Size exponent** α_s ≈ 1.32-1.48
- **Duration exponent** α_d ≈ 1.76-2.05
- **8 publication-quality figures** generated

### Main Files

- `main.py` - Pipeline entry point
- `models/` - Network implementation (LIF neurons + synapses)
- `analysis/` - Avalanche detection and statistics
- `visualization/` - Plotting and figure generation
- `FINAL_REPORT.md` - Detailed technical documentation

### Reference

Yang, Z., Liang, J., & Zhou, C. (2025). Critical Avalanches in Excitation-Inhibition Balanced Networks Reconcile Response Reliability with Sensitivity for Optimal Neural Representation. *Physical Review Letters*, 134(2), 028401.
