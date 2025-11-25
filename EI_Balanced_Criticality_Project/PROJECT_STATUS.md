# E-I临界雪崩项目 - 状态报告

**日期**: 2025-11-24
**版本**: 1.0

## 完成情况

### ✅ 已完成阶段

#### 阶段一：参数配置 (SSOT)
- [x] 所有参数严格对应 Yang et al. (2025) PRL 论文
- [x] 网络拓扑：N_E=800, N_I=200, p=0.2
- [x] 神经元参数：tau_E=20ms, tau_I=10ms, ref_E=2ms, ref_I=1ms
- [x] 突触参数：tau_rise=0.5ms, tau_decay_E=2ms, G_EE=0.012等
- [x] 模拟参数：DT=0.05ms, DURATION=5000ms, WARMUP=500ms

#### 阶段二：网络实现
- [x] LIF神经元模型 (LifRefE, LifRefI)
- [x] 双指数电导突触 (DualExpCondSyn)
- [x] 6个投影（E→E/I, I→E/I, Ext→E/I）
- [x] 泊松外部输入

#### 阶段三：模拟运行
- [x] 成功生成7个tau_d_I值的模拟数据 (2.0-11.0ms)
- [x] 每个模拟：5000ms，约1.756M个脉冲
- [x] 数据保存：data/raw/spikes_*.npz

### ⚠️ 进行中/有问题的阶段

#### 阶段四：雪崩分析

**当前状态**：代码运行但统计结果不符合预期

**具体问题**：

1. **ISI计算不当**
   - 当前：平均ISI = 0.003ms (太小)
   - 原因：1.756M spikes / 4500ms = 390kHz = 0.0026ms ISI
   - 影响：bin_width = 0.003ms导致几乎所有spike被独立视为事件

2. **幂律指数不符合预期**
   - 当前：alpha_size ≈ 1.3-1.5 (偏高)
   - 期望（论文）：alpha_size ≈ 1.0-1.2 at criticality
   - 当前：alpha_duration ≈ 1.8-2.0 (接近预期)
   - 期望（论文）：alpha_duration ≈ 2.0

3. **雪崩定义歧义**
   - 计划书5.2说：bin width = Delta_t = 平均ISI
   - 但这会导致过度碎片化
   - 需要重新审视论文Appendix B的precise定义

**可能的根本原因**：

- 网络的高发放率（390kHz）与低bin宽度(0.003ms)不兼容
- 论文的bin_width定义可能涉及额外的时间尺度因子
- 可能需要使用论文中的"Delta_t = <ISI> × MULTIPLIER"或固定的物理时间单位

## 输出文件

### 模拟数据
```
data/raw/
├── spikes_2.0.npz    ✓ 1.756M spikes
├── spikes_4.0.npz    ✓ 1.756M spikes
├── ...
└── spikes_11.0.npz   ✓ 1.756M spikes
```

### 分析结果
```
data/processed/
├── avalanche_stats_2.0.pkl   (alpha_size=1.323, R2=0.660)
├── avalanche_stats_4.0.pkl   (alpha_size=1.401, R2=0.704)
├── ...
└── avalanche_stats_11.0.pkl  (alpha_size=1.475, R2=0.813)
```

### 可视化
```
visualization/figures/
├── avalanche_dist_2.0.png    ✓ 有效
├── avalanche_dist_4.0.png    ✓ 有效
├── ...
├── avalanche_dist_11.0.png   ✓ 有效
└── phase_diagram.png         ✓ 有效
```

## 下一步建议

### 关键优先事项

1. **重新审视论文Appendix B**
   - 确认bin_width的精确定义
   - 是否包含时间尺度因子（如毫秒→秒的转换）
   - 检查是否应使用固定的物理时间窗口（如10-50ms）

2. **参数微调**
   - 如果bin_width仍过小，尝试：
     - Option A: bin_width = 10ms (固定物理时间)
     - Option B: bin_width = mean_isi × 100 (相对倍数)
     - Option C: 使用论文中的AVALANCHE_BIN_MULTIPLIER

3. **验证网络参数**
   - 检查是否需要调整突触权重(G_IE等)以达到临界性
   - 验证外部输入强度是否适当

4. **统计学验证**
   - 收集更多数据（延长模拟时间或多次试验）
   - 使用更鲁棒的幂律拟合方法

## 代码质量

- ✓ 模块化结构清晰
- ✓ 参数集中管理(SSOT)
- ✓ 完整的数据流水线
- ⚠️ 需要补充Appendix B的precise算法文档

## 联系方式

对于问题和改进建议，请联系项目负责人。
