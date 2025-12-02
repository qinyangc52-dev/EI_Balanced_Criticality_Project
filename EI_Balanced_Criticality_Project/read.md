project_root/
├── README.md                  # 项目说明书
│                              
├── main.py                    # 主程序入口
│                                负责完整工作流：模拟 → 分析 → 输出统计结果
│                                支持 --all（全参数扫描） 或 --tau（单参数调试）
├── quick_test.py              # 快速验证脚本
│                                短时程（200ms）模拟，检查发放率是否合理（1-5 Hz）
│
├── configs/
│   └── model_config.py        # 参数单点真相源（SSOT）
│                                定义 N, p, τ, g, TAU_DECAY_I_LIST, 路径等所有全局常量
│
├── core/
│   └── inputs.py              # 外部泊松输入生成器
│                                PoissonInput 类，生成背景电导噪声（带正确归一化）
│
├── models/
│   ├── neurons.py             # LIF 神经元实现（带绝对不应期）
│   │                            LifRefE（兴奋性）、LifRefI（抑制性）
│   ├── synapses.py            # 双指数电导突触模型
│   │                            DualExpCondSyn（Rise + Decay）
│   └── network.py             # 完整兴奋-抑制平衡网络组装
│                                BalancedNetwork（包含 E/I 神经元 + 4种突触连接）
│
├── experiments/
│   └── phase_transition_runner.py   # 相变扫描主运行器
│                                      遍历所有 τ^d_I，分块模拟（防OOM），输出原始脉冲 .npz
│
├── analysis/
│   ├── preprocessing.py       # 预处理 + 雪崩检测
│         ├─ 去温（warmup）
│         ├─ 计算人口 ISI → 确定分箱宽度 Δt（论文 Appendix B）
│         └─ 空箱分隔法检测雪崩 → 提取大小 S 和时长 T
│   └── avalanche_metrics.py   # 临界性指标计算
│         ├─ MLE 拟合幂律（τ, α）
│         ├─ KS 距离
│         └─ Crackling Noise Relation 验证 + 标度误差
│                                输出 .pkl 统计结果
│
├── visualization/
│   ├── plot_criticality.py         # 主图（Fig 1 复现）
│         ├─ 光栅图（亚/临/超临界示例）
│         ├─ 雪崩大小/时长分布
│         └─ KS 距离 vs τ^d_I 曲线（定位临界点）
│   └── plot_scaling_relation.py    # 补充图 S2 复现
│                                      <S>(T) 标度关系，验证 γ = (α-1)/(τ-1)
│
├── utils/
│   └── io_manager.py          # 统一的 I/O 封装
│                                save/load npz、pkl，自动创建文件夹，跨平台路径兼容
│
└── data/
    ├── raw/                   # 原始脉冲数据（.npz）
    └── processed/             # 分析后的统计结果（.pkl）
└── figures/                   # 自动生成的出版级图片（.png/.pdf）



## 敏感性与可靠性的测量
本节描述了在兴奋-抑制（E-I）平衡网络模型和分支过程模型中测量**敏感性**（sensitivity）和**可靠性**（reliability）的方法，如 Yang et al. (2025) 论文所述。这些指标用于评估网络对外界信号的响应，突出临界态如何调和敏感性（检测小扰动的能力）与可靠性（跨试验一致响应）。
### 平衡网络中的敏感性
- **方法**：向网络中额外添加频率为 25 Hz 的泊松输入。
- **定义**：兴奋性神经元在接收到该额外输入后，平均发放率的相对变化量。
- **目的**：量化网络对扰动的敏感响应，在临界态附近达到最大值。
### 平衡网络中的可靠性
- **方法**：针对固定的网络连接结构，进行 100 次独立试验，每次添加相同的“frozen” 25 Hz 泊松信号。
- **定义**：跨试验的小时间窗发放率的 Fano 因子的倒数（1 / Fano factor，其中 Fano factor = variance / mean）。类似于先前研究 [12] 的定义。
  - 时间窗：50 ms，滑动步长 20 ms。
  - Fano 因子在所有时间窗上取平均。
- **目的**：测量对相同刺激的试验间一致性，在平衡模型的临界态附近达到最大值。
### 分支过程模型中的敏感性
- **方法**：为了模拟平衡网络中的时序信号，通过在泊松分布时间点随机激活神经元引入输入。
  - 从均值为 3 的泊松分布中抽取间隔序列 {I_k}（Pois(μ = 3)）。
  - 计算输入时刻 T_i = Σ_{k=1}^i I_k。
  - 对于每个 T_i，随机选择神经元 N_i 并将其状态置为激活（S_{N_i}(T_i) = 1）[见图 S5(a)]。
- **定义**：接收信号后的激发活动概率与自发活动概率之差。
  - 自发活动概率：初始激活所有神经元，让系统收敛至稳态所得。
  - 这不同于传统动态敏感性 [10]（整个模拟中固定单个节点激活）；此处使用稀疏输入，导致峰值敏感性略向超临界区偏移。
- **目的**：评估对稀疏扰动的响应，在临界态附近达到最大值，但因概率本质而有所差异。

### 分支过程模型中的可靠性
- **方法**：针对固定的网络连接结构，进行 100 次独立试验，每次添加与敏感性测量相同的 frozen 泊松信号 [图 S5(a)]。
- **定义**：跨试验的小时间窗发放率的 Fano 因子的倒数，与平衡模型一致。
- **目的**：评估一致性，但不同于平衡模型，在临界态可靠性最低（因固有随机性）。

### 模拟结果与可视化
- 图 1(c) 和 1(f) 的结果为不同输入信号和网络结构下的平均值，阴影区域表示标准差。
- 此表示方式与图 2(a)、2(g)、3(a)、3(b) 和 4(c) 一致。
- **实现注意**：在项目中，可将这些测量集成到 `analysis/` 模块（例如扩展 `avalanche_metrics.py` 以计算敏感性和可靠性）。使用 frozen 泊松信号确保可重复性，并对多试验/网络取平均。

完整细节请参阅原论文 Appendix B。如需实现，确保扫描参数如 τ^d_I 以观察相变。




 python experiments/sensitivity_reliability.py

============================================================
Running Balanced Network: Sensitivity & Reliability
============================================================
Generating Frozen Signal Template...

Processing tau_I = 2.0 ms...
  Sensitivity: 0.0548 (Base=10.36Hz, Sig=10.93Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.1165

Processing tau_I = 2.5 ms...
  Sensitivity: 0.0705 (Base=10.53Hz, Sig=11.27Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0960

Processing tau_I = 3.0 ms...
  Sensitivity: 0.0865 (Base=10.76Hz, Sig=11.69Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0935

Processing tau_I = 4.0 ms...
  Sensitivity: 0.0598 (Base=10.92Hz, Sig=11.57Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0830

Processing tau_I = 5.0 ms...
  Sensitivity: 0.0069 (Base=11.34Hz, Sig=11.42Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0724

Processing tau_I = 6.0 ms...
  Sensitivity: 0.0079 (Base=11.39Hz, Sig=11.48Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0720

Processing tau_I = 7.0 ms...
  Sensitivity: 0.1374 (Base=10.72Hz, Sig=12.19Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0665

Processing tau_I = 7.5 ms...
  Sensitivity: 0.0718 (Base=11.11Hz, Sig=11.90Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0670

Processing tau_I = 8.0 ms...
  Sensitivity: 0.0083 (Base=11.51Hz, Sig=11.61Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0644

Processing tau_I = 8.2 ms...
  Sensitivity: 0.1047 (Base=11.04Hz, Sig=12.20Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0630

Processing tau_I = 9.0 ms...
  Sensitivity: -0.0459 (Base=11.65Hz, Sig=11.11Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0583

Processing tau_I = 10.0 ms...
  Sensitivity: 0.0444 (Base=11.43Hz, Sig=11.94Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0594

Processing tau_I = 11.0 ms...
  Sensitivity: 0.0169 (Base=11.46Hz, Sig=11.65Hz)
  Measuring Reliability (100 trials)...
  Reliability: 0.0549

============================================================
Running Branching Model (Control)
============================================================
Processing branching param m = 0.8...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 13676.22it/s]
  Sens=0.000, Rel=0.054
Processing branching param m = 0.9...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 13792.97it/s]
  Sens=0.000, Rel=0.013
Processing branching param m = 0.95...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 14373.21it/s]
  Sens=0.000, Rel=0.003
Processing branching param m = 0.98...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 15334.82it/s]
  Sens=0.000, Rel=0.000
Processing branching param m = 1.0...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 14897.30it/s]
  Sens=0.000, Rel=0.000
Processing branching param m = 1.02...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 13226.78it/s]
  Sens=0.000, Rel=0.001
Processing branching param m = 1.05...
Running for 1,000 iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 13792.29it/s]
  Sens=0.000, Rel=0.012

Experiment Complete. Data saved to processed/ directory.
(eib) PS C:\BrainpyEi\EI_Balanced_Criticality_Project> python C:\BrainpyEi\EI_Balanced_Criticality_Project\visualization\plot_response.py
C:\BrainpyEi\EI_Balanced_Criticality_Project\visualization\plot_response.py:41: SyntaxWarning: invalid escape sequence '\D'
  ax1.set_ylabel('Sensitivity ($\Delta r / r$)', color='#377eb8')
Generating Sensitivity & Reliability Plots...