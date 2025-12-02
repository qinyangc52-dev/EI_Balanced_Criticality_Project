# %% Stage 2: Model Components - LIF Neurons with Refractory Period
# RESTORED: Standard LIF implementation (Input divided by tau)  # 注释：恢复标准 LIF 实现（输入除以 tau），对应论文 Appendix A 中的 conductance-based LIF 方程

import brainpy as bp  # 导入 BrainPy 框架，用于动态系统模拟
import brainpy.math as bm  # 导入 BrainPy 的数学模块，用于变量管理和向量化操作
from configs.model_config import V_REST, V_RESET, V_TH, TAU_E, TAU_I, REF_E, REF_I  # 从配置模块导入全局参数（SSOT）：V_REST (-70 mV), V_RESET (-60 mV), V_TH (-50 mV), TAU_E (20 ms), TAU_I (10 ms), REF_E (2 ms), REF_I (1 ms)

class LifRefE(bp.DynamicalSystem):  # 定义兴奋性 LIF 神经元类，继承 BrainPy 的动态系统，用于模拟兴奋性神经元群
    def __init__(self, size, name=None):  # 初始化方法：size 是神经元数量，name 是可选名称
        super().__init__(name=name)  # 调用父类初始化
        self.V_rest = V_REST  # 静息电位 (V_rest = -70 mV，论文 Appendix A)
        self.V_reset = V_RESET  # 重置电位 (V_reset = -60 mV，尖峰后重置)
        self.V_th = V_TH  # 阈值电位 (V_th = -50 mV，超过即尖峰)
        self.tau = TAU_E  # 膜时间常数 (τ_E = 20 ms，兴奋性神经元)
        self.t_ref = REF_E  # 绝对不应期 (2 ms，尖峰后不响应)
        self.num = size  # 神经元群规模
        self.V = bm.Variable(bm.ones(size) * V_REST)  # 膜电位变量 (初始为 V_rest，使用 bm.Variable 支持自动微分)
        self.input = bm.Variable(bm.zeros(size))  # 输入电流/电导 (初始 0，累积 Irec + Ii)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))  # 尖峰标志 (bool，初始 False)
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)  # 上次尖峰时间 (初始远古值，避免初始不应期)
    
    def update(self):  # 更新方法：在每个时间步调用，模拟 LIF 积分和尖峰
        t = bp.share['t']  # 获取当前模拟时间 t
        dt = bp.share['dt']  # 获取时间步长 dt
        refractory = (t - self.t_last_spike) <= self.t_ref  # 计算不应期掩码：如果 t - t_last_spike <= t_ref，则为 True（不应期内）
        # Standard LIF: Input current is divided by tau  # 标准 LIF 方程：dV/dt = (V_rest - V + input) / tau (论文公式简化，无噪声)
        dV = ((self.V_rest - self.V) + self.input) / self.tau * dt  # 使用 Euler 方法计算 ΔV
        V_new = bm.where(refractory, self.V, self.V + dV)  # 如果不应期内，V 不变；否则更新 V
        spike = V_new >= self.V_th  # 检测尖峰：V_new >= V_th
        self.V.value = bm.where(spike, self.V_reset, V_new)  # 如果尖峰，重置为 V_reset；否则保持 V_new
        self.spike.value = spike  # 更新尖峰标志
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)  # 如果尖峰，更新上次尖峰时间为 t
        self.input[:] = 0.0  # 清零输入（为下一个时间步准备，input 是瞬时累积）

class LifRefI(bp.DynamicalSystem):  # 定义抑制性 LIF 神经元类，类似 LifRefE，但使用抑制性参数
    def __init__(self, size, name=None):  # 初始化方法：类似 LifRefE
        super().__init__(name=name)
        self.V_rest = V_REST  # 静息电位 (同上)
        self.V_reset = V_RESET  # 重置电位 (同上)
        self.V_th = V_TH  # 阈值电位 (同上)
        self.tau = TAU_I  # 膜时间常数 (τ_I = 10 ms，抑制性神经元，论文 Appendix A)
        self.t_ref = REF_I  # 绝对不应期 (1 ms，抑制性)
        self.num = size  # 神经元群规模
        self.V = bm.Variable(bm.ones(size) * V_REST)  # 膜电位变量 (初始 V_rest)
        self.input = bm.Variable(bm.zeros(size))  # 输入电流/电导 (初始 0)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))  # 尖峰标志
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)  # 上次尖峰时间
    
    def update(self):  # 更新方法：与 LifRefE 完全相同，仅参数不同
        t = bp.share['t']
        dt = bp.share['dt']
        refractory = (t - self.t_last_spike) <= self.t_ref
        dV = ((self.V_rest - self.V) + self.input) / self.tau * dt
        V_new = bm.where(refractory, self.V, self.V + dV)
        spike = V_new >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V_new)
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.input[:] = 0.0