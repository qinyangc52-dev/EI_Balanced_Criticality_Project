# %% Stage 1: Configuration - Single Source of Truth (SSOT)
# ============ SIMPLIFIED: Clean Version for Reproduction ============

import brainpy as bp
import brainpy.math as bm

# ============================================================================
# 1. Network Size & Connectivity
# ============================================================================
# 标准复现配置 (Paper Default)
N_TOTAL = 1000
N_E = 800
N_I = 200
CONN_PROB = 0.2  # p = K/N = 200/1000 = 0.2

# --- 如果你想跑 N=5000 (Robustness)，手动修改上面为：---
# N_TOTAL = 5000
# N_E = 4000
# N_I = 1000
# CONN_PROB = 0.04  # p = 200/5000 = 0.04 (保持 K=200 不变)

# ============================================================================
# 2. External Input (The "Throttle")
# ============================================================================
# 配合下方的 W_SCALE 使用。
# 如果发放率过低 (<1Hz)，调大这个值；如果过高 (>5Hz)，调小这个值。
EXT_FREQ_TOTAL = 2200.0 

# ============================================================================
# 3. Criticality Scan Range
# ============================================================================
# 扫描抑制衰减时间常数，寻找临界点
TAU_DECAY_I_LIST = [2.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0]

# 预设标记点 (分析用)
TAU_DECAY_I_SUBCRITICAL = 2.0
TAU_DECAY_I_CRITICAL = 8.0
TAU_DECAY_I_SUPERCRITICAL = 11.0

# ============================================================================
# 4. Neuron Parameters (LIF)
# ============================================================================
V_REST = -60.0   
V_RESET = -60.0  
V_TH = -50.0     

TAU_E = 20.0     
TAU_I = 10.0     
REF_E = 2.0      
REF_I = 1.0      

# ============================================================================
# 5. Synaptic Dynamics & Weights (The "Engine")
# ============================================================================
V_REV_E = 0.0    
V_REV_I = -70.0  
TAU_RISE = 0.5   
TAU_DECAY_E = 2.0  

# --- [核心修正] 全局权重缩放 ---
# 原因：仿真器积分产生的电荷量比论文公式大，必须缩小权重以防止癫痫。
W_SCALE = 0.6

# 应用缩放 (保持 E/I 比例完美平衡)
G_E_TO_E = 0.012 * W_SCALE   # 0.0072
G_E_TO_I = 0.024 * W_SCALE   # 0.0144
G_I_TO_E = 0.18  * W_SCALE   # 0.108  (强抑制)
G_I_TO_I = 0.31  * W_SCALE   # 0.186

# 外部输入权重也同步缩放
G_EXT_TO_E = 0.022 * W_SCALE 
G_EXT_TO_I = 0.040 * W_SCALE 

# Poisson Source Details
N_EXT = 160  
EXT_RATE = EXT_FREQ_TOTAL / float(N_EXT)

# --------------------------------------------------------------------------
# Aliases (兼容旧代码变量名)
# --------------------------------------------------------------------------
G_EE = G_E_TO_E
G_EI = G_E_TO_I 
G_IE = G_I_TO_E 
G_II = G_I_TO_I
G_EXT_E = G_EXT_TO_E
G_EXT_I = G_EXT_TO_I

# ============================================================================
# 6. Simulation Settings
# ============================================================================
DT = 0.05 
DURATION = 5000.0 
WARMUP = 500.0     

# Avalanche Analysis
AVALANCHE_BIN_MULTIPLIER = 1.0
AVALANCHE_SIZE_THRESHOLD = 50
POWERLAW_FIT_MIN_SIZE = 10

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FIGURE_DIR = "visualization/figures"

# Quick Test Switch
QUICK_TEST = True 
QUICK_TEST_DURATION = 200.0

# Random Seed
SEED = 42
bm.random.seed(SEED)