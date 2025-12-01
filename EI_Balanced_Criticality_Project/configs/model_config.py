# %% Stage 1: Configuration - Balanced State (Long Simulation for Fig. 1)
# STRATEGY: Increase DURATION to 200s to capture rare large avalanches (Power-law tail).

import brainpy as bp
import brainpy.math as bm
import os

# ============================================================================
# 1. Network Size
# ============================================================================
N_TOTAL = 1000    
N_E = 800
N_I = 200
CONN_PROB = 0.2   

# ============================================================================
# 2. External Input
# ============================================================================
EXT_FREQ_TOTAL = 4000.0 

# ============================================================================
# 3. Criticality Scan Range (Optimized for Long Simulation)
# ============================================================================
# 为了复现 Fig 1，我们主要关注临界点(8.0)及其对比组。
# 跑 200秒比较久，所以只选几个关键点。
TAU_DECAY_I_LIST = [
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,   # 亚临界 (Subcritical) - 对照组
    6.0, 
    7.0,
    7.5,  # 接近临界
    8.0,   # [核心] 理论临界点 (Critical) - 复现 Fig 1e 必须跑这个
    8.2, 
    9.0,
    10.0,  # 验证临界点偏移
    11.0   # 超临界 (Supercritical) - 对照组
]

TAU_DECAY_I_SUBCRITICAL = 2.0
TAU_DECAY_I_CRITICAL = 8.0
TAU_DECAY_I_SUPERCRITICAL = 11.0

# ============================================================================
# 4. Neuron Parameters (Fixed: V_REST = -70.0)
# ============================================================================
V_REST = -70.0   # 修正为文献值 (之前是 -60)
V_RESET = -60.0  
V_TH = -50.0     
TAU_E = 20.0     
TAU_I = 10.0     
REF_E = 2.0      
REF_I = 1.0      

# ============================================================================
# 5. Synaptic Dynamics & Weights (Corrected Normalization)
# ============================================================================
V_REV_E = 0.0    
V_REV_I = -70.0  
TAU_RISE = 0.5   
TAU_DECAY_E = 2.0  

# --- Recurrent Weights ---
G_E_TO_E = 0.012   
G_E_TO_I = 0.024   
G_I_TO_E = 0.18    # 文献原值 (配合正确的归一化)
G_I_TO_I = 0.31    

# --- External Weights ---
G_EXT_TO_E = 0.022 
G_EXT_TO_I = 0.04  

# Derived
N_EXT = int(N_E * CONN_PROB)
EXT_RATE = EXT_FREQ_TOTAL / N_EXT 

# Aliases
G_EE = G_E_TO_E
G_EI = G_E_TO_I 
G_IE = G_I_TO_E 
G_II = G_I_TO_I
G_EXT_E = G_EXT_TO_E
G_EXT_I = G_EXT_TO_I

# ============================================================================
# 6. Simulation Settings (UPDATED FOR STEP 1)
# ============================================================================
DT = 0.05 

# === 核心修改: 增加仿真时长 ===
# 200,000 ms = 200 秒
# 预计生成约 100,000 个雪崩事件，足以画出完美的幂律长尾
DURATION = 200000.0 

# 热身时间 (Warmup)
# 让网络先跑 1秒钟达到稳态，这部分数据会被丢弃
WARMUP = 1000.0     

# Analysis Config
AVALANCHE_BIN_MULTIPLIER = 1.0 
AVALANCHE_SIZE_THRESHOLD = 0
POWERLAW_FIT_MIN_SIZE = 1

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURE_DIR = os.path.join(BASE_DIR, "visualization", "figures")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

QUICK_TEST = False 
SEED = 42
bm.random.seed(SEED)