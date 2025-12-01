# %% Stage 1: Configuration - Balanced State (Strong Inhibition)
# STRATEGY: Boost I->E to prevent epilepsy (400Hz). 
#           Restore G_EXT to provide drive (prevent 0Hz).
#           This forces the network into the "Balanced Regime".

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
# 3. Criticality Scan Range
# ============================================================================
TAU_DECAY_I_LIST = [2.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]

TAU_DECAY_I_SUBCRITICAL = 2.0
TAU_DECAY_I_CRITICAL = 8.0
TAU_DECAY_I_SUPERCRITICAL = 12.0

# ============================================================================
# 4. Neuron Parameters
# ============================================================================
V_REST = -60.0   
V_RESET = -60.0  
V_TH = -50.0     
TAU_E = 20.0     
TAU_I = 10.0     
REF_E = 2.0      
REF_I = 1.0      

# ============================================================================
# 5. Synaptic Dynamics & Weights (BALANCED FIX)
# ============================================================================
V_REV_E = 0.0    
V_REV_I = -70.0  
TAU_RISE = 0.5   
TAU_DECAY_E = 2.0  

W_SCALE = 1.0

# --- Recurrent Weights ---
# G_EE: Paper 0.012. We keep it modest.
G_E_TO_E = 0.012   
G_E_TO_I = 0.024   

# G_IE: THE FIX. Boosted significantly to clamp runaway excitation.
# Previous attempts failed because this was too low (0.18).
# Increasing to 0.35 ensures inhibition dominates excitation.
G_I_TO_E = 0.35    
G_I_TO_I = 0.31    

# --- External Weights ---
# Restored to ~0.02 (Paper level) to ensure the network isn't silent.
# With strong inhibition, we can afford higher drive.
G_EXT_TO_E = 0.018
G_EXT_TO_I = 0.018 

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
# 6. Simulation Settings
# ============================================================================
DT = 0.05 
DURATION = 5000.0 
WARMUP = 500.0     

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