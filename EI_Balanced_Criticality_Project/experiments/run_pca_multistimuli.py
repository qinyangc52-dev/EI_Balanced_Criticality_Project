# %% Experiment: Neural Representation PCA Analysis
# FIXED: bp.share access and JIT compatibility
# Replicates Figure 3(c): PCA projection of avalanches under different inputs

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm
from sklearn.decomposition import PCA 

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    N_E, N_I, N_EXT, EXT_RATE, DT, 
    G_EXT_E, G_EXT_I, TAU_DECAY_E, V_REV_E, TAU_RISE,
    FIGURE_DIR
)
from models.network import BalancedNetwork

# ==========================================
# 1. Custom Component: Frozen Input
# ==========================================
class FrozenPoissonInput(bp.DynamicalSystem):
    """Injects a pre-defined (frozen) spike train."""
    def __init__(self, network, frozen_spikes_E, frozen_spikes_I):
        super().__init__()
        self.network = network
        # frozen_spikes shape: (TimeSteps, N_neurons)
        self.spikes_E = bm.asarray(frozen_spikes_E, dtype=float)
        self.spikes_I = bm.asarray(frozen_spikes_I, dtype=float)
        
        # Synaptic parameters
        self.tau_d = TAU_DECAY_E
        self.tau_r = TAU_RISE
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        # Variables
        self.h_E = bm.Variable(bm.zeros(N_E))
        self.g_E = bm.Variable(bm.zeros(N_E))
        self.h_I = bm.Variable(bm.zeros(N_I))
        self.g_I = bm.Variable(bm.zeros(N_I))

    def update(self):
        # FIX: Use dictionary access for shared variables
        t_idx = bp.share['i']
        dt = bp.share['dt']
        
        # FIX: Removed Python 'if' check to support JIT compilation.
        # Ensure inputs are generated with sufficient length (steps >= duration/dt).
        # BrainPy's DSRunner guarantees t_idx is within range if duration matches.
        ext_E = self.spikes_E[t_idx]
        ext_I = self.spikes_I[t_idx]

        # Update Conductance (Double Exponential)
        self.h_E.value = self.h_E + (-self.h_E / self.tau_r * dt + ext_E)
        self.g_E.value = self.g_E + (-self.g_E / self.tau_d * dt + self.h_E)
        
        self.h_I.value = self.h_I + (-self.h_I / self.tau_r * dt + ext_I)
        self.g_I.value = self.g_I + (-self.g_I / self.tau_d * dt + self.h_I)
        
        # Current
        I_ext_E = self.g_E * G_EXT_E * self.norm * (V_REV_E - self.network.E.V)
        I_ext_I = self.g_I * G_EXT_I * self.norm * (V_REV_E - self.network.I.V)
        
        self.network.E.input.value += I_ext_E
        self.network.I.input.value += I_ext_I

# ==========================================
# 2. Main Experiment Logic
# ==========================================
def run_pca_experiment():
    # --- Settings ---
    tau_crit = 9.0       # Critical state parameter
    n_signals = 10       # 10 distinct signals
    n_trials = 20        # 20 trials per signal
    duration = 1000.0    # 1 second per trial
    
    # Calculate exact steps to match runner's execution
    steps = int(np.round(duration / DT))
    
    print(f"Generating {n_signals} frozen signals ({steps} steps each)...")
    
    # --- Generate Frozen Signals ---
    signals_E = []
    signals_I = []
    
    for i in range(n_signals):
        # Generate random Poisson spikes
        prob = EXT_RATE * DT / 1000.0 * N_EXT
        # Generate boolean mask then convert to float
        # Note: We generate exactly 'steps' length to match runner iteration
        spikes_E = np.random.rand(steps, N_E) < prob
        spikes_I = np.random.rand(steps, N_I) < prob
        signals_E.append(spikes_E)
        signals_I.append(spikes_I)

    # --- Run Simulations ---
    print(f"Running {n_signals * n_trials} trials...")
    
    # Store population vectors: (Total_Trials, N_E)
    activity_vectors = []
    labels = [] # To store signal ID (0-9)
    
    for sig_id in range(n_signals):
        print(f"  Processing Signal {sig_id+1}/{n_signals}...", end="", flush=True)
        
        # Prepare input for this signal
        current_input_E = signals_E[sig_id]
        current_input_I = signals_I[sig_id]
        
        for trial in range(n_trials):
            # Clean up
            bm.clear_name_cache()
            
            # Init Network
            net = BalancedNetwork(tau_d_I=tau_crit)
            # Use our custom frozen input
            inp = FrozenPoissonInput(net, current_input_E, current_input_I)
            
            # Runner
            runner = bp.DSRunner(
                bp.DynSysGroup(net=net, inp=inp),
                monitors={'spikes': net.E.spike},
                dt=DT,
                progress_bar=False
            )
            
            runner.run(duration)
            
            # Collect Activity Vector
            # We count total spikes per neuron during the stimulus (Rate Vector)
            spikes = runner.mon['spikes'] # (Time, N)
            rate_vector = np.sum(spikes, axis=0) # (N,)
            
            activity_vectors.append(rate_vector)
            labels.append(sig_id)
            
        print(" Done.")

    # --- PCA Analysis ---
    print("Performing PCA...")
    X = np.array(activity_vectors) # (200, 800)
    
    # Check if we have enough activity
    if np.sum(X) == 0:
        print("Warning: Network was completely silent. Cannot perform PCA.")
        return

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X) # (200, 3)
    
    # --- Plotting ---
    print("Plotting 3D Scatter...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_signals))
    
    for sig_id in range(n_signals):
        # Select points belonging to this signal
        mask = np.array(labels) == sig_id
        points = X_pca[mask]
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   color=colors[sig_id], s=40, alpha=0.8, 
                   label=f'Signal {sig_id+1}')
        
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title(f'Neural Representation PCA ($\\tau_I^d={tau_crit}$ms)', fontsize=15, fontweight='bold')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = Path(FIGURE_DIR) / "PCA_Representation_3D.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    run_pca_experiment()