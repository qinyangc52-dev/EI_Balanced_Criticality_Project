# %% Visualization: Scaling Relation (Replicating Paper Fig S2.C)
# Plots Average Avalanche Size <S> vs Duration T for different regimes
# Comparisons with Theoretical Predictions (Crackling Noise Relation)

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import FIGURE_DIR, PROCESSED_DATA_DIR
from utils.io_manager import load_pkl

# ==========================================
# Configuration
# ==========================================
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

# Define the three regimes to plot (based on your simulation results)
REGIMES = [
    {'tau': 2.0,  'label': 'Subcritical (2ms)',   'color': '#66c2a5'}, # Teal
    {'tau': 8.0,  'label': 'Critical (8ms)',      'color': '#fc8d62'}, # Orange
    {'tau': 11.0, 'label': 'Supercritical (11ms)','color': '#8da0cb'}  # Purple
]

def load_data(tau):
    """Helper to load processed stats."""
    stat_path = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
    stats = load_pkl(stat_path) if stat_path.exists() else None
    return stats

def plot_scaling_relation_multipanel():
    print("Plotting Scaling Relation for multiple regimes (Fig S2.C style)...")
    
    # Setup Figure: 3 Vertical Subplots sharing X axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.1) # Reduce space between plots

    for i, regime in enumerate(REGIMES):
        tau_val = regime['tau']
        ax = axes[i]
        
        stats = load_data(tau_val)
        if not stats:
            print(f"Warning: Data for tau={tau_val} not found. Skipping.")
            continue

        # 1. Extract Data
        raw_sizes = stats['preprocessing']['avalanche_sizes']
        raw_durs = stats['preprocessing']['avalanche_durations']
        
        # Get fitted exponents for this specific regime
        fit_tau = stats.get('size_exponent')
        fit_alpha = stats.get('duration_exponent')

        # 2. Compute Mean Size per Duration <S>(T)
        unique_durs = np.unique(raw_durs)
        mean_sizes = []
        valid_durs = []
        
        for d in unique_durs:
            sizes_in_dur = raw_sizes[raw_durs == d]
            if len(sizes_in_dur) >= 3: # Filter noise
                mean_sizes.append(np.mean(sizes_in_dur))
                valid_durs.append(d)
        
        valid_durs = np.array(valid_durs)
        mean_sizes = np.array(mean_sizes)

        # 3. Plot Experimental Data
        ax.loglog(valid_durs, mean_sizes, 'o', color=regime['color'], 
                  markersize=6, alpha=0.8, label=f"Sim: {regime['label']}")

        # 4. Plot Theoretical Prediction (Grey Line)
        # Relation: <S> ~ T^gamma, where gamma = (alpha - 1) / (tau - 1)
        if fit_tau and fit_alpha and fit_tau > 1:
            gamma = (fit_alpha - 1) / (fit_tau - 1)
            
            # Create a guide line passing through the geometric center of the data
            if len(valid_durs) > 0:
                mid_idx = len(valid_durs) // 2
                mid_x = valid_durs[mid_idx]
                mid_y = mean_sizes[mid_idx]
                
                # Generate line points
                x_line = np.logspace(np.log10(min(valid_durs)), np.log10(max(valid_durs)), 100)
                y_line = mid_y * (x_line / mid_x) ** gamma
                
                ax.plot(x_line, y_line, '-', color='gray', linewidth=2.5, alpha=0.7,
                        label=f'Theory $\gamma={gamma:.2f}$')
                
                # Add text annotation for the exponents
                text_str = f"Fit: $\\tau={fit_tau:.2f}, \\alpha={fit_alpha:.2f}$"
                ax.text(0.05, 0.8, text_str, transform=ax.transAxes, fontsize=11,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 5. Styling
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        
        # Y-axis label for the middle plot mainly, or all
        ax.set_ylabel(r"Avg Size $\langle S \rangle$")

    # Common X-axis label
    axes[-1].set_xlabel("Duration $T$ (time bins)")
    
    # Main Title
    fig.suptitle("Scaling Relation Across Regimes\nCrackling Noise Verification", 
                 fontweight='bold', y=0.92)

    # 6. Save
    save_path = Path(FIGURE_DIR) / "Scaling_Relation_MultiPanel.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    plot_scaling_relation_multipanel()