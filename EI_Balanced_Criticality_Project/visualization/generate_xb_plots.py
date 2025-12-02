# %% Visualization: Scaling Relation Check against Theory
# Plots <S>(T) vs T and compares with predicted slope gamma = (alpha-1)/(tau-1)
# Usage:
#   python visualization/plot_scaling_theory.py --taus 8.0 8.2
#   python visualization/plot_scaling_theory.py (defaults to 2.0 8.0 11.0)

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import PROCESSED_DATA_DIR
from utils.io_manager import load_pkl

# ==========================================
# Configuration & Style
# ==========================================
OUTPUT_DIR = Path(__file__).parent / "xb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6)

# Colors matching the distribution plot
COLORS = ['#2b8cbe', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
STYLES = ['o', 's', '^', 'D']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.5,
    'legend.frameon': False,
})

def calculate_scaling(sizes, durs):
    """Compute average size <S> for each duration T."""
    unique_durs = np.unique(durs)
    mean_sizes = []
    valid_durs = []
    
    for d in unique_durs:
        # Filter: require at least 5 samples to compute a reliable mean
        idx = (durs == d)
        if np.sum(idx) >= 5:
            mean_sizes.append(np.mean(sizes[idx]))
            valid_durs.append(d)
            
    return np.array(valid_durs), np.array(mean_sizes)

def plot_scaling_theory(taus):
    print(f"\n{'='*60}")
    print(f"Generating Scaling Relation vs Theory Plots")
    print(f"Target Taus: {taus} ms")
    print(f"{'='*60}\n")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    found_data = False
    
    for i, tau in enumerate(taus):
        file_path = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
        
        if not file_path.exists():
            print(f"Warning: Data for tau={tau} not found. Skipping.")
            continue
            
        try:
            stats = load_pkl(file_path)
            # Use RAW data
            sizes = stats['preprocessing']['avalanche_sizes']
            durs = stats['preprocessing']['avalanche_durations']
            
            # Get Fitted Exponents (Tau and Alpha)
            fit_tau = stats.get('size_exponent')
            fit_alpha = stats.get('duration_exponent')
            
            found_data = True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # 1. Calculate Data Curve: <S> vs T
        T_data, S_data = calculate_scaling(sizes, durs)
        
        if len(T_data) < 3:
            print(f"  tau={tau}: Not enough data points for scaling.")
            continue

        color = COLORS[i % len(COLORS)]
        marker = STYLES[i % len(STYLES)]
        
        # Plot Data Points
        label_data = f"Sim: $\\tau_I^d={tau}$ ms"
        ax.loglog(T_data, S_data, marker=marker, linestyle='', color=color, 
                  markersize=7, alpha=0.7, label=label_data)
        
        # 2. Plot Theoretical Line
        # Gamma = (alpha - 1) / (tau - 1)
        if fit_tau and fit_alpha and fit_tau > 1:
            gamma_pred = (fit_alpha - 1) / (fit_tau - 1)
            
            # Anchor the theory line to the middle of the data
            mid_idx = len(T_data) // 2
            mid_T = T_data[mid_idx]
            mid_S = S_data[mid_idx]
            
            # Generate line coordinates
            x_line = np.logspace(np.log10(min(T_data)), np.log10(max(T_data)), 100)
            y_line = mid_S * (x_line / mid_T) ** gamma_pred
            
            # Plot Line
            label_theory = f"Theory: $\\gamma={gamma_pred:.2f}$"
            ax.loglog(x_line, y_line, linestyle='--', color=color, 
                      linewidth=2.5, alpha=0.9, label=label_theory)
            
            print(f"  tau={tau}: Predicted Slope gamma = {gamma_pred:.3f}")
        else:
            print(f"  tau={tau}: Cannot compute theoretical slope (invalid exponents)")

    if not found_data:
        print("No valid data found.")
        return

    # Styling
    ax.set_title("Scaling Relation Verification: $\\langle S \\rangle \\sim T^{\\gamma}$", 
                 fontweight='bold', pad=15)
    ax.set_xlabel("Duration $T$ (ms)", fontsize=16)
    ax.set_ylabel("Average Size $\\langle S \\rangle$", fontsize=16)
    
    # Grid
    ax.grid(True, which="both", ls="-", alpha=0.15)
    
    # Smart Legend: Put it outside if too crowded
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
    
    # Save
    save_name = f"Scaling_Theory_Compare_{'_'.join(map(str, taus))}.png"
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Scaling Relation Check.")
    
    parser.add_argument(
        '--taus', 
        type=float, 
        nargs='+', 
        default=[2.0, 8.0, 11.0],
        help='List of tau values to plot. Default: 2.0 8.0 11.0'
    )
    
    args = parser.parse_args()
    
    plot_scaling_theory(taus=args.taus)