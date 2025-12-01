# %% Visualization: Generate INDEPENDENT plots for all results
# COMPLETE VERSION: Smart Raster Window + Distributions + KS Curve
# 1. Individual Raster Plots for EVERY tau (Centered on Peak Activity)
# 2. Avalanche Distribution Plot for Critical State
# 3. KS Distance Curve

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    FIGURE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, 
    TAU_DECAY_I_LIST, N_TOTAL  # Ensure N_TOTAL is imported
)
from utils.io_manager import load_pkl

# ==========================================
# Configuration
# ==========================================
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.8) # Large font for separate figures

# Colors
COLOR_RASTER = '#2c3e50' # Dark Blue Grey
COLOR_CRI = '#d95f02'    # Orange
COLOR_FIT = 'black'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

def load_data(tau):
    """Helper to load spikes and stats safely."""
    raw_path = Path(RAW_DATA_DIR) / f"spikes_{tau:.1f}.npz"
    stat_path = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
    
    spikes = None
    if raw_path.exists():
        try: spikes = np.load(raw_path)
        except: pass

    stats = None
    if stat_path.exists():
        try: stats = load_pkl(stat_path)
        except: pass
            
    return spikes, stats

# ==========================================
# 1. Function: Plot All Rasters Separately (Smart Window)
# ==========================================
def plot_all_rasters():
    print("\n--- Generating Independent Raster Plots (Smart Window) ---")
    plot_window = 2000.0 # 2 seconds window
    
    existing_files = sorted(Path(RAW_DATA_DIR).glob("spikes_*.npz"))
    
    if not existing_files:
        print("No spike files found!")
        return

    for file_path in existing_files:
        try:
            tau_val = float(file_path.stem.split('_')[1])
        except:
            continue
            
        try:
            data = np.load(file_path)
            ts = data['spike_times']
            ns = data['neuron_indices']
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if len(ts) > 100: # Only plot if there's meaningful activity
            # --- Smart Window Algorithm ---
            # 1. Histogram (100ms bins)
            bins = np.arange(ts.min(), ts.max(), 100.0)
            if len(bins) > 1:
                hist, bin_edges = np.histogram(ts, bins=bins)
                
                # 2. Find Peak
                peak_idx = np.argmax(hist)
                t_peak = bin_edges[peak_idx]
                
                # 3. Center Window
                t_start = max(ts.min(), t_peak - plot_window / 2)
                t_end = t_start + plot_window
                
                # Boundary check
                if t_end > ts.max():
                    t_end = ts.max()
                    t_start = t_end - plot_window

                # 4. Slice Data
                mask = (ts >= t_start) & (ts <= t_end)
                ts_view = ts[mask]
                ns_view = ns[mask]
                
                # 5. Plot (Zero-centered time: t - t_start)
                ax.scatter(ts_view - t_start, ns_view, s=3.0, c=COLOR_RASTER, alpha=0.8, edgecolors='none')
                ax.set_xlim(0, plot_window)
                ax.text(0.02, 0.95, f"Peak Activity @ {t_peak/1000:.1f}s", transform=ax.transAxes,
                        fontsize=14, color='red', fontweight='bold')
            else:
                ax.scatter(ts, ns, s=3.0, c=COLOR_RASTER)
                
        else:
            ax.text(0.5, 0.5, "Network Silent (Rate < 0.01Hz)", ha='center', va='center', fontsize=20)

        ax.set_ylim(0, N_TOTAL)
        ax.set_title(f"Raster Plot ($\\tau_I^d = {tau_val:.1f}$ ms)", fontweight='bold', pad=15)
        ax.set_xlabel('Time Window (ms) [Centered on Peak Activity]')
        ax.set_ylabel('Neuron Index')
        
        # Save
        Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
        save_name = f"Raster_tau_{tau_val:.1f}.png"
        plt.savefig(Path(FIGURE_DIR) / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

# ==========================================
# 2. Function: Plot Distribution (Critical)
# ==========================================
def plot_critical_distribution(tau_crit=9.0):
    print(f"\n--- Generating Critical Distribution Plot (tau={tau_crit}) ---")
    _, stats = load_data(tau_crit)
    
    if not stats or 'distribution' not in stats:
        print(f"Stats for tau={tau_crit} not found. Skipping.")
        return

    dist = stats['distribution']
    tau_exp = stats.get('size_exponent')
    alpha_exp = stats.get('duration_exponent')

    fig, ax = plt.subplots(figsize=(10, 10)) # Square figure
    
    # --- Size ---
    s_vals = dist['size_values']
    s_probs = dist['size_probs']
    mask_s = (s_vals > 0) & (s_probs > 0)
    s_x, s_y = s_vals[mask_s], s_probs[mask_s]
    
    ax.loglog(s_x, s_y, 'o', color=COLOR_CRI, markersize=10, alpha=0.7, label='Size $P(S)$')
    
    # Fit line Size
    if tau_exp:
        mid = len(s_x)//2
        y_fit = s_y[mid] * (s_x / s_x[mid])**(-tau_exp)
        ax.loglog(s_x, y_fit, '--', color='black', linewidth=3, label=f'Fit $\\tau={tau_exp:.2f}$')

    # --- Duration (Shifted) ---
    t_vals = dist['duration_values']
    t_probs = dist['duration_probs']
    mask_t = (t_vals > 0) & (t_probs > 0)
    t_x, t_y = t_vals[mask_t], t_probs[mask_t] * 100 # Shift up
    
    ax.loglog(t_x, t_y, 's', color='gray', markersize=10, alpha=0.6, label='Duration $P(T) \\times 100$')
    
    # Fit line Duration
    if alpha_exp:
        mid = len(t_x)//2
        y_fit = t_y[mid] * (t_x / t_x[mid])**(-alpha_exp)
        ax.loglog(t_x, y_fit, '--', color='gray', linewidth=3, label=f'Fit $\\alpha={alpha_exp:.2f}$')

    ax.set_title(f"Critical Avalanches ($\\tau_I^d={tau_crit}$ms)", fontweight='bold', pad=20)
    ax.set_xlabel("Size (S) or Duration (T)")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    
    save_name = "Avalanche_Distribution_Critical.png"
    plt.savefig(Path(FIGURE_DIR) / save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}")

# ==========================================
# 3. Function: Plot KS Distance Curve
# ==========================================
def plot_ks_curve():
    print("\n--- Generating KS Distance Curve ---")
    
    taus = []
    ks_vals = []
    
    # Iterate through all processed stats
    files = sorted(Path(PROCESSED_DATA_DIR).glob("avalanche_stats_*.pkl"))
    for f in files:
        try:
            stats = load_pkl(f)
            t = stats['tau_d_I']
            k = stats.get('size_ks')
            if k is not None and not np.isnan(k):
                taus.append(t)
                ks_vals.append(k)
        except:
            continue
            
    if not taus:
        print("No KS data found.")
        return

    # Sort by tau
    taus, ks_vals = zip(*sorted(zip(taus, ks_vals)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(taus, ks_vals, '-', color=COLOR_CRI, linewidth=4, alpha=0.6)
    ax.plot(taus, ks_vals, 'o', color=COLOR_CRI, markersize=12, markeredgecolor='white', markeredgewidth=2, label='KS Distance')
    
    # Highlight min
    min_idx = np.argmin(ks_vals)
    best_tau = taus[min_idx]
    
    ax.axvline(best_tau, color='gray', linestyle='--', linewidth=3)
    ax.annotate(f'Critical Point\n{best_tau} ms', 
                xy=(best_tau, ks_vals[min_idx]), 
                xytext=(best_tau, max(ks_vals)*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', fontweight='bold', fontsize=16)

    ax.set_title("Criticality Identification (KS Test)", fontweight='bold', pad=20)
    ax.set_xlabel(r"Inhibitory Time Constant $\tau_I^d$ (ms)")
    ax.set_ylabel("KS Distance (Lower is Better)")
    ax.set_ylim(0, max(ks_vals)*1.1)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    save_name = "KS_Distance_Curve.png"
    plt.savefig(Path(FIGURE_DIR) / save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}")

if __name__ == "__main__":
    # Execute all plotting functions
    plot_all_rasters()
    # 9.0 is your best critical point from logs
    plot_critical_distribution(tau_crit=9.0) 
    plot_ks_curve()