# %% Generate GS Raster Plots (Flexible Version)
# Usage: 
#   python visualization/generate_gs_rasters.py --taus 2.0 3.0 8.0
#   python visualization/generate_gs_rasters.py (Generates ALL found)

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import RAW_DATA_DIR, N_TOTAL

# ==========================================
# Configuration
# ==========================================
OUTPUT_DIR = Path(__file__).parent / "gs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style Settings
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6)
COLOR_RASTER = '#2c3e50' 

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

def generate_rasters(target_taus=None, window_size=1000.0):
    print(f"\n{'='*60}")
    print(f"Generating Raster Plots for 'gs' folder")
    if target_taus:
        print(f"Target Taus: {target_taus} ms")
    else:
        print(f"Target Taus: ALL available in data/raw")
    print(f"Time Window: {window_size} ms")
    print(f"{'='*60}\n")

    # 1. Get all spike files
    all_files = sorted(Path(RAW_DATA_DIR).glob("spikes_*.npz"))
    
    if not all_files:
        print(f"Error: No data found in {RAW_DATA_DIR}")
        return

    count = 0
    for f in all_files:
        try:
            # Parse tau from filename "spikes_8.0.npz"
            tau_str = f.stem.split('_')[1]
            tau_val = float(tau_str)
        except:
            continue
            
        # 2. Check if this tau is requested
        # If target_taus is provided, skip if not in list
        if target_taus is not None:
            # Use small epsilon for float comparison
            if not any(np.isclose(tau_val, t, atol=0.01) for t in target_taus):
                continue
            
        print(f"Processing tau = {tau_val} ms...", end="")
        
        try:
            data = np.load(f)
            ts = data['spike_times']
            ns = data['neuron_indices']
        except Exception as e:
            print(f" [Error loading: {e}]")
            continue
            
        # 3. Setup Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 4. Smart Window Logic
        plot_window = float(window_size)
        
        if len(ts) > 100:
            # Auto-center on peak activity
            bins = np.arange(ts.min(), ts.max(), 100.0)
            if len(bins) > 1:
                hist, bin_edges = np.histogram(ts, bins=bins)
                peak_idx = np.argmax(hist)
                t_peak = bin_edges[peak_idx]
                
                t_start = max(ts.min(), t_peak - plot_window/2)
                t_end = t_start + plot_window
                
                if t_end > ts.max():
                    t_end = ts.max()
                    t_start = max(ts.min(), t_end - plot_window)
                    
                mask = (ts >= t_start) & (ts <= t_end)
                ts_view = ts[mask]
                ns_view = ns[mask]
                
                # Plot (centered relative time)
                ax.scatter(ts_view - t_start, ns_view, s=4.0, c=COLOR_RASTER, alpha=0.9, edgecolors='none')
                ax.set_xlim(0, plot_window)
            else:
                ax.scatter(ts, ns, s=4.0, c=COLOR_RASTER)
        else:
            ax.text(0.5, 0.5, "Network Silent", ha='center', va='center', fontsize=15, color='gray')

        # 5. Styling
        ax.set_ylim(0, N_TOTAL)
        ax.set_title(f"Inhibitory Decay $\\tau_I^d = {tau_val}$ ms", fontweight='bold', fontsize=18, pad=15)
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Neuron Index", fontsize=16)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # 6. Save
        save_name = f"Raster_tau_{tau_val}.png"
        save_path = OUTPUT_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Saved.")
        count += 1

    print(f"\nDone! {count} images saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Raster Plots for specific tau values.")
    
    # Optional arguments
    parser.add_argument(
        '--taus', 
        type=float, 
        nargs='+', 
        help='List of tau values to plot (e.g. 2.0 3.0 8.0). If omitted, plots ALL found.'
    )
    
    parser.add_argument(
        '--window', 
        type=float, 
        default=1000.0, 
        help='Time window size in ms (default: 1000.0)'
    )
    
    args = parser.parse_args()
    
    generate_rasters(target_taus=args.taus, window_size=args.window)