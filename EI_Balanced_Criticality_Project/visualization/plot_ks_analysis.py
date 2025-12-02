# %% Visualization: KS Distance Analysis & Criticality Report
# Goal: 1. Identify critical point via KS distance minimization.
#       2. Print a comprehensive assessment table (Theory vs Sim) for ALL tau values.
# Output: 'ks' folder (visualization/ks) and Console Report

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import PROCESSED_DATA_DIR, TAU_DECAY_I_LIST
from utils.io_manager import load_pkl

# ==========================================
# Configuration & Style
# ==========================================
OUTPUT_DIR = Path(__file__).parent / "ks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6)

# Colors: Distinct colors for Size and Duration
COLOR_SIZE = '#d95f02'    # Orange
COLOR_DUR = '#7570b3'     # Purple
COLOR_GUIDE = '#636363'   # Dark Grey

# Reference Values for Table
REF_MF = {'tau': 1.50, 'alpha': 2.00}
REF_PAPER = {'tau': 2.26, 'alpha': 2.54}

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

# ==========================================
# Helper Functions for Table Calculation
# ==========================================
def calculate_scaling_slope(sizes, durs):
    """Compute slope of <S> vs T from raw data."""
    if len(sizes) == 0 or len(durs) == 0: return None
    
    unique_durs = np.unique(durs)
    mean_sizes = []
    valid_durs = []
    
    for d in unique_durs:
        idx = (durs == d)
        if np.sum(idx) >= 5: # Min samples filter
            mean_sizes.append(np.mean(sizes[idx]))
            valid_durs.append(d)
            
    if len(valid_durs) < 3: return None
    
    # Linear fit in log-log space
    log_x = np.log10(valid_durs)
    log_y = np.log10(mean_sizes)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

def plot_ks_minimization():
    print(f"\n{'='*100}")
    print(f"CRITICALITY ASSESSMENT REPORT (ALL TAU VALUES)")
    print(f"Reference 1 (Classic MF): tau={REF_MF['tau']}, alpha={REF_MF['alpha']}")
    print(f"Reference 2 (Paper BN): tau={REF_PAPER['tau']}, alpha={REF_PAPER['alpha']}")
    print(f"{'='*100}\n")
    
    # Table Header
    print(f"{'Tau_d':<8} | {'Tau (Size)':<12} | {'Alpha (Dur)':<12} | {'KS Dist':<10} | {'Rel. Err':<10} | {'Scaling':<10} | {'Status'}")
    print("-" * 100)

    # Data containers for plotting
    plot_taus = []
    plot_ks_size = []
    plot_ks_dur = []

    # 1. Load Data & Generate Report
    # Use the full list from config as requested
    sorted_taus = sorted(TAU_DECAY_I_LIST)
    
    # Store metrics to find global minimum later
    all_metrics = []

    for tau in sorted_taus:
        file_path = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
        
        # Default values for table
        tau_str = "N/A"
        alpha_str = "N/A"
        ks_str = "N/A"
        err_str = "N/A"
        scale_stat = "N/A"
        status = ""
        
        if file_path.exists():
            try:
                stats = load_pkl(file_path)
                
                # Extract Metrics
                fit_tau = stats.get('size_exponent')
                fit_alpha = stats.get('duration_exponent')
                ks_s = stats.get('size_ks')
                ks_d = stats.get('duration_ks')
                
                # Check Validity
                if ks_s is not None and ks_d is not None:
                    # Update Plot Data
                    plot_taus.append(tau)
                    plot_ks_size.append(ks_s)
                    plot_ks_dur.append(ks_d)
                    
                    # Store for best fit check
                    all_metrics.append({'tau': tau, 'ks': ks_s})

                    # Format Table Strings
                    tau_str = f"{fit_tau:.3f}" if fit_tau else "N/A"
                    alpha_str = f"{fit_alpha:.3f}" if fit_alpha else "N/A"
                    ks_str = f"{ks_s:.4f}"
                    
                    # Calculate Crackling Noise Relation Error
                    if fit_tau and fit_alpha and fit_tau > 1:
                        # Theory
                        gamma_theory = (fit_alpha - 1) / (fit_tau - 1)
                        # Simulation
                        sizes = stats['preprocessing']['avalanche_sizes']
                        durs = stats['preprocessing']['avalanche_durations']
                        gamma_sim = calculate_scaling_slope(sizes, durs)
                        
                        if gamma_sim:
                            error = abs(gamma_sim - gamma_theory)
                            err_str = f"{error:.3f}"
                            scale_stat = "Match" if error < 0.1 else "Fail"
                            if error < 0.05: status = "[GOOD REL]"

            except Exception as e:
                status = f"Error: {str(e)[:10]}..."

        # Identify Best Fit (Local Logic, global check later)
        # We will mark the global best fit after the loop if needed, 
        # but for streaming output, we just print the data.
        
        print(f"{tau:<8.1f} | {tau_str:<12} | {alpha_str:<12} | {ks_str:<10} | {err_str:<10} | {scale_stat:<10} | {status}")

    if not plot_taus:
        print("\nError: No valid statistics found. Please run simulations and analysis first.")
        return

    # Identify Global Best Fit
    min_ks = min(plot_ks_size)
    best_tau_idx = np.argmin(plot_ks_size)
    best_tau = plot_taus[best_tau_idx]
    
    print("-" * 100)
    print(f"CONCLUSION: Global Minimum KS Distance found at tau = {best_tau} ms (KS={min_ks:.4f})")
    print("\n")

    # ==========================================
    # 2. Plotting (KS Curve)
    # ==========================================
    print(f"Generating Plot in {OUTPUT_DIR}...")
    
    # Convert to arrays
    p_taus = np.array(plot_taus)
    p_ks_s = np.array(plot_ks_size)
    p_ks_d = np.array(plot_ks_dur)

    # Find Minima indices
    min_idx_s = np.argmin(p_ks_s)
    min_idx_d = np.argmin(p_ks_d)
    
    best_tau_s = p_taus[min_idx_s]
    best_tau_d = p_taus[min_idx_d]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Curves
    ax.plot(p_taus, p_ks_s, 'o-', color=COLOR_SIZE, linewidth=2.5, markersize=8, 
            label='Avalanche Size ($S$)')
    ax.plot(p_taus, p_ks_d, 's-', color=COLOR_DUR, linewidth=2.5, markersize=8, 
            label='Avalanche Duration ($T$)')

    # Highlight Minima
    if abs(best_tau_s - best_tau_d) <= 0.5:
        avg_best = (best_tau_s + best_tau_d) / 2
        ax.axvline(avg_best, color=COLOR_GUIDE, linestyle='--', linewidth=2, alpha=0.8)
        ax.text(avg_best, max(np.max(p_ks_s), np.max(p_ks_d)) * 1.02, 
                f"Critical\n~{avg_best:.1f}ms", ha='center', va='bottom', 
                color=COLOR_GUIDE, fontweight='bold', fontsize=14)
    else:
        ax.axvline(best_tau_s, color=COLOR_SIZE, linestyle='--', alpha=0.6)
        ax.axvline(best_tau_d, color=COLOR_DUR, linestyle='--', alpha=0.6)

    # Styling
    ax.set_title("Criticality Identification: KS Distance Minimization", fontweight='bold', pad=20)
    ax.set_xlabel(r"Inhibitory Decay Time Constant $\tau_I^d$ (ms)", fontsize=16)
    ax.set_ylabel("KS Distance (Deviation from Power-Law)", fontsize=16)
    
    ax.legend(fontsize=14, loc='upper center')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    # Save
    save_name = "KS_Distance_Minimization.png"
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    plot_ks_minimization()