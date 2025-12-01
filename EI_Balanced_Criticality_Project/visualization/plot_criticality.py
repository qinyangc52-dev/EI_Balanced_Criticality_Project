# %% Visualization: Paper Replication Figure (Publication Quality)
# FIXED: IndexError in guide line plotting
# UPDATED VERSION: Added Scaling Relation Plot (<S> vs T)

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import FIGURE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TAU_DECAY_I_LIST
from utils.io_manager import load_pkl

# ==========================================
# 1. Configuration & Style
# ==========================================
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)

# Colors
COLOR_SUB = '#2b8cbe'  # Subcritical
COLOR_CRI = '#d95f02'  # Critical
COLOR_SUP = '#7570b3'  # Supercritical
COLOR_SCALING = '#1b9e77' # Green for Scaling

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.constrained_layout.use': True
})

def load_data(tau):
    """Helper to load spikes and stats safely."""
    raw_path = Path(RAW_DATA_DIR) / f"spikes_{tau:.1f}.npz"
    stat_path = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
    spikes = np.load(raw_path) if raw_path.exists() else None
    stats = load_pkl(stat_path) if stat_path.exists() else None
    return spikes, stats

def plot_publication_figure():
    # 1. Define Regimes (Adjust based on your best fit)
    tau_sub = 2.0
    tau_crit = 8.0 
    tau_sup = 11.0 # Or 12.0

    print(f"Plotting Regimes: Sub={tau_sub}, Crit={tau_crit}, Sup={tau_sup}")

    # 2. Setup Figure (Increased width for the new panel)
    fig = plt.figure(figsize=(18, 10)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 1.0], hspace=0.35)
    
    # Top Row: 3 Raster Plots
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.15)
    # Bottom Row: 3 Analysis Plots (Dist, Scaling, Phase)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)

    # =================================================================
    # TOP ROW: RASTER PLOTS
    # =================================================================
    configs = [
        (tau_sub, COLOR_SUB, 'Subcritical'),
        (tau_crit, COLOR_CRI, 'Critical'),
        (tau_sup, COLOR_SUP, 'Supercritical')
    ]

    for i, (tau, color, label) in enumerate(configs):
        ax = fig.add_subplot(gs_top[i])
        spikes, _ = load_data(tau)
        
        if spikes:
            ts = spikes['spike_times']
            ns = spikes['neuron_indices']
            t_max = ts.max()
            t_min = max(0, t_max - 1000)
            mask = (ts >= t_min) & (ts <= t_max)
            
            ax.scatter(ts[mask] - t_min, ns[mask], s=1.0, c=color, alpha=0.8, edgecolors='none')
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000) # Assuming 1000 neurons

        title_str = f"{label}\n($\\tau_I^d = {tau}$ ms)"
        ax.set_title(title_str, color=color, fontweight='bold', fontsize=14)
        ax.set_xlabel('Time (ms)')
        
        if i == 0:
            ax.set_ylabel('Neuron Index')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # =================================================================
    # BOTTOM LEFT: DISTRIBUTIONS (P(S) & P(T))
    # =================================================================
    ax_dist = fig.add_subplot(gs_bot[0])
    _, stats_crit = load_data(tau_crit)
    
    if stats_crit and 'distribution' in stats_crit:
        dist = stats_crit['distribution']
        s_vals = dist['size_values']
        s_probs = dist['size_probs']
        t_vals = dist['duration_values']
        t_probs = dist['duration_probs']

        # Size
        mask_s = (s_vals > 0) & (s_probs > 0)
        filtered_s = s_vals[mask_s]
        filtered_s_probs = s_probs[mask_s]
        
        ax_dist.loglog(filtered_s, filtered_s_probs, 'o', color=COLOR_CRI, 
                       markersize=5, alpha=0.6, label='Size $P(S)$')
        
        # Draw guide line for Size
        tau_val = stats_crit.get('size_exponent', 1.5)
        if len(filtered_s) > 5:
            # FIX: Use length of filtered array
            mid_idx = len(filtered_s) // 3
            mid_x = filtered_s[mid_idx]
            mid_y = filtered_s_probs[mid_idx]
            
            x_guide = np.linspace(min(filtered_s), max(filtered_s), 100)
            y_guide = mid_y * (x_guide / mid_x)**(-tau_val)
            ax_dist.loglog(x_guide, y_guide, '--', color=COLOR_CRI, alpha=0.5)

        # Duration (Shifted for visibility)
        mask_t = (t_vals > 0) & (t_probs > 0)
        filtered_t = t_vals[mask_t]
        filtered_t_probs = t_probs[mask_t] * 10 # Shift up
        
        ax_dist.loglog(filtered_t, filtered_t_probs, 's', color='gray', 
                       markersize=5, alpha=0.6, label='Duration $P(T) \\times 10$')

        # Draw guide line for Duration
        alpha_val = stats_crit.get('duration_exponent', 2.0)
        if len(filtered_t) > 5:
            # FIX: Use length of filtered array
            mid_idx_t = len(filtered_t) // 3
            mid_x_t = filtered_t[mid_idx_t]
            mid_y_t = filtered_t_probs[mid_idx_t]
            
            x_guide_t = np.linspace(min(filtered_t), max(filtered_t), 100)
            y_guide_t = mid_y_t * (x_guide_t / mid_x_t)**(-alpha_val)
            ax_dist.loglog(x_guide_t, y_guide_t, '--', color='gray', alpha=0.5)

        # Text Label
        ax_dist.text(0.05, 0.1, f"$\\tau \\approx {tau_val:.2f}$\n$\\alpha \\approx {alpha_val:.2f}$", 
                     transform=ax_dist.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    ax_dist.set_title("Avalanche Distributions", fontweight='bold')
    ax_dist.set_xlabel("Size (S) or Duration (T)")
    ax_dist.set_ylabel("Probability Density")
    ax_dist.legend(loc='upper right', fontsize=10)
    ax_dist.grid(True, which="both", ls="-", alpha=0.1)

    # =================================================================
    # BOTTOM CENTER: SCALING RELATION (<S> vs T)
    # =================================================================
    ax_scal = fig.add_subplot(gs_bot[1])
    
    if stats_crit and 'preprocessing' in stats_crit:
        # 1. Extract Raw Data
        raw_sizes = stats_crit['preprocessing']['avalanche_sizes']
        raw_durs = stats_crit['preprocessing']['avalanche_durations']
        
        # 2. Compute Mean Size per Duration <S>(T)
        unique_durs = np.unique(raw_durs)
        mean_sizes = []
        valid_durs = []
        
        for d in unique_durs:
            # Filter for sufficient statistics
            sizes_in_dur = raw_sizes[raw_durs == d]
            if len(sizes_in_dur) >= 3: 
                mean_sizes.append(np.mean(sizes_in_dur))
                valid_durs.append(d)
        
        valid_durs = np.array(valid_durs)
        mean_sizes = np.array(mean_sizes)

        # 3. Plot Data
        ax_scal.loglog(valid_durs, mean_sizes, 'o', color=COLOR_SCALING, 
                       markersize=6, alpha=0.8, label='Data $\\langle S \\rangle (T)$')
        
        # 4. Theoretical Prediction
        # Relation: <S> ~ T ^ (1 / sigma*nu*z)
        # Predicted exponent gamma = (alpha - 1) / (tau - 1)
        if tau_val > 1:
            gamma_pred = (alpha_val - 1) / (tau_val - 1)
            
            # Fit line for visual guide
            if len(valid_durs) > 5:
                mid_x = valid_durs[len(valid_durs)//2]
                mid_y = mean_sizes[len(mean_sizes)//2]
                x_guide = np.linspace(min(valid_durs), max(valid_durs), 100)
                y_guide = mid_y * (x_guide / mid_x)**gamma_pred
                
                ax_scal.loglog(x_guide, y_guide, 'k--', linewidth=2, 
                               label=f'Pred Slope $\\approx {gamma_pred:.2f}$')

    ax_scal.set_title("Scaling Relation", fontweight='bold')
    ax_scal.set_xlabel("Duration $T$ (bins)")
    ax_scal.set_ylabel("Average Size $\\langle S \\rangle$")
    ax_scal.legend(fontsize=10)
    ax_scal.grid(True, which="both", ls="-", alpha=0.1)

    # =================================================================
    # BOTTOM RIGHT: PHASE TRANSITION (KS DISTANCE)
    # =================================================================
    ax_ks = fig.add_subplot(gs_bot[2])
    
    taus = []
    ks_sizes = []
    
    for t in sorted(TAU_DECAY_I_LIST):
        _, s = load_data(t)
        if s and 'size_ks' in s and s['size_ks'] is not None:
            taus.append(t)
            ks_sizes.append(s['size_ks'])
    
    if taus:
        ax_ks.plot(taus, ks_sizes, 'o-', color='#e7298a', linewidth=2.5, label='KS Distance')
        
        # Highlight minimum
        min_idx = np.argmin(ks_sizes)
        best_tau = taus[min_idx]
        
        ax_ks.axvline(best_tau, color='gray', linestyle='--', alpha=0.5)
        ax_ks.text(best_tau, max(ks_sizes)*0.9, f"Optimum\n{best_tau}ms", 
                   ha='center', color='#e7298a', fontweight='bold')

    ax_ks.set_title("Critical Point Identification", fontweight='bold')
    ax_ks.set_xlabel(r"$\tau_I^d$ (ms)")
    ax_ks.set_ylabel("KS Distance (Lower is Better)")
    ax_ks.grid(True, linestyle='--', alpha=0.3)

    # =================================================================
    # SAVE
    # =================================================================
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(FIGURE_DIR) / "Paper_Replication_Figure_1_Complete.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    print("Starting visualization...")
    plot_publication_figure()