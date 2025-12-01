# %% Visualization: Paper Replication Figure (Publication Quality)
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # <--- 之前缺失的就是这一行
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
sns.set_context("paper", font_scale=1.4)

# Colors
COLOR_SUB = '#66c2a5'  # Teal
COLOR_CRI = '#fc8d62'  # Orange
COLOR_SUP = '#8da0cb'  # Periwinkle Blue
COLOR_PHASE = '#d62728' # Red

# Plot Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
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
    # 1. Define Regimes
    tau_sub = 2.0
    tau_crit = 8.0
    tau_sup = 12.0 

    print(f"Plotting Regimes: Sub={tau_sub}, Crit={tau_crit}, Sup={tau_sup}")

    # 2. Setup Figure
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 1.0], hspace=0.25)
    
    # Top Row (Rasters)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.2)
    # Bottom Row (Dist & Phase)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.8, 1.2], wspace=0.25)

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
            # Show last 1000ms
            t_max = ts.max()
            t_min = max(0, t_max - 1000)
            mask = (ts >= t_min) & (ts <= t_max)
            
            # Plot Spikes
            ax.scatter(ts[mask] - t_min, ns[mask], s=0.5, c=color, alpha=0.9, edgecolors='none')
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, ns.max()*1.05) 

        title_str = f"{label}\n($\\tau_I^d = {tau}$ ms)"
        ax.set_title(title_str, color=color, fontweight='bold', fontsize=14, loc='center')
        ax.set_xlabel('Time (ms)')
        if i == 0:
            ax.set_ylabel('Neuron Index', fontsize=12)
        else:
            ax.set_yticklabels([])
            
        ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.2, color='gray')

    # =================================================================
    # BOTTOM LEFT: DISTRIBUTIONS
    # =================================================================
    ax_dist = fig.add_subplot(gs_bot[0])
    _, stats_crit = load_data(tau_crit)
    
    if stats_crit and 'distribution' in stats_crit:
        dist = stats_crit['distribution']
        
        # Duration
        t_vals = np.array(dist['duration_values'])
        t_probs = np.array(dist['duration_probs'])
        mask_t = (t_vals > 0) & (t_probs > 0)
        ax_dist.loglog(t_vals[mask_t], t_probs[mask_t], 's', color='#9ebcda', 
                       markersize=5, label='Duration (T)', alpha=0.8)

        # Size
        s_vals = np.array(dist['size_values'])
        s_probs = np.array(dist['size_probs'])
        mask_s = (s_vals > 0) & (s_probs > 0)
        ax_dist.loglog(s_vals[mask_s], s_probs[mask_s], 'o', color='#e34a33', 
                       markersize=5, label='Size (S)', alpha=0.6)

    ax_dist.set_title("Avalanche Distributions", fontweight='bold', fontsize=14)
    ax_dist.set_xlabel("Size (S) / Duration (T)", fontsize=12)
    ax_dist.set_ylabel("Probability P(x)", fontsize=12)
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, which="both", ls="-", alpha=0.15)

    # =================================================================
    # BOTTOM RIGHT: PHASE TRANSITION
    # =================================================================
    ax_phase = fig.add_subplot(gs_bot[1])
    
    taus = []
    exponents = []
    
    for t in sorted(TAU_DECAY_I_LIST):
        _, s = load_data(t)
        if s:
            val = s.get('size_exponent')
            # Valid range filter
            if val is not None and not np.isnan(val) and -2.0 < val < 6.0:
                taus.append(t)
                exponents.append(val)
    
    if taus:
        ax_phase.plot(taus, exponents, 'o-', color=COLOR_PHASE, linewidth=2.5, markersize=7, label='Size Exp ($\\tau$)')
        
        # Shade
        y = np.array(exponents)
        ax_phase.fill_between(taus, y - 0.15, y + 0.15, color=COLOR_PHASE, alpha=0.1)

        # Critical Line
        ax_phase.axvline(tau_crit, color='gray', linestyle='--', linewidth=2, label='Critical Point')

    ax_phase.set_title("Phase Transition", fontweight='bold', fontsize=14)
    ax_phase.set_xlabel(r"Inhibitory Decay Time $\tau_I^d$ (ms)", fontsize=12)
    ax_phase.set_ylabel(r"Size Exponent $\tau$", color=COLOR_PHASE, fontsize=12)
    ax_phase.tick_params(axis='y', labelcolor=COLOR_PHASE)
    ax_phase.grid(True, which='major', linestyle='-', alpha=0.3)
    ax_phase.legend(loc='upper right')

    # =================================================================
    # SAVE
    # =================================================================
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(FIGURE_DIR) / "Paper_Replication_Final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    # plt.show() # Comment out if running on a server without display

if __name__ == "__main__":
    print("Starting visualization...")
    plot_publication_figure()