# visualization/plot_criticality_metrics.py
"""
Visualization for Criticality Metrics: Sensitivity & Reliability
Replicates Figure 2 from Yang et al. (2025)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import PROCESSED_DATA_DIR, FIGURE_DIR
from utils.io_manager import load_pkl


def plot_criticality_metrics(results_file='criticality_metrics_results.pkl'):
    """
    Create comprehensive visualization of sensitivity and reliability.
    
    Plots:
        (a) Sensitivity vs tau_d_I
        (b) Reliability vs tau_d_I  
        (c) Sensitivity × Reliability (combined metric)
        (d) Firing rates (baseline vs signal)
    """
    
    # Load results
    filepath = Path(PROCESSED_DATA_DIR) / results_file
    if not filepath.exists():
        print(f"Error: Results file not found: {filepath}")
        print("Please run experiments/run_criticality_metrics.py first")
        return
    
    results = load_pkl(filepath)
    
    tau = np.array(results['tau'])
    sensitivity = np.array(results['sensitivity'])
    reliability = np.array(results['reliability'])
    r_baseline = np.array(results['r_baseline'])
    r_signal = np.array(results['r_signal'])
    
    # Find critical point
    combined = sensitivity * reliability
    critical_idx = np.argmax(combined)
    critical_tau = tau[critical_idx]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Style settings
    colors = {
        'sensitivity': '#E74C3C',
        'reliability': '#3498DB', 
        'combined': '#9B59B6',
        'baseline': '#95A5A6',
        'signal': '#27AE60',
        'critical': '#F39C12'
    }
    
    # === Panel (a): Sensitivity ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tau, sensitivity, 'o-', color=colors['sensitivity'], 
             linewidth=2.5, markersize=8, label='Sensitivity')
    ax1.axvline(critical_tau, color=colors['critical'], linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Critical ({critical_tau:.1f} ms)')
    
    ax1.set_xlabel(r'$\tau_I^d$ (ms)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sensitivity', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Sensitivity to Input Signal', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # === Panel (b): Reliability ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(tau, reliability, 's-', color=colors['reliability'],
             linewidth=2.5, markersize=8, label='Reliability')
    ax2.axvline(critical_tau, color=colors['critical'], linestyle='--',
                linewidth=2, alpha=0.7, label=f'Critical ({critical_tau:.1f} ms)')
    
    ax2.set_xlabel(r'$\tau_I^d$ (ms)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Reliability', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Response Reliability', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # === Panel (c): Combined Metric ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(tau, combined, 'D-', color=colors['combined'],
             linewidth=2.5, markersize=8, label='Sensitivity × Reliability')
    ax3.axvline(critical_tau, color=colors['critical'], linestyle='--',
                linewidth=2, alpha=0.7)
    ax3.scatter([critical_tau], [combined[critical_idx]], 
                s=200, color=colors['critical'], marker='*', 
                edgecolors='black', linewidths=1.5, zorder=10,
                label=f'Peak @ {critical_tau:.1f} ms')
    
    ax3.set_xlabel(r'$\tau_I^d$ (ms)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Sens × Rel', fontsize=13, fontweight='bold')
    ax3.set_title('(c) Combined Metric (Optimal Performance)', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, linestyle=':')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # === Panel (d): Firing Rates ===
    ax4 = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(tau))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, r_baseline, width, 
                    label='Baseline', color=colors['baseline'], alpha=0.8)
    bars2 = ax4.bar(x + width/2, r_signal, width,
                    label='Signal', color=colors['signal'], alpha=0.8)
    
    # Highlight critical point
    ax4.axvline(critical_idx, color=colors['critical'], linestyle='--',
                linewidth=2, alpha=0.5)
    
    ax4.set_xlabel(r'$\tau_I^d$ (ms)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Firing Rate (Hz)', fontsize=13, fontweight='bold')
    ax4.set_title('(d) Network Activity Levels', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{t:.1f}' for t in tau], rotation=45)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, linestyle=':', axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle('Criticality Metrics: Sensitivity & Reliability Coexistence',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    save_path = Path(FIGURE_DIR) / 'criticality_metrics_full.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Critical point: tau_d_I = {critical_tau:.1f} ms")
    print(f"  Sensitivity:  {sensitivity[critical_idx]:.4f}")
    print(f"  Reliability:  {reliability[critical_idx]:.4f}")
    print(f"  Combined:     {combined[critical_idx]:.4f}")
    print(f"  Baseline rate: {r_baseline[critical_idx]:.2f} Hz")
    print(f"  Signal rate:   {r_signal[critical_idx]:.2f} Hz")
    print("="*60 + "\n")


def plot_reliability_raster(tau_value=8.5, results_file='criticality_metrics_results.pkl'):
    """
    Plot raster plot showing trial-to-trial reliability for a specific tau.
    """
    filepath = Path(PROCESSED_DATA_DIR) / results_file
    results = load_pkl(filepath)
    
    if tau_value not in results['details']:
        print(f"Error: No data for tau = {tau_value}")
        return
    
    spike_counts = results['details'][tau_value]['reliability']['spike_counts']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Raster plot (sample first 20 trials)
    n_show = min(20, spike_counts.shape[0])
    im = ax1.imshow(spike_counts[:n_show], aspect='auto', cmap='hot',
                    interpolation='nearest')
    ax1.set_ylabel('Trial #', fontsize=12, fontweight='bold')
    ax1.set_title(f'Trial-to-Trial Spike Counts (tau_d_I = {tau_value} ms)',
                  fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Spike Count')
    
    # Mean ± std across trials
    mean_counts = np.mean(spike_counts, axis=0)
    std_counts = np.std(spike_counts, axis=0)
    time = np.arange(len(mean_counts)) * 0.05  # DT = 0.05 ms
    
    ax2.plot(time, mean_counts, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(time, mean_counts - std_counts, mean_counts + std_counts,
                     alpha=0.3, color='b', label='±1 SD')
    ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spike Count', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    save_path = Path(FIGURE_DIR) / f'reliability_raster_tau{tau_value:.1f}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Raster plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', type=float, default=None,
                       help='Plot reliability raster for specific tau value')
    args = parser.parse_args()
    
    if args.raster is not None:
        plot_reliability_raster(tau_value=args.raster)
    else:
        plot_criticality_metrics()