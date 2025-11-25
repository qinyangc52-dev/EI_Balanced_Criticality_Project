# %% Stage 5: Visualization - Generate Criticality Figures
# NOTE: This file is intentionally minimal as per requirements.
# Actual plotting code should be added here after analysis is complete.

"""
Visualization Module

This module should generate figures similar to those in the paper:
1. Avalanche size/duration distributions (log-log plots)
2. Power-law exponents vs tau_d_I
3. Crackle noise relation verification
4. Spike raster plots for different regimes
5. Phase diagram (subcritical/critical/supercritical)

Example usage:
    from utils.io_manager import load_pkl
    results = load_pkl('data/processed/avalanche_stats_8.0.pkl')
    plot_avalanche_distribution(results)
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent  # 从 visualization 向上两级到项目根目录
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from configs.model_config import FIGURE_DIR, PROCESSED_DATA_DIR
from utils.io_manager import load_pkl


def plot_avalanche_distribution(tau_d_I: float):
    """
    Plot avalanche size and duration distributions for a given tau_d_I.
    """
    # Load processed data
    filename = f"avalanche_stats_{tau_d_I:.1f}.pkl"
    filepath = Path(PROCESSED_DATA_DIR) / filename
    results = load_pkl(filepath)
    
    # Extract data
    dist = results['distribution']
    size_values = dist['size_values']
    size_probs = dist['size_probs']
    duration_values = dist['duration_values']
    duration_probs = dist['duration_probs']
    size_exp = results['size_exponent']
    size_r2 = results['size_r2']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Size distribution
    ax = axes[0]
    ax.loglog(size_values + 1, size_probs + 1e-6, 'o-', alpha=0.6, markersize=6, label='Data')
    if size_exp:
        ax.set_title(f'Avalanche Size Distribution (τ_d_I = {tau_d_I:.1f} ms)\nα = {size_exp:.3f} (R2={size_r2:.3f})', 
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'Avalanche Size Distribution (τ_d_I = {tau_d_I:.1f} ms)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Avalanche Size (spikes)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Duration distribution
    ax = axes[1]
    ax.loglog(duration_values + 1, duration_probs + 1e-6, 's-', alpha=0.6, markersize=6, color='orange', label='Data')
    ax.set_title(f'Avalanche Duration Distribution (τ_d_I = {tau_d_I:.1f} ms)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Avalanche Duration (ms)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    filename = f"avalanche_dist_{tau_d_I:.1f}.png"
    filepath = Path(FIGURE_DIR) / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VISUALIZATION] Saved: {filepath}")
    plt.close()


def plot_phase_diagram():
    """
    Plot phase diagram showing power-law exponents across tau_d_I values.
    """
    from configs.model_config import TAU_DECAY_I_LIST
    
    # Load all results
    tau_values = []
    exponents = []
    r2_values = []
    
    for tau_d_I in TAU_DECAY_I_LIST:
        filename = f"avalanche_stats_{tau_d_I:.1f}.pkl"
        filepath = Path(PROCESSED_DATA_DIR) / filename
        try:
            results = load_pkl(filepath)
            tau_values.append(tau_d_I)
            exponents.append(results['size_exponent'] if results['size_exponent'] else 0)
            r2_values.append(results['size_r2'] if results['size_r2'] else 0)
        except:
            continue
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot exponents with error bars
    scatter = ax.scatter(tau_values, exponents, s=150, c=r2_values, cmap='viridis', 
                        edgecolors='black', linewidth=1.5, alpha=0.7, vmin=0, vmax=0.6)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Critical (α=1.0)', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('τ_d_I (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Size Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_title('Phase Diagram: Criticality vs Inhibitory Decay Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.1])
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Goodness-of-fit (R2)', fontsize=11)
    
    # Legend
    ax.legend(fontsize=11, loc='lower right')
    
    # Save figure
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    filepath = Path(FIGURE_DIR) / "phase_diagram.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VISUALIZATION] Saved: {filepath}")
    plt.close()


if __name__ == "__main__":
    print("Starting visualization...")
    
    # 绘制所有 tau_d_I 值的雪崩分布
    from configs.model_config import TAU_DECAY_I_LIST
    
    for tau_d_I in TAU_DECAY_I_LIST:
        try:
            print(f"Plotting avalanche distribution for tau_d_I = {tau_d_I}")
            plot_avalanche_distribution(tau_d_I)
        except Exception as e:
            print(f"Error plotting tau_d_I = {tau_d_I}: {e}")
    
    # 绘制相图
    try:
        print("Plotting phase diagram...")
        plot_phase_diagram()
    except Exception as e:
        print(f"Error plotting phase diagram: {e}")
    
    print("Visualization complete!")