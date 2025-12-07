#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
灵敏性与可靠性可视化 - 复现论文 Figure 2
展示临界态下功能指标的协同优化
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


# 颜色方案
COLORS = {
    'sensitivity': '#E74C3C',
    'reliability': '#3498DB',
    'combined': '#9B59B6',
    'baseline': '#95A5A6',
    'critical_region': '#FFF9C4'
}


def load_sr_data():
    """
    加载sensitivity & reliability数据
    """
    # 优先加载当前生成的 bn/bp 文件
    bn_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bn.pkl'
    bp_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bp.pkl'
    if bn_path.exists():
        print(f"加载文件: {bn_path}")
        bn = load_pkl(bn_path)
        balanced = {
            'tau_values': bn.get('tau', bn.get('tau_values', [])),
            'sensitivity': bn.get('sensitivity', []),
            'reliability': bn.get('reliability', []),
        }
        return {'balanced': balanced}

    # 回退到旧文件名以兼容早期版本
    possible_files = [
        'sensitivity_reliability_balanced.pkl',
        'sensitivity_reliability.pkl',
        'sr_results.pkl'
    ]
    for filename in possible_files:
        filepath = Path(PROCESSED_DATA_DIR) / filename
        if filepath.exists():
            print(f"加载文件: {filepath}")
            return load_pkl(filepath)

    print("错误: 未找到sensitivity/reliability数据文件")
    print("请先运行: python experiments/sensitivity_reliability.py")
    return None


def plot_sensitivity_reliability_coexistence():
    """
    主函数：绘制灵敏性与可靠性的共存图
    """
    print("\n" + "="*70)
    print("生成灵敏性与可靠性可视化（Figure 2风格）")
    print("="*70 + "\n")
    
    # 加载数据
    data = load_sr_data()
    if data is None:
        return
    
    # 提取balanced network数据
    if 'balanced' in data:
        balanced = data['balanced']
        taus = balanced['tau_values']
        sensitivity = balanced['sensitivity']
        reliability = balanced['reliability']
    else:
        # 兼容旧格式
        taus = data.get('tau_values', [])
        sensitivity = data.get('sensitivity', [])
        reliability = data.get('reliability', [])
    
    if len(taus) == 0:
        print("错误: 数据为空")
        return
    
    print(f"加载了 {len(taus)} 个tau值的数据")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # ===== (a) Sensitivity vs tau =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.plot(taus, sensitivity, 'o-', color=COLORS['sensitivity'],
            linewidth=2.5, markersize=8, label='Sensitivity',
            markeredgecolor='white', markeredgewidth=1)
    
    # 标记临界区域
    critical_region = (7, 9)
    ax1.axvspan(critical_region[0], critical_region[1], 
               alpha=0.2, color=COLORS['critical_region'], 
               label='Critical region')
    
    # 找到峰值
    max_idx = np.argmax(sensitivity)
    max_tau = taus[max_idx]
    max_sens = sensitivity[max_idx]
    
    ax1.plot(max_tau, max_sens, '*', color='gold', 
            markersize=20, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=10,
            label=f'Peak: τ={max_tau:.1f}ms')
    
    ax1.set_xlabel('τ_d_I (ms)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Sensitivity (Δr/r)', fontweight='bold', fontsize=12)
    ax1.set_title('(a) Sensitivity to Input Signal', 
                 fontweight='bold', fontsize=13, loc='left')
    ax1.legend(loc='best', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # ===== (b) Reliability vs tau =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.plot(taus, reliability, 's-', color=COLORS['reliability'],
            linewidth=2.5, markersize=8, label='Reliability',
            markeredgecolor='white', markeredgewidth=1)
    
    # 标记临界区域
    ax2.axvspan(critical_region[0], critical_region[1],
               alpha=0.2, color=COLORS['critical_region'])
    
    # 找到峰值
    max_idx = np.argmax(reliability)
    max_tau = taus[max_idx]
    max_rel = reliability[max_idx]
    
    ax2.plot(max_tau, max_rel, '*', color='gold',
            markersize=20, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10,
            label=f'Peak: τ={max_tau:.1f}ms')
    
    ax2.set_xlabel('τ_d_I (ms)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Reliability (1/FF)', fontweight='bold', fontsize=12)
    ax2.set_title('(b) Response Reliability',
                 fontweight='bold', fontsize=13, loc='left')
    ax2.legend(loc='best', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # ===== (c) Combined Metric =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    combined = np.array(sensitivity) * np.array(reliability)
    
    ax3.plot(taus, combined, 'D-', color=COLORS['combined'],
            linewidth=2.5, markersize=8, label='Sens × Rel',
            markeredgecolor='white', markeredgewidth=1)
    
    # 标记临界区域
    ax3.axvspan(critical_region[0], critical_region[1],
               alpha=0.2, color=COLORS['critical_region'])
    
    # 找到峰值
    max_idx = np.argmax(combined)
    max_tau = taus[max_idx]
    max_combined = combined[max_idx]
    
    ax3.plot(max_tau, max_combined, '*', color='gold',
            markersize=20, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10,
            label=f'Optimal: τ={max_tau:.1f}ms')
    
    ax3.set_xlabel('τ_d_I (ms)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Combined Metric', fontweight='bold', fontsize=12)
    ax3.set_title('(c) Optimal Trade-off Point',
                 fontweight='bold', fontsize=13, loc='left')
    ax3.legend(loc='best', frameon=True, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle=':')
    
    # ===== (d) Dual-axis plot =====
    ax4 = fig.add_subplot(gs[1, :2])
    
    # 左轴：Sensitivity
    ax4_left = ax4
    line1 = ax4_left.plot(taus, sensitivity, 'o-', 
                          color=COLORS['sensitivity'],
                          linewidth=3, markersize=9, 
                          label='Sensitivity',
                          markeredgecolor='white', 
                          markeredgewidth=1.5)
    ax4_left.set_xlabel('τ_d_I (ms)', fontweight='bold', fontsize=12)
    ax4_left.set_ylabel('Sensitivity', fontweight='bold', 
                       fontsize=12, color=COLORS['sensitivity'])
    ax4_left.tick_params(axis='y', labelcolor=COLORS['sensitivity'])
    
    # 右轴：Reliability
    ax4_right = ax4_left.twinx()
    line2 = ax4_right.plot(taus, reliability, 's-',
                          color=COLORS['reliability'],
                          linewidth=3, markersize=9,
                          label='Reliability',
                          markeredgecolor='white',
                          markeredgewidth=1.5)
    ax4_right.set_ylabel('Reliability', fontweight='bold',
                        fontsize=12, color=COLORS['reliability'])
    ax4_right.tick_params(axis='y', labelcolor=COLORS['reliability'])
    
    # 标记临界区域
    ax4_left.axvspan(critical_region[0], critical_region[1],
                    alpha=0.15, color=COLORS['critical_region'],
                    label='Critical region')
    
    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4_left.legend(lines, labels, loc='upper left', 
                   frameon=True, framealpha=0.9, fontsize=10)
    
    ax4_left.set_title('(d) Coexistence of Sensitivity and Reliability',
                      fontweight='bold', fontsize=13, loc='left')
    ax4_left.grid(True, alpha=0.3, linestyle=':')
    
    # ===== (e) 散点图：Sensitivity vs Reliability =====
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 根据tau着色
    colors = []
    for tau in taus:
        if tau < 7:
            colors.append(COLORS['baseline'])  # 次临界
        elif tau <= 9:
            colors.append(COLORS['combined'])  # 临界
        else:
            colors.append(COLORS['baseline'])  # 超临界
    
    scatter = ax5.scatter(sensitivity, reliability, 
                         c=colors, s=150, alpha=0.7,
                         edgecolors='black', linewidth=1.5)
    
    # 添加tau标签
    for i, tau in enumerate(taus):
        ax5.annotate(f'{tau:.1f}', 
                    (sensitivity[i], reliability[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # 理想点（右上角）
    ax5.axhline(np.max(reliability), color='gray', 
               linestyle='--', alpha=0.3, linewidth=1)
    ax5.axvline(np.max(sensitivity), color='gray',
               linestyle='--', alpha=0.3, linewidth=1)
    
    ax5.set_xlabel('Sensitivity', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Reliability', fontweight='bold', fontsize=12)
    ax5.set_title('(e) Trade-off Space',
                 fontweight='bold', fontsize=13, loc='left')
    ax5.grid(True, alpha=0.3, linestyle=':')
    
    # 总标题
    fig.suptitle('Sensitivity and Reliability Coexistence at Criticality',
                fontsize=16, fontweight='bold', y=0.98)
    
    # 保存
    save_path = Path(FIGURE_DIR) / 'sensitivity_reliability_comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"\n✓ 图表已保存: {save_path}")
    
    # 打印统计信息
    print("\n" + "="*70)
    print("统计摘要:")
    print("="*70)
    
    max_sens_idx = np.argmax(sensitivity)
    max_rel_idx = np.argmax(reliability)
    max_combined_idx = np.argmax(combined)
    
    print(f"\n最大Sensitivity:")
    print(f"  τ_d_I = {taus[max_sens_idx]:.1f} ms")
    print(f"  值 = {sensitivity[max_sens_idx]:.4f}")
    
    print(f"\n最大Reliability:")
    print(f"  τ_d_I = {taus[max_rel_idx]:.1f} ms")
    print(f"  值 = {reliability[max_rel_idx]:.4f}")
    
    print(f"\n最优Combined:")
    print(f"  τ_d_I = {taus[max_combined_idx]:.1f} ms")
    print(f"  Sens × Rel = {combined[max_combined_idx]:.6f}")
    print(f"  Sensitivity = {sensitivity[max_combined_idx]:.4f}")
    print(f"  Reliability = {reliability[max_combined_idx]:.4f}")
    
    print("\n" + "="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    plot_sensitivity_reliability_coexistence()
