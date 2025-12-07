#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合临界性可视化 - 复现论文 Figure 1
展示：光栅图、幂律分布、临界性指标
自动检测最佳临界点，不使用硬编码
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    TAU_DECAY_I_LIST, RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURE_DIR
)
from utils.io_manager import load_npz, load_pkl

# 设置绘图风格
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

# 颜色方案
COLORS = {
    'subcritical': '#3498DB',
    'critical': '#E74C3C',
    'supercritical': '#95A5A6',
    'size': '#C0392B',
    'duration': '#2980B9',
    'powerlaw': '#E67E22',
}


def load_all_data():
    """加载所有tau值的数据"""
    data = {}
    
    for tau in TAU_DECAY_I_LIST:
        raw_file = Path(RAW_DATA_DIR) / f"spikes_{tau:.1f}.npz"
        if raw_file.exists():
            raw_data = dict(load_npz(raw_file))
            
            # 统一字段名
            if 'neuron_indices' in raw_data and 'neuron_ids' not in raw_data:
                raw_data['neuron_ids'] = raw_data['neuron_indices']
            
            processed_file = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
            if processed_file.exists():
                processed_data = load_pkl(processed_file)
                data[tau] = {
                    'raw': raw_data,
                    'processed': processed_data
                }
    
    return data


def find_best_critical_point(data):
    """
    自动检测最佳临界点
    基于: Size exponent最接近1.5 + KS距离最小
    """
    best_tau = None
    best_score = float('inf')
    
    for tau, d in data.items():
        proc = d['processed']
        size_exp = proc.get('size_exponent')
        size_ks = proc.get('size_ks')
        
        if size_exp and size_ks:
            # 评分: 距离理论值1.5的偏差 + KS距离
            score = abs(size_exp - 1.5) + size_ks
            if score < best_score:
                best_score = score
                best_tau = tau
    
    return best_tau


def find_representative_taus(data, tau_critical):
    """
    找到代表性的三个tau值：次临界、临界、超临界
    """
    taus = sorted(data.keys())
    
    # 次临界：临界点之前最小的tau
    subcritical = min(taus)
    
    # 超临界：临界点之后最大的tau
    supercritical = max(taus)
    
    # 确保三个值不同
    if subcritical == tau_critical:
        subcritical = taus[0] if len(taus) > 1 else tau_critical
    if supercritical == tau_critical:
        supercritical = taus[-1] if len(taus) > 1 else tau_critical
    
    return subcritical, tau_critical, supercritical


def extract_sizes_durations(processed):
    """从processed数据中提取sizes和durations，兼容多种数据结构"""
    sizes = None
    durations = None
    
    # 尝试多种可能的位置
    if 'sizes' in processed:
        sizes = processed['sizes']
    elif 'distribution' in processed:
        dist = processed['distribution']
        if 'sizes' in dist:
            sizes = dist['sizes']
        elif 'avalanche_sizes' in dist:
            sizes = dist['avalanche_sizes']
    elif 'avalanche_sizes' in processed:
        sizes = processed['avalanche_sizes']
    
    if 'durations' in processed:
        durations = processed['durations']
    elif 'distribution' in processed:
        dist = processed['distribution']
        if 'durations' in dist:
            durations = dist['durations']
        elif 'avalanche_durations' in dist:
            durations = dist['avalanche_durations']
    elif 'avalanche_durations' in processed:
        durations = processed['avalanche_durations']
    
    # 转换为numpy数组
    if sizes is not None:
        sizes = np.array(sizes)
    if durations is not None:
        durations = np.array(durations)
    
    return sizes, durations


def plot_raster_comparison(data, ax, tau_values, time_window=(1000, 2000), max_neurons=100):
    """绘制不同tau值的光栅图对比"""
    n_conditions = len(tau_values)
    labels = ['Subcritical', 'Critical', 'Supercritical']
    colors = [COLORS['subcritical'], COLORS['critical'], COLORS['supercritical']]
    
    for i, tau in enumerate(tau_values):
        if tau not in data:
            continue
        
        spike_times = data[tau]['raw']['spike_times']
        if 'neuron_ids' in data[tau]['raw']:
            neuron_ids = data[tau]['raw']['neuron_ids']
        else:
            neuron_ids = data[tau]['raw']['neuron_indices']
        
        # 过滤时间窗口
        mask = (spike_times >= time_window[0]) & (spike_times < time_window[1])
        times = spike_times[mask]
        neurons = neuron_ids[mask]
        
        # 只显示前N个神经元
        neuron_mask = neurons < max_neurons
        times = times[neuron_mask]
        neurons = neurons[neuron_mask]
        
        # 计算y位置（从下到上：次临界、临界、超临界）
        y_offset = (n_conditions - 1 - i) * (max_neurons + 15)
        
        # 绘制spikes
        ax.scatter(times, neurons + y_offset, s=0.5, c='black', alpha=0.6, rasterized=True)
        
        # 添加tau标签
        ax.text(time_window[0] - 50, y_offset + max_neurons/2, 
                f'τ={tau:.1f}ms', ha='right', va='center', fontsize=10, fontweight='bold')
        
        # 添加状态标签
        ax.text(time_window[1] + 20, y_offset + max_neurons/2,
                labels[i], ha='left', va='center', fontsize=9, 
                style='italic', color=colors[i])
        
        # 分隔线
        if i < n_conditions - 1:
            ax.axhline(y_offset - 7, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(time_window)
    ax.set_ylim(-10, n_conditions * (max_neurons + 15))
    ax.set_xlabel('Time (ms)', fontweight='bold')
    ax.set_ylabel('Neuron ID', fontweight='bold')
    ax.set_title('(a) Network Activity Rasters', fontweight='bold', loc='left')
    ax.set_yticks([])


def plot_size_distribution(data, ax, tau_critical):
    """绘制临界点的Size分布"""
    if tau_critical not in data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    processed = data[tau_critical]['processed']
    sizes, _ = extract_sizes_durations(processed)
    
    if sizes is None or len(sizes) == 0:
        ax.text(0.5, 0.5, f'No size data\nKeys: {list(processed.keys())}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return
    
    # 计算概率分布
    sizes_int = sizes.astype(int)
    size_counts = np.bincount(sizes_int)
    size_values = np.arange(len(size_counts))
    size_probs = size_counts / size_counts.sum()
    
    # 过滤非零
    mask = size_probs > 0
    size_values = size_values[mask]
    size_probs = size_probs[mask]
    
    # 绘制散点
    ax.scatter(size_values, size_probs, s=30, alpha=0.7, 
               c=COLORS['size'], label='Empirical', edgecolors='white', linewidth=0.5)
    
    # 绘制幂律拟合
    size_exp = processed.get('size_exponent')
    size_xmin = processed.get('size_xmin', 1)
    
    if size_exp and size_xmin and len(size_values) > 0:
        x_fit = np.logspace(np.log10(max(1, size_xmin)), np.log10(size_values.max()), 50)
        # 归一化
        C = size_probs[size_values >= size_xmin].sum() / np.sum(x_fit ** (-size_exp))
        y_fit = C * x_fit ** (-size_exp)
        
        ax.plot(x_fit, y_fit, '--', color=COLORS['powerlaw'], 
                linewidth=2.5, label=f'τ={size_exp:.2f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche Size S', fontweight='bold')
    ax.set_ylabel('P(S)', fontweight='bold')
    ax.set_title(f'(b) Size Distribution (τ_d_I={tau_critical:.1f}ms)', 
                 fontweight='bold', loc='left')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')


def plot_duration_distribution(data, ax, tau_critical):
    """绘制临界点的Duration分布"""
    if tau_critical not in data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    processed = data[tau_critical]['processed']
    _, durations = extract_sizes_durations(processed)
    
    if durations is None or len(durations) == 0:
        ax.text(0.5, 0.5, f'No duration data\nKeys: {list(processed.keys())}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return
    
    # 计算概率分布
    dur_int = durations.astype(int)
    dur_counts = np.bincount(dur_int)
    dur_values = np.arange(len(dur_counts))
    dur_probs = dur_counts / dur_counts.sum()
    
    # 过滤非零
    mask = dur_probs > 0
    dur_values = dur_values[mask]
    dur_probs = dur_probs[mask]
    
    # 绘制散点
    ax.scatter(dur_values, dur_probs, s=30, alpha=0.7,
               c=COLORS['duration'], label='Empirical', edgecolors='white', linewidth=0.5)
    
    # 绘制幂律拟合
    dur_exp = processed.get('duration_exponent')
    dur_xmin = processed.get('duration_xmin', 1)
    
    if dur_exp and dur_xmin and len(dur_values) > 0:
        x_fit = np.logspace(np.log10(max(1, dur_xmin)), np.log10(dur_values.max()), 50)
        C = dur_probs[dur_values >= dur_xmin].sum() / np.sum(x_fit ** (-dur_exp))
        y_fit = C * x_fit ** (-dur_exp)
        
        ax.plot(x_fit, y_fit, '--', color=COLORS['powerlaw'],
                linewidth=2.5, label=f'α={dur_exp:.2f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche Duration T', fontweight='bold')
    ax.set_ylabel('P(T)', fontweight='bold')
    ax.set_title(f'(c) Duration Distribution (τ_d_I={tau_critical:.1f}ms)',
                 fontweight='bold', loc='left')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')


def plot_exponents_vs_tau(data, ax, tau_critical):
    """绘制Size和Duration指数随tau的变化"""
    taus = sorted(data.keys())
    size_exps = []
    dur_exps = []
    
    for tau in taus:
        size_exp = data[tau]['processed'].get('size_exponent')
        dur_exp = data[tau]['processed'].get('duration_exponent')
        size_exps.append(size_exp if size_exp else np.nan)
        dur_exps.append(dur_exp if dur_exp else np.nan)
    
    # 绘制曲线
    ax.plot(taus, size_exps, 'o-', color=COLORS['size'], 
            linewidth=2.5, markersize=8, label='Size exponent (τ)',
            markeredgecolor='white', markeredgewidth=1)
    
    ax.plot(taus, dur_exps, 's-', color=COLORS['duration'],
            linewidth=2.5, markersize=8, label='Duration exponent (α)',
            markeredgecolor='white', markeredgewidth=1)
    
    # 理论值参考线
    ax.axhline(1.5, color=COLORS['size'], linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axhline(2.0, color=COLORS['duration'], linestyle='--', alpha=0.4, linewidth=1.5)
    
    # 标记最佳临界点
    ax.axvline(tau_critical, color=COLORS['critical'], linestyle=':', 
               alpha=0.7, linewidth=2, label=f'Best: τ_d_I={tau_critical:.1f}ms')
    
    ax.set_xlabel('τ_d_I (ms)', fontweight='bold')
    ax.set_ylabel('Exponent', fontweight='bold')
    ax.set_title('(d) Power-law Exponents', fontweight='bold', loc='left')
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(1.0, 3.0)


def plot_ks_distance(data, ax, tau_critical):
    """绘制KS距离随tau的变化"""
    taus = sorted(data.keys())
    size_ks = []
    dur_ks = []
    
    for tau in taus:
        s_ks = data[tau]['processed'].get('size_ks')
        d_ks = data[tau]['processed'].get('duration_ks')
        size_ks.append(s_ks if s_ks else np.nan)
        dur_ks.append(d_ks if d_ks else np.nan)
    
    ax.plot(taus, size_ks, 'o-', color=COLORS['size'],
            linewidth=2.5, markersize=8, label='Size KS distance',
            markeredgecolor='white', markeredgewidth=1)
    
    ax.plot(taus, dur_ks, 's-', color=COLORS['duration'],
            linewidth=2.5, markersize=8, label='Duration KS distance',
            markeredgecolor='white', markeredgewidth=1)
    
    # 阈值线
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Good (KS<0.1)')
    ax.axhline(0.15, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Acceptable (KS<0.15)')
    
    # 标记最佳临界点
    ax.axvline(tau_critical, color=COLORS['critical'], linestyle=':', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('τ_d_I (ms)', fontweight='bold')
    ax.set_ylabel('KS Distance', fontweight='bold')
    ax.set_title('(e) Goodness of Fit', fontweight='bold', loc='left')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 0.3)


def create_comprehensive_figure():
    """创建完整的综合图"""
    print("\n" + "="*70)
    print("生成综合临界性可视化（Figure 1风格）")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载数据...")
    data = load_all_data()
    
    if not data:
        print("错误: 没有找到任何数据文件！")
        print("请先运行: python main.py --all")
        return
    
    print(f"已加载 {len(data)} 个tau值的数据")
    
    # 自动检测最佳临界点
    tau_critical = find_best_critical_point(data)
    print(f"✓ 自动检测最佳临界点: τ_d_I = {tau_critical:.1f} ms")
    
    # 找到代表性的三个tau值
    tau_sub, tau_crit, tau_super = find_representative_taus(data, tau_critical)
    print(f"  次临界: {tau_sub:.1f} ms")
    print(f"  临界:   {tau_crit:.1f} ms")
    print(f"  超临界: {tau_super:.1f} ms")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 子图
    ax_raster = fig.add_subplot(gs[:, 0])
    ax_size = fig.add_subplot(gs[0, 1])
    ax_dur = fig.add_subplot(gs[0, 2])
    ax_exp = fig.add_subplot(gs[1, 1])
    ax_ks = fig.add_subplot(gs[1, 2])
    
    # 绘制
    print("\n绘制子图:")
    
    print("  (a) 光栅图对比...")
    plot_raster_comparison(data, ax_raster, [tau_sub, tau_crit, tau_super])
    
    print("  (b) 雪崩Size分布...")
    plot_size_distribution(data, ax_size, tau_critical)
    
    print("  (c) 雪崩Duration分布...")
    plot_duration_distribution(data, ax_dur, tau_critical)
    
    print("  (d) 幂律指数...")
    plot_exponents_vs_tau(data, ax_exp, tau_critical)
    
    print("  (e) KS距离...")
    plot_ks_distance(data, ax_ks, tau_critical)
    
    # 总标题
    fig.suptitle(f'Evidence of Criticality in E-I Balanced Network (Best τ_d_I={tau_critical:.1f}ms)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 保存
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(FIGURE_DIR) / 'comprehensive_criticality.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ 图表已保存: {save_path}")
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("临界性指标摘要")
    print("="*70)
    proc = data[tau_critical]['processed']
    print(f"最佳临界点: τ_d_I = {tau_critical:.1f} ms")
    print(f"  Size exponent:     {proc.get('size_exponent', 'N/A'):.3f} (理论: ~1.5)")
    print(f"  Duration exponent: {proc.get('duration_exponent', 'N/A'):.3f} (理论: ~2.0)")
    print(f"  Size KS:           {proc.get('size_ks', 'N/A'):.4f} (越小越好)")
    print(f"  Duration KS:       {proc.get('duration_ks', 'N/A'):.4f} (越小越好)")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    create_comprehensive_figure()