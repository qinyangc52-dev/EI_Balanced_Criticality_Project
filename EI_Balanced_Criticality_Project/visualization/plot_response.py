# %% Enhanced Analysis for Sensitivity-Reliability Trade-off

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from configs.model_config import PROCESSED_DATA_DIR, FIGURE_DIR

def analyze_results(bn_data, bp_data):
    """
    深度分析灵敏度-可靠性权衡
    """
    
    # 1. 找到关键点
    bn_df = pd.DataFrame(bn_data)
    
    # 灵敏度峰值
    max_sens_idx = bn_df['sensitivity'].idxmax()
    max_sens_tau = bn_df.loc[max_sens_idx, 'tau']
    max_sens_val = bn_df.loc[max_sens_idx, 'sensitivity']
    
    # 可靠性峰值
    max_rel_idx = bn_df['reliability'].idxmax()
    max_rel_tau = bn_df.loc[max_rel_idx, 'tau']
    max_rel_val = bn_df.loc[max_rel_idx, 'reliability']
    
    print("\n" + "="*60)
    print("关键发现总结")
    print("="*60)
    print(f"最大灵敏度: {max_sens_val:.4f} at τ_I = {max_sens_tau:.1f} ms")
    print(f"最大可靠性: {max_rel_val:.4f} at τ_I = {max_rel_tau:.1f} ms")
    print(f"分离距离: Δτ_I = {abs(max_sens_tau - max_rel_tau):.1f} ms")
    
    # 2. 计算综合性能指标
    # 方法1: 加权和
    weights = [0.5, 0.5]  # 可调整权重
    bn_df['composite_equal'] = (
        weights[0] * bn_df['sensitivity'] / bn_df['sensitivity'].max() +
        weights[1] * bn_df['reliability'] / bn_df['reliability'].max()
    )
    
    # 方法2: 几何平均（惩罚不平衡）
    bn_df['composite_geometric'] = np.sqrt(
        (bn_df['sensitivity'] / bn_df['sensitivity'].max()) *
        (bn_df['reliability'] / bn_df['reliability'].max())
    )
    
    best_composite_idx = bn_df['composite_geometric'].idxmax()
    best_tau = bn_df.loc[best_composite_idx, 'tau']
    
    print(f"\n综合最优点 (几何平均): τ_I = {best_tau:.1f} ms")
    print(f"  - 灵敏度: {bn_df.loc[best_composite_idx, 'sensitivity']:.4f}")
    print(f"  - 可靠性: {bn_df.loc[best_composite_idx, 'reliability']:.4f}")
    
    # 3. 分析不同regime
    print("\n" + "-"*60)
    print("动力学状态分类:")
    print("-"*60)
    
    for idx, row in bn_df.iterrows():
        tau = row['tau']
        sens = row['sensitivity']
        rel = row['reliability']
        
        if tau < 4.0:
            regime = "快速抑制区 (Fast Inhibition)"
        elif 6.0 <= tau <= 9.0:
            regime = "临界态候选区 (Critical Zone)"
        else:
            regime = "慢速抑制区 (Slow Inhibition)"
        
        print(f"τ_I={tau:4.1f} ms: {regime:30s} | S={sens:6.4f}, R={rel:6.4f}")
    
    return bn_df


def plot_enhanced_analysis(bn_df):
    """
    增强版可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trade-off 散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        bn_df['sensitivity'], 
        bn_df['reliability'],
        c=bn_df['tau'], 
        cmap='viridis', 
        s=100,
        edgecolors='black',
        linewidths=1.5
    )
    
    # 标注关键点
    max_sens_idx = bn_df['sensitivity'].idxmax()
    max_rel_idx = bn_df['reliability'].idxmax()
    
    ax1.scatter(
        bn_df.loc[max_sens_idx, 'sensitivity'],
        bn_df.loc[max_sens_idx, 'reliability'],
        marker='*', s=500, c='red', edgecolors='black', linewidths=2,
        label=f"Max Sens (τ={bn_df.loc[max_sens_idx, 'tau']:.1f}ms)"
    )
    
    ax1.scatter(
        bn_df.loc[max_rel_idx, 'sensitivity'],
        bn_df.loc[max_rel_idx, 'reliability'],
        marker='s', s=300, c='blue', edgecolors='black', linewidths=2,
        label=f"Max Rel (τ={bn_df.loc[max_rel_idx, 'tau']:.1f}ms)"
    )
    
    ax1.set_xlabel('Sensitivity (Δr/r)', fontsize=12)
    ax1.set_ylabel('Reliability (1/FF)', fontsize=12)
    ax1.set_title('Sensitivity-Reliability Trade-off Space', fontsize=13, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('τ_I (ms)', fontsize=11)
    
    # 2. 综合性能指标
    ax2 = axes[0, 1]
    ax2.plot(bn_df['tau'], bn_df['composite_geometric'], 
             'o-', linewidth=2.5, markersize=8, color='purple', label='Geometric Mean')
    ax2.axvline(bn_df.loc[bn_df['composite_geometric'].idxmax(), 'tau'],
                color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    ax2.set_xlabel('τ (ms)', fontsize=12)
    ax2.set_ylabel('Composite Performance', fontsize=12)
    ax2.set_title('Overall Performance Index', fontsize=13, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. 归一化对比
    ax3 = axes[1, 0]
    sens_norm = bn_df['sensitivity'] / bn_df['sensitivity'].max()
    rel_norm = bn_df['reliability'] / bn_df['reliability'].max()
    
    width = 0.35
    x = np.arange(len(bn_df))
    
    ax3.bar(x - width/2, sens_norm, width, label='Sensitivity (norm)', 
            color='#377eb8', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, rel_norm, width, label='Reliability (norm)', 
            color='#e41a1c', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('τ (ms)', fontsize=12)
    ax3.set_ylabel('Normalized Score', fontsize=12)
    ax3.set_title('Normalized Metrics Comparison', fontsize=13, weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t:.1f}" for t in bn_df['tau']], rotation=45)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 差异热图
    ax4 = axes[1, 1]
    
    # 计算每个点到理想点 (1, 1) 的距离
    distances = np.sqrt(
        (1 - sens_norm)**2 + (1 - rel_norm)**2
    )
    
    colors = plt.cm.RdYlGn_r(distances / distances.max())
    ax4.barh(bn_df['tau'], distances, color=colors, edgecolor='black')
    ax4.set_xlabel('Distance to Ideal Point', fontsize=12)
    ax4.set_ylabel('τ (ms)', fontsize=12)
    ax4.set_title('Deviation from Ideal (S=1, R=1)', fontsize=13, weight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    out_path = Path(FIGURE_DIR) / 'enhanced_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ 增强分析图已保存到 {out_path}")


def suggest_next_experiments(bn_df):
    """
    基于结果建议下一步实验
    """
    print("\n" + "="*60)
    print("实验建议")
    print("="*60)
    
    # 找到有趣的区域
    max_sens_tau = bn_df.loc[bn_df['sensitivity'].idxmax(), 'tau']
    
    print(f"\n1. 高分辨率扫描建议:")
    print(f"   在 τ_I = {max_sens_tau-1:.1f} 到 {max_sens_tau+1:.1f} ms 之间")
    print(f"   以 0.1 ms 步长进行细扫描")
    
    print(f"\n2. 机制探索:")
    print(f"   - 记录 τ_I = {max_sens_tau:.1f} ms 处的完整脉冲序列")
    print(f"   - 分析 E-I 平衡的瞬态破缺")
    print(f"   - 测量雪崩分布和功率谱")
    
    print(f"\n3. 参数鲁棒性测试:")
    print(f"   - 改变连接概率 p")
    print(f"   - 改变 E/I 比例")
    print(f"   - 改变网络规模")
    
    print("\n4. 与实验数据对比:")
    print("   - 如果有皮层数据,比较时间尺度")
    print("   - 验证是否存在类似的 trade-off")


# 使用示例
if __name__ == "__main__":
    # 加载数据
    import pickle
    
    bn_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bn.pkl'
    bp_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bp.pkl'
    if not bn_path.exists() or not bp_path.exists():
        print(f"数据文件不存在:\n  bn: {bn_path}\n  bp: {bp_path}\n请先运行 experiments/sensitivity_reliability.py 生成数据。")
        raise FileNotFoundError(str(bn_path if not bn_path.exists() else bp_path))
    with open(bn_path, 'rb') as f:
        bn_data = pickle.load(f)
    
    with open(bp_path, 'rb') as f:
        bp_data = pickle.load(f)
    
    # 分析
    bn_df = analyze_results(bn_data, bp_data)
    
    # 可视化
    plot_enhanced_analysis(bn_df)
    
    # 建议
    suggest_next_experiments(bn_df)
