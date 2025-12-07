#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键生成所有可视化图表
运行: python visualization/generate_all_figures.py
"""

import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import FIGURE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def check_data_availability():
    """
    检查必要的数据文件是否存在
    """
    print("\n" + "="*70)
    print("检查数据文件...")
    print("="*70)
    
    issues = []
    
    # 检查原始数据
    raw_dir = Path(RAW_DATA_DIR)
    raw_files = list(raw_dir.glob("spikes_*.npz"))
    
    if len(raw_files) == 0:
        issues.append("❌ 未找到原始spikes数据")
        issues.append("   运行: python main.py --all")
    else:
        print(f"✓ 找到 {len(raw_files)} 个原始数据文件")
    
    # 检查处理后数据
    proc_dir = Path(PROCESSED_DATA_DIR)
    proc_files = list(proc_dir.glob("avalanche_stats_*.pkl"))
    
    if len(proc_files) == 0:
        issues.append("❌ 未找到雪崩统计数据")
        issues.append("   运行: python main.py --all")
    else:
        print(f"✓ 找到 {len(proc_files)} 个统计数据文件")
    
    # 检查S&R数据
    sr_files = list(proc_dir.glob("*sensitivity*reliability*.pkl"))
    
    if len(sr_files) == 0:
        issues.append("⚠️  未找到Sensitivity/Reliability数据")
        issues.append("   运行: python experiments/sensitivity_reliability.py")
        print("⚠️  Sensitivity/Reliability数据缺失（可选）")
    else:
        print(f"✓ 找到Sensitivity/Reliability数据")
    
    print("="*70)
    
    if issues:
        print("\n问题:")
        for issue in issues:
            print(issue)
        print()
        return False
    
    return True


def generate_figure(script_name, description, optional=False):
    """
    运行指定的可视化脚本
    """
    print(f"\n{'='*70}")
    print(f"生成: {description}")
    print(f"{'='*70}")
    
    script_path = Path(project_root) / "visualization" / script_name
    
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✓ 成功")
            # 打印脚本的输出
            if result.stdout:
                print(result.stdout)
            return True
        else:
            if optional:
                print(f"⚠️  跳过（可选）")
                if result.stderr:
                    print(result.stderr)
            else:
                print(f"❌ 失败")
                if result.stderr:
                    print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ 超时（>2分钟）")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    """
    主函数：生成所有图表
    """
    print("\n" + "🎨"*35)
    print("一键生成所有可视化图表")
    print("🎨"*35)
    
    # 检查数据
    if not check_data_availability():
        print("\n请先运行必要的实验获取数据")
        return
    
    print("\n" + "="*70)
    print("开始生成图表...")
    print("="*70)
    
    results = {}
    
    # 1. 综合临界性图（必须）
    results['criticality'] = generate_figure(
        'plot_criticality_comprehensive.py',
        'Figure 1: 综合临界性证据（光栅图+幂律+指标）',
        optional=False
    )
    
    # 2. Sensitivity & Reliability（可选）
    results['sr'] = generate_figure(
        'plot_sensitivity_reliability_v2.py',
        'Figure 2: Sensitivity & Reliability共存',
        optional=True
    )
    
    # 3. 生成简化版（快速预览）
    results['simple'] = generate_figure(
        'plot_criticality_comprehensive.py --simple',
        'Simple: 简化版（3子图）',
        optional=True
    )
    
    # 汇总
    print("\n" + "="*70)
    print("生成完成！")
    print("="*70)
    
    print("\n生成结果:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    # 列出生成的文件
    print(f"\n生成的图表位于: {FIGURE_DIR}")
    
    fig_dir = Path(FIGURE_DIR)
    if fig_dir.exists():
        figures = list(fig_dir.glob("*.png"))
        if figures:
            print("\n可用图表:")
            for fig in sorted(figures):
                print(f"  - {fig.name}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()