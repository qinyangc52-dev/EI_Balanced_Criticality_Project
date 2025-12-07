# experiments/monitor_progress.py
"""
实时监控模拟进度
在另一个终端运行: python experiments/monitor_progress.py
"""

import time
from pathlib import Path
from datetime import datetime, timedelta

# 你的自定义tau列表
TAU_LIST = [2.0, 4.0, 6.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    try:
        return filepath.stat().st_size / 1024 / 1024
    except:
        return 0

def check_progress():
    """检查当前进度"""
    completed_raw = []
    completed_processed = []
    
    for tau in TAU_LIST:
        raw_file = RAW_DIR / f"spikes_{tau:.1f}.npz"
        processed_file = PROCESSED_DIR / f"avalanche_stats_{tau:.1f}.pkl"
        
        if raw_file.exists():
            completed_raw.append((tau, get_file_size_mb(raw_file)))
        
        if processed_file.exists():
            completed_processed.append((tau, get_file_size_mb(processed_file)))
    
    return completed_raw, completed_processed

def format_time(seconds):
    """格式化时间"""
    return str(timedelta(seconds=int(seconds)))

def main():
    """主监控循环"""
    print("\n" + "="*70)
    print("模拟进度监控")
    print("="*70)
    print(f"目标: {len(TAU_LIST)} 个tau值")
    print(f"tau列表: {TAU_LIST}")
    print("="*70 + "\n")
    
    start_time = time.time()
    last_completed = 0
    
    try:
        while True:
            completed_raw, completed_processed = check_progress()
            n_raw = len(completed_raw)
            n_processed = len(completed_processed)
            
            # 清屏效果（打印多个换行）
            print("\033[H\033[J", end='')  # ANSI清屏（部分终端支持）
            
            # 打印标题
            print("\n" + "="*70)
            print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            elapsed = time.time() - start_time
            print(f"已运行: {format_time(elapsed)}")
            print("="*70)
            
            # 进度条
            progress = n_processed / len(TAU_LIST)
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\n总进度: [{bar}] {n_processed}/{len(TAU_LIST)} ({progress*100:.1f}%)")
            
            # 详细状态
            print("\n详细状态:")
            print("-"*70)
            print(f"{'tau_d_I':>8} {'模拟':>10} {'分析':>10} {'文件大小':>15}")
            print("-"*70)
            
            for tau in TAU_LIST:
                raw_status = "✓" if any(t == tau for t, _ in completed_raw) else "..."
                processed_status = "✓" if any(t == tau for t, _ in completed_processed) else "..."
                
                # 获取文件大小
                raw_size = next((s for t, s in completed_raw if t == tau), 0)
                
                size_str = f"{raw_size:.1f}MB" if raw_size > 0 else "-"
                
                print(f"{tau:>8.1f} {raw_status:>10} {processed_status:>10} {size_str:>15}")
            
            # 预估剩余时间
            if n_processed > last_completed and n_processed > 0:
                avg_time_per_tau = elapsed / n_processed
                remaining = (len(TAU_LIST) - n_processed) * avg_time_per_tau
                eta = datetime.now() + timedelta(seconds=remaining)
                
                print("\n" + "-"*70)
                print(f"平均每个tau: {format_time(avg_time_per_tau)}")
                print(f"预计剩余: {format_time(remaining)}")
                print(f"预计完成: {eta.strftime('%H:%M:%S')}")
                last_completed = n_processed
            
            # 完成检查
            if n_processed == len(TAU_LIST):
                print("\n" + "="*70)
                print("🎉 所有模拟已完成！")
                print("="*70)
                print(f"\n总用时: {format_time(elapsed)}")
                print(f"\n下一步:")
                print(f"  python experiments/validate_all_new_results.py")
                print("="*70 + "\n")
                break
            
            # 等待30秒后刷新
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print(f"当前进度: {n_processed}/{len(TAU_LIST)}")

if __name__ == "__main__":
    main()