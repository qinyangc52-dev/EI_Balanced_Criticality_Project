# 创建文件：check_data_consistency.py

import numpy as np
from pathlib import Path

raw_dir = Path('data/raw')
files = sorted(raw_dir.glob('spikes_*.npz'))

print("\n发放率检查:")
print("tau   Rate(Hz)  Spikes   Duration")
print("-" * 45)

rates = []
for file in files:
    tau = float(file.stem.split('_')[1])
    
    data = np.load(file)
    spike_times = data['spike_times']
    duration = float(data['duration'])
    
    # 去除warm-up
    spikes_after_warmup = spike_times[spike_times > 1000]
    n_spikes = len(spikes_after_warmup)
    
    # 计算发放率
    rate = n_spikes / (duration - 1000) / 800 * 1000
    rates.append(rate)
    
    print(f"{tau:5.1f} {rate:7.2f}   {n_spikes:7d}  {duration:8.1f}")

# 分析趋势
print("\n趋势分析:")
for i in range(1, len(rates)):
    change = rates[i] - rates[i-1]
    direction = "↑" if change > 0 else "↓"
    print(f"  tau {files[i-1].stem.split('_')[1]} → {files[i].stem.split('_')[1]}: {direction} {abs(change):.3f} Hz")

# 检查是否有合理的峰值
max_idx = np.argmax(rates)
max_tau = float(files[max_idx].stem.split('_')[1])
print(f"\n当前最大发放率: {rates[max_idx]:.2f} Hz @ tau={max_tau:.1f}ms")

if max_tau < 6:
    print("❌ 异常：峰值在次临界区（tau<6），应该在8-10ms")
elif max_tau > 12:
    print("❌ 异常：峰值在超临界区（tau>12），应该在8-10ms")
else:
    print("✓ 峰值位置合理")

# 检查单调性（前半部分应该上升）
early_rates = rates[:4]  # 前4个tau
if early_rates[-1] > early_rates[0]:
    print("\n✓ 早期趋势正常（上升）")
else:
    print(f"\n❌ 早期趋势异常：{early_rates[0]:.2f} → {early_rates[-1]:.2f}")