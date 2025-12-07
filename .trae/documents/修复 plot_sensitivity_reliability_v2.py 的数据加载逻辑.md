## 问题原因
- 可视化脚本当前仅尝试加载 `sensitivity_reliability_balanced.pkl`、`sensitivity_reliability.pkl` 或 `sr_results.pkl`。
- 实际生成的数据文件是 `data/processed/sensitivity_reliability_bn.pkl` 和 `..._bp.pkl`，因此脚本找不到文件并提示未生成。
- 同时，已生成的数据结构键为 `{'tau', 'sensitivity', 'reliability'}`，而脚本兼容分支期望 `{'tau_values', 'sensitivity', 'reliability'}`。

## 修改方案
- 在 `visualization/plot_sensitivity_reliability_v2.py` 的 `load_sr_data()` 中：
  - 先检测并加载 `sensitivity_reliability_bn.pkl` 与 `sensitivity_reliability_bp.pkl`。
  - 将 `bn.pkl` 内容重组为脚本期望格式：
    - `balanced = {'tau_values': bn['tau'], 'sensitivity': bn['sensitivity'], 'reliability': bn['reliability']}`
    - 返回 `{'balanced': balanced}`，使后续绘图逻辑无需改动。
  - 若未找到，则回退到原来的三个旧文件名检查。
- 保持现有 `PROCESSED_DATA_DIR` 读取路径，避免运行目录差异导致的查找失败。

## 代码改动要点
- 更新 `load_sr_data()`：
```
from pathlib import Path
from configs.model_config import PROCESSED_DATA_DIR
from utils.io_manager import load_pkl

def load_sr_data():
    bn_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bn.pkl'
    bp_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bp.pkl'
    if bn_path.exists():
        bn = load_pkl(bn_path)
        balanced = {
            'tau_values': bn.get('tau', bn.get('tau_values', [])),
            'sensitivity': bn.get('sensitivity', []),
            'reliability': bn.get('reliability', []),
        }
        return {'balanced': balanced}
    # 旧文件名回退
    for name in ['sensitivity_reliability_balanced.pkl','sensitivity_reliability.pkl','sr_results.pkl']:
        p = Path(PROCESSED_DATA_DIR) / name
        if p.exists():
            return load_pkl(p)
    print('错误: 未找到sensitivity/reliability数据文件')
    print('请先运行: python experiments/sensitivity_reliability.py')
    return None
```

## 验证步骤
- 在工程根目录运行：`python visualization/plot_sensitivity_reliability_v2.py`
- 预期：正常加载 `bn.pkl`，生成并保存图到 `FIGURE_DIR/sensitivity_reliability_comprehensive.png`，无“未找到数据文件”的提示。

## 影响范围
- 仅影响该可视化脚本的数据加载逻辑，不变更数据生成流程。
- 兼容旧文件命名，避免破坏现有使用习惯。