## 问题原因
- 报错来自在不同工作目录运行脚本导致的相对路径不一致。当前脚本使用相对路径，若从工程根目录运行会找不到 `../data/processed/...`。
- 正确数据文件实际位于：`c:\BrainpyEi\EI_Balanced_Criticality_Project\data\processed\sensitivity_reliability_bn.pkl` 与 `..._bp.pkl`。

## 修改方案
- 在 `visualization/plot_response.py` 中统一使用工程配置的绝对路径，避免与运行目录相关：
  - 引入 `from configs.model_config import PROCESSED_DATA_DIR`
  - 使用 `Path(PROCESSED_DATA_DIR)/'sensitivity_reliability_bn.pkl'` 与 `Path(PROCESSED_DATA_DIR)/'sensitivity_reliability_bp.pkl'`
- 增加文件存在性检查，给出更清晰的错误信息：
  - 若文件不存在，打印实际尝试的路径并提示先运行 `experiments/sensitivity_reliability.py` 生成数据。
- 保持此前已修复的列名一致性（使用 `'tau'` 而非 `'tau_I'`）。

## 具体代码调整
- 顶部新增：
  - `from pathlib import Path`
  - `from configs.model_config import PROCESSED_DATA_DIR`
- 读取数据部分替换为：
```
from pathlib import Path
from configs.model_config import PROCESSED_DATA_DIR
bn_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bn.pkl'
bp_path = Path(PROCESSED_DATA_DIR) / 'sensitivity_reliability_bp.pkl'
if not bn_path.exists() or not bp_path.exists():
    print(f"数据文件不存在:\n  bn: {bn_path}\n  bp: {bp_path}\n请先运行 experiments/sensitivity_reliability.py 生成处理数据。")
    raise FileNotFoundError(str(bn_path if not bn_path.exists() else bp_path))
with open(bn_path, 'rb') as f: bn_data = pickle.load(f)
with open(bp_path, 'rb') as f: bp_data = pickle.load(f)
```

## 验证步骤
- 在工程根目录与 `visualization` 目录分别运行：
  - `python visualization/plot_response.py`
  - `cd visualization; python plot_response.py`
- 预期：脚本均可加载数据、生成并保存图到 `visualization/figures/enhanced_analysis.png`，无路径报错。

## 兼容性与影响
- 路径改为基于工程配置的绝对路径，运行位置不再影响脚本。
- 不改动数据结构，且列名统一为 `'tau'`，与已生成的 `pkl` 文件键一致。