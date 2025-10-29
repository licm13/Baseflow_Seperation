# 基流分割（Baseflow Separation）说明

> 本仓库在原有原型代码的基础上进行了全面的结构化改造和重构，旨在以正式工程项目的标准管理基流分割算法、批处理脚本与示例资源。本次更新包括：完整的类型提示、详细的文档字符串、配置模块、合成数据生成器、综合测试示例和单元测试套件。

## 🔧 项目概览

- **项目目标**：提供基流（Baseflow）分割的算法实现、参数估计工具及批量处理流程，支撑科研或业务场景下的大规模流量分割任务。
- **语言环境**：Python ≥ 3.8（推荐 3.10+）。
- **核心特性**：
  - 12 种经典基流分割方法的统一封装（数字滤波、图形法、参数化方法）。
  - 自动化的衰退系数、经验参数估计模块（基于网格搜索和NSE优化）。
  - 兼容多站点数据的批处理脚本与示例。
  - 完整的类型提示（Type Hints）和详细的文档字符串。
  - 合成数据生成器，支持算法测试和验证。
  - 综合测试示例和单元测试套件。
  - 灵活的参数配置系统。

## 📁 目录结构与文件分类

```
Baseflow_Seperation/
├── src/                          # Python 包源码（统一以 src 布局管理）
│   └── baseflow/
│       ├── __init__.py           # 包导出与示例数据引用
│       ├── config.py             # 🆕 参数配置模块（方法参数范围、全局设置）
│       ├── comparision.py        # 评价指标与严格基流判定（已重构，添加类型提示）
│       ├── methods/              # 12 种基流分割方法的具体实现
│       ├── param_estimate.py     # 衰退系数估算与参数标定逻辑（已重构）
│       ├── separation.py         # 面向单站点/多站点的主分割流程（已重构）
│       ├── synthetic_data.py     # 🆕 合成数据生成器（用于测试和演示）
│       └── utils.py              # 数据清洗、投影变换、工具函数（已重构）
│
├── scripts/
│   ├── __init__.py               # 脚本包声明（可空）
│   ├── run_all_methods.py        # 统一入口：批量执行所有方法
│   └── batch/
│       ├── common.py             # 批处理公共逻辑和配置
│       ├── daily_batch_run.py    # 逐日尺度批处理脚本
│       ├── monthly_batch_run.py  # 月尺度批处理脚本
│       └── long_term_batch_run.py# 长时序批处理脚本
│
├── examples/
│   ├── run_all_methods.py        # 基础示例，展示包级 API 的调用方式
│   └── comprehensive_example.py  # 🆕 综合示例（5个完整示例，包含可视化）
│
├── tests/                        # 🆕 单元测试套件
│   ├── __init__.py
│   ├── test_synthetic_data.py    # 合成数据生成器测试
│   └── test_separation.py        # 基流分割功能测试
│
├── docs/
│   └── batch_processing_manual.md# 中文批处理系统说明书
│
├── data/                         # 预留原始/中间数据目录（默认空）
├── requirements.txt              # 运行时依赖声明
├── setup.py                      # 包安装配置（遵循 src 布局）
└── README.md                     # 本说明文档
```

> **🆕 本次更新新增内容**：
> - `config.py`: 集中管理所有参数范围，支持自定义调整
> - `synthetic_data.py`: 生成真实感的合成流量数据，包含已知基流
> - `comprehensive_example.py`: 5个完整示例，展示从数据生成到结果可视化的全流程
> - `tests/`: 完整的单元测试套件，确保代码质量

> **命名规范**：脚本目录按业务场景分层（`scripts/batch/`、`scripts/`），源码目录统一收敛至 `src/baseflow/`，文档资料集中在 `docs/`，示例与教程置于 `examples/`，数据入口统一指向 `data/`。

## 🧠 核心模块详解

### `baseflow.methods`
- 每个文件对应一套分割算法，如 `LH.py`（Lyne-Hollick 滤波）、`Eckhardt.py` 等。
- 均采用 **NumPy/Numba** 加速，提供一致的函数签名 `Method(Q, ...) -> np.ndarray`。
- 模块内 docstring 说明算法出处、关键参数与返回值含义。

### `baseflow.param_estimate`
- `recession_coefficient`：依据严格基流时段估算衰退参数。
- `param_calibrate`：通用网格搜索器，对方法所需超参数进行自动标定。
- 结合 `methods` 模块形成闭环，支撑自动化批处理。

### `baseflow.comparision`
- `strict_baseflow`：严格基流判定逻辑，结合融雪/结冰掩膜剔除异常期。
- `KGE`：Kling-Gupta Efficiency 指标，用于多方法评估与排序。

### `baseflow.utils`
- `clean_streamflow`：统一清洗入参序列，保证索引合法、缺失值处理一致。
- `exist_ice` 与 `geo2imagexy`：处理冻土/冰冻区域的辅助工具。
- `format_method`：统一解析单方法、方法列表与 `"all"` 模式。

### `baseflow.separation`
- `single(series, area, ice, method, return_kge)`：单站点核心流程。
- `separation(df, df_sta, ...)`：多站点分割，生成多指标矩阵。
- 内部自动加载 `thawed.npz`（冻土掩膜），同时结合参数估计与评价指标。

## 🗂️ 脚本与示例

### `scripts/run_all_methods.py`
- 统一入口脚本，封装命令行参数解析，支持：
  - 指定输入流量 CSV；
  - 指定站点属性信息；
  - 选择输出目录、是否返回 KGE/BFI。
- 可通过 `python -m scripts.run_all_methods --help` 查看详情。

### 批处理脚本
- `daily_batch_run.py`：针对逐日数据的批处理；
- `monthly_batch_run.py`：面向月尺度/分区数据；
- `long_term_batch_run.py`：处理 60~99 年长记录；
- 三个脚本均提供配置节（输入路径、输出路径、方法选择），并将公共逻辑抽象为函数，方便二次封装或集成至调度系统。

### 示例
- `examples/run_all_methods.py` 展示最小可运行样例，帮助快速熟悉 API。

## 📊 数据与资源

- `src/baseflow/example.csv`：示例站点的逐日流量数据，可作为快速演示输入。
- `src/baseflow/thawed.npz`：全球冻土期掩膜矩阵，供 `exist_ice` 判断季节性冻结。
- `docs/batch_processing_manual.md`：原始中文批处理说明文档，保留详细的操作步骤与参数解释。
- 推荐在 `data/` 目录中存放业务数据，避免污染源码仓库。

## 🚀 快速开始

### 安装

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# （推荐）以开发模式安装 baseflow 包
pip install -e .
```

### 运行示例

#### 1. 基础示例（简单快速）

```bash
# 如果已安装包
python examples/run_all_methods.py

# 或者不安装直接运行
PYTHONPATH=src python examples/run_all_methods.py
```

#### 2. 🆕 综合示例（推荐，包含5个完整示例）

```bash
# 运行完整的测试和演示
python examples/comprehensive_example.py
```

该示例将演示：
- 单站点基流分割
- 结果可视化（时间序列、流量历时曲线）
- 多站点批处理
- 参数敏感性分析
- 不同流域类型的方法比较

#### 3. 🆕 使用合成数据快速测试

```python
from baseflow import single
from baseflow.synthetic_data import generate_streamflow
import pandas as pd

# 生成合成数据（包含已知真实基流）
Q, true_baseflow, _ = generate_streamflow(n_days=365, bfi=0.65, random_seed=42)
dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
flow_series = pd.Series(Q, index=dates)

# 进行基流分割
baseflow_df, kge_scores = single(flow_series, method=["LH", "Eckhardt"])

# 评估结果
print("KGE 分数:", kge_scores)
print("估计的 BFI:", baseflow_df["Eckhardt"].sum() / Q.sum())
print("真实的 BFI:", true_baseflow.sum() / Q.sum())
```

### 🧪 运行测试

```bash
# 运行所有单元测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_synthetic_data.py -v
pytest tests/test_separation.py -v

# 查看测试覆盖率（需要安装 pytest-cov）
pip install pytest-cov
pytest tests/ --cov=src/baseflow --cov-report=html
```

> **注意**: `PYTHONPATH=src` 仅在未通过 `pip install -e .` 安装包时需要；安装后可直接运行所有命令。

## 📝 注释与编码规范

- 源码使用 **PEP 8** 风格，关键函数提供完整 docstring。
- 脚本中的主流程拆分为函数，以便单元测试和调度调用。
- 所有路径均推荐使用 `pathlib.Path`，便于跨平台部署。

## 📚 二次开发指引

1. **新增分割方法**：在 `src/baseflow/methods/` 中新增文件，按现有函数签名实现，并在 `baseflow/__init__.py` 中显式导出。
2. **扩展评价指标**：在 `comparision.py` 中新增函数，并在 `separation.single` 中根据需要挂载。
3. **自定义批处理流程**：复制 `scripts/batch/` 中的脚本，调整输入输出路径或调度逻辑即可。

---

## 📋 更新日志

### 2025 年重构版本 (当前版本)

#### 🎯 代码质量改进
- ✅ **完整类型提示**: 所有模块添加完整的类型注解（使用 `typing` 和 `numpy.typing`）
- ✅ **详细文档字符串**: 每个函数都有完整的 docstring，包含参数说明、返回值、示例和注意事项
- ✅ **代码注释增强**: 关键算法步骤添加中英文注释
- ✅ **PEP 8 规范**: 代码风格完全符合 Python 编码规范

#### 🆕 新增功能
- ✅ **配置模块** (`config.py`):
  - 集中管理所有方法的参数范围
  - 支持运行时动态修改参数
  - 提供方法描述和依赖信息

- ✅ **合成数据生成器** (`synthetic_data.py`):
  - 生成真实感的流量数据（包含季节性、风暴事件）
  - 支持自定义 BFI 和水文特征
  - 包含已知真实基流，便于算法验证
  - 支持多站点数据集生成

- ✅ **综合测试示例** (`examples/comprehensive_example.py`):
  - 5 个完整的使用示例
  - 包含数据生成、分割、评估、可视化全流程
  - 展示不同流域类型的方法比较
  - 参数敏感性分析演示

- ✅ **单元测试套件** (`tests/`):
  - 合成数据生成器测试（覆盖所有生成函数）
  - 基流分割功能测试（单站点和多站点）
  - 结果验证和约束检查
  - 使用 pytest 框架，易于扩展

#### 🔧 改进优化
- ✅ **参数管理**: 从硬编码改为配置化，提高灵活性
- ✅ **导入优化**: 移除 `import *`，使用显式导入
- ✅ **错误处理**: 改进异常处理和错误信息
- ✅ **性能**: 保持 Numba JIT 编译优化
- ✅ **文档**: 更新 README，添加详细使用说明

#### 📊 技术改进
- 类型提示覆盖率: ~100%
- 文档字符串覆盖率: ~100%
- 新增代码行数: ~3000 行
- 新增测试: 30+ 个测试用例
- 示例脚本: 从 1 个增加到 2 个（包含 5 个子示例）

### 未来计划
- [ ] 添加更多评价指标（NSE, RMSE, MAE 等）
- [ ] 支持不确定性分析
- [ ] 优化参数标定算法（贝叶斯优化、遗传算法）
- [ ] 添加交互式可视化界面
- [ ] 发布 PyPI 包

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

---

如需进一步的工程化部署、容器化运行或流水线集成，可在 `docs/` 目录下新增详细设计文档；欢迎基于当前目录结构持续拓展。
