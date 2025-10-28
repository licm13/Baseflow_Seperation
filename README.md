# 基流分割（Baseflow Separation）工程化项目指南

> 本仓库在原有原型代码的基础上进行了全面的结构化改造，旨在以正式工程项目的标准管理基流分割算法、批处理脚本与示例资源。本文档详细说明目录划分、核心模块职责、数据流向、批处理流程以及二次开发注意事项。

## 🔧 项目概览

- **项目目标**：提供基流（Baseflow）分割的算法实现、参数估计工具及批量处理流程，支撑科研或业务场景下的大规模流量分割任务。
- **语言环境**：Python ≥ 3.8（推荐 3.10）。
- **核心特性**：
  - 12 种经典基流分割方法的统一封装。
  - 自动化的衰退系数、经验参数估计模块。
  - 兼容多站点数据的批处理脚本与示例。
  - 完整的输入/输出数据结构说明与中文操作手册。

## 📁 目录结构与文件分类

```
Baseflow_Seperation/
├── src/                          # Python 包源码（统一以 src 布局管理）
│   └── baseflow/
│       ├── __init__.py           # 包导出与示例数据引用
│       ├── comparision.py        # 评价指标与严格基流判定
│       ├── methods/              # 12 种基流分割方法的具体实现
│       ├── param_estimate.py     # 衰退系数估算与参数标定逻辑
│       ├── separation.py         # 面向单站点/多站点的主分割流程
│       └── utils.py              # 数据清洗、投影变换、工具函数
│
├── scripts/
│   ├── __init__.py               # 脚本包声明（可空）
│   ├── run_all_methods.py        # 统一入口：批量执行所有方法
│   └── batch/
│       ├── daily_batch_run.py    # 逐日尺度批处理脚本
│       ├── monthly_batch_run.py  # 月尺度批处理脚本
│       └── long_term_batch_run.py# 长时序批处理脚本
│
├── examples/
│   └── run_all_methods.py        # 教程式示例，展示包级 API 的调用方式
│
├── docs/
│   └── batch_processing_manual.md# 中文批处理系统说明书
│
├── data/                         # 预留原始/中间数据目录（默认空）
├── requirements.txt              # 运行时依赖声明
├── setup.py                      # 包安装配置（遵循 src 布局）
└── README.md                     # 本说明文档
```

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

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# （可选）以开发模式安装 baseflow 包
pip install -e .

# 运行示例脚本
PYTHONPATH=src python examples/run_all_methods.py
```

> `PYTHONPATH=src` 仅在未通过 `pip install -e .` 安装包时需要；安装后可直接 `python -m scripts.run_all_methods`。

## 🧪 测试建议

- 建议针对自有数据编写 `pytest` 测试，存放于 `tests/` 目录（需自行创建）。
- 可通过构造小规模流量序列，对 `baseflow.separation.single` 的输出维度、非负性、KGE 计算等进行断言。

## 📝 注释与编码规范

- 源码使用 **PEP 8** 风格，关键函数提供完整 docstring。
- 脚本中的主流程拆分为函数，以便单元测试和调度调用。
- 所有路径均推荐使用 `pathlib.Path`，便于跨平台部署。

## 📚 二次开发指引

1. **新增分割方法**：在 `src/baseflow/methods/` 中新增文件，按现有函数签名实现，并在 `baseflow/__init__.py` 中显式导出。
2. **扩展评价指标**：在 `comparision.py` 中新增函数，并在 `separation.single` 中根据需要挂载。
3. **自定义批处理流程**：复制 `scripts/batch/` 中的脚本，调整输入输出路径或调度逻辑即可。

---

如需进一步的工程化部署、容器化运行或流水线集成，可在 `docs/` 目录下新增详细设计文档；欢迎基于当前目录结构持续拓展。
