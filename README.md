# 🌊 Baseflow Separation Toolkit | 基流分割工具包

## 📋 Table of Contents | 目录

- [Project Overview | 项目概述](#-project-overview--项目概述)
- [Quick Start | 快速开始](#-quick-start--快速开始)
- [File Structure | 目录结构](#-file-structure--目录结构)
- [Core Code Navigation | 核心代码导航](#-core-code-navigation--核心代码导航)
- [Code Reading Guide | 源码阅读指南](#-code-reading-guide--源码阅读指南)
- [Business Logic Mapping | 业务场景映射](#-business-logic-mapping--业务场景映射)
- [Development Guide | 开发指南](#-development-guide--开发指南)
- [Testing | 测试](#-testing--测试)
- [Contributing | 贡献](#-contributing--贡献)

---

## 🎯 Project Overview | 项目概述

### English Version

**One-sentence summary**: A scientific Python toolkit for separating river streamflow into baseflow (groundwater contribution) and quickflow (surface runoff) using 12 classical hydrological algorithms with automated parameter estimation and batch processing capabilities.

**Core Technology Stack**:
- **Language**: Python 3.8+ (Recommended: 3.10+)
- **Performance**: Numba JIT compilation for computational efficiency
- **Data Processing**: NumPy, Pandas for time series analysis
- **Parallel Computing**: Joblib for multi-station batch processing
- **Scientific Computing**: SciKit-Learn for parameter optimization
- **Visualization**: Matplotlib (optional)

**Key Features**:
- ✅ 12 baseflow separation methods spanning 3 algorithm families
- ✅ Automatic parameter estimation using recession analysis
- ✅ Grid search calibration with Nash-Sutcliffe Efficiency (NSE)
- ✅ Global frozen period detection (permafrost consideration)
- ✅ Batch processing with parallel computing support
- ✅ Synthetic data generator for algorithm validation
- ✅ Comprehensive evaluation metrics (KGE, BFI, NSE)
- ✅ Educational Jupyter notebook with interactive widgets

### 中文版本

**一句话概括**: 这是一个科学的 Python 工具包,用于将河流流量分解为基流(地下水贡献)和快速流(地表径流),提供 12 种经典水文算法,支持自动参数估计和批量处理。

**核心技术栈**:
- **语言**: Python 3.8+ (推荐 3.10+)
- **性能优化**: Numba JIT 编译以提高计算效率
- **数据处理**: NumPy、Pandas 用于时间序列分析
- **并行计算**: Joblib 用于多站点批量处理
- **科学计算**: SciKit-Learn 用于参数优化
- **可视化**: Matplotlib (可选)

**核心功能**:
- ✅ 12 种基流分割方法,涵盖 3 个算法家族
- ✅ 使用退水分析自动估计参数
- ✅ 使用 Nash-Sutcliffe 效率系数(NSE)进行网格搜索校准
- ✅ 全球冻土期检测(考虑永久冻土)
- ✅ 支持并行计算的批量处理
- ✅ 用于算法验证的合成数据生成器
- ✅ 综合评估指标(KGE、BFI、NSE)
- ✅ 带交互式小部件的教学 Jupyter notebook

---

## 🚀 Quick Start | 快速开始

### Installation | 安装

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/licm13/Baseflow_Seperation.git
cd Baseflow_Seperation

# Install dependencies | 安装依赖
pip install -r requirements.txt

# Install package in development mode | 以开发模式安装包
pip install -e .
```

### Basic Usage | 基本使用

```python
import pandas as pd
from baseflow import single, separation

# Example 1: Single station analysis | 示例 1: 单站点分析
# Load your streamflow data | 加载流量数据
flow = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)['flow']

# Separate baseflow using Lyne-Hollick filter | 使用 Lyne-Hollick 滤波器分离基流
baseflow_df, kge_scores = single(
    flow,
    method=["LH", "Eckhardt"],
    area=1000  # drainage area in km² | 流域面积(平方公里)
)

# Example 2: Multi-station batch processing | 示例 2: 多站点批量处理
# Load multi-station data | 加载多站点数据
df = pd.read_csv('multi_station.csv', index_col=0, parse_dates=True)
station_info = pd.DataFrame({
    'area': [1000, 2000, 1500],
    'lon': [-120, -119, -121],
    'lat': [45, 46, 44]
})

# Process all stations in parallel | 并行处理所有站点
results, bfi, kge = separation(
    df,
    df_sta=station_info,
    method="all",  # use all 12 methods | 使用全部 12 种方法
    return_bfi=True,
    n_jobs=-1  # use all CPU cores | 使用所有 CPU 核心
)
```

---

## 📁 File Structure | 目录结构

### Overview Diagram | 目录概览图

```
Baseflow_Seperation/
│
├── 📦 src/baseflow/              # Core package source code | 核心包源代码
│   ├── __init__.py               # Public API exports | 公共 API 导出
│   ├── config.py                 # Centralized configuration | 集中配置
│   ├── separation.py             # High-level separation APIs | 高级分割 API
│   ├── param_estimate.py         # Parameter estimation | 参数估计
│   ├── comparision.py            # Evaluation metrics | 评估指标
│   ├── utils.py                  # Utility functions | 实用函数
│   ├── synthetic_data.py         # Test data generator | 测试数据生成器
│   ├── example.csv               # Sample dataset | 示例数据集
│   ├── thawed.npz                # Global permafrost mask | 全球永久冻土掩膜
│   └── methods/                  # Algorithm implementations | 算法实现
│       ├── _base.py              # Base class architecture | 基类架构
│       ├── _wrappers.py          # OOP wrappers | 面向对象包装器
│       ├── LH.py                 # Lyne-Hollick filter | Lyne-Hollick 滤波器
│       ├── Eckhardt.py           # Eckhardt two-parameter filter | Eckhardt 双参数滤波器
│       ├── UKIH.py               # UK Institute of Hydrology | 英国水文研究所方法
│       ├── Chapman.py            # Chapman filter | Chapman 滤波器
│       ├── CM.py                 # Combined method | 组合方法
│       ├── Local.py              # Local minimum (HYSEP) | 局部最小值法
│       ├── Fixed.py              # Fixed interval (HYSEP) | 固定间隔法
│       ├── Slide.py              # Sliding interval (HYSEP) | 滑动间隔法
│       ├── Boughton.py           # Boughton recursive | Boughton 递归法
│       ├── Furey.py              # Furey recession | Furey 退水法
│       ├── EWMA.py               # Exponential weighted MA | 指数加权移动平均
│       ├── Willems.py            # Willems method | Willems 方法
│       └── ChengBudykoML.py      # ML-based method | 基于机器学习的方法
│
├── 📜 scripts/                   # Batch processing scripts | 批处理脚本
│   ├── run_all_methods.py        # CLI entry point | 命令行入口
│   ├── baseflow_cli.py           # Advanced CLI tool | 高级命令行工具
│   └── batch/                    # Batch processing utilities | 批处理工具
│       ├── daily_batch_run.py    # Daily timescale | 日尺度处理
│       ├── monthly_batch_run.py  # Monthly timescale | 月尺度处理
│       └── long_term_batch_run.py # Long-term records | 长期记录处理
│
├── 📚 examples/                  # Comprehensive examples | 综合示例
│   ├── quick_test.py             # Fast validation | 快速验证
│   ├── interactive_explorer.py   # CLI exploration | 命令行探索
│   ├── comprehensive_example.py  # 5 complete workflows | 5 个完整工作流
│   ├── advanced_visualization.py # Publication figures | 出版级图表
│   ├── benchmark_performance.py  # Performance analysis | 性能分析
│   └── real_world_workflow.py    # Production workflow | 生产工作流
│
├── 🧪 tests/                     # Unit test suite | 单元测试套件
│   ├── test_separation.py        # Separation logic tests | 分割逻辑测试
│   ├── test_synthetic_data.py    # Data generation tests | 数据生成测试
│   └── test_lh_core.py           # LH filter tests | LH 滤波器测试
│
├── 📖 docs/                      # Documentation | 文档
│   ├── algorithm_details.md      # Mathematical formulas | 数学公式
│   └── batch_processing_manual.md # Batch guide (Chinese) | 批处理指南(中文)
│
├── 📊 Cheng-3D-Budyko/           # ML research sub-project | 机器学习研究子项目
│   ├── 01_data_preprocessing.py  # Data preparation | 数据准备
│   ├── 03_model_training.py      # Model training | 模型训练
│   └── utils.py                  # Research utilities | 研究工具
│
├── 📓 baseflow_tutorial_freshmen.ipynb  # Interactive tutorial | 交互式教程
├── ⚙️ config.yml                 # Workflow configuration | 工作流配置
├── 📦 setup.py                   # Package installation | 包安装配置
├── 📋 requirements.txt           # Dependencies | 依赖列表
└── 📄 README.md                  # This file | 本文件
```

### Detailed Directory Explanation | 目录详细说明

#### 📦 `src/baseflow/` - Core Package | 核心包

**English**: This is the heart of the project. Contains all core algorithms, APIs, and utilities for baseflow separation.

**中文**: 这是项目的核心。包含所有用于基流分割的核心算法、API 和工具。

**Key Dependencies | 关键依赖关系**:
- `separation.py` depends on → `methods/`, `param_estimate.py`, `comparision.py`
- `param_estimate.py` depends on → `methods/`, `comparision.py`
- All methods depend on → `_base.py`, `utils.py`

#### 📜 `scripts/` - Automation Scripts | 自动化脚本

**English**: Production-ready scripts for batch processing multiple stations or long-term datasets. Used for operational hydrology workflows.

**中文**: 用于批量处理多个站点或长期数据集的生产就绪脚本。用于业务水文工作流。

#### 📚 `examples/` - Learning Resources | 学习资源

**English**: Progressive examples from simple to advanced. Start here to understand usage patterns. Each example is self-contained and runnable.

**中文**: 从简单到高级的渐进式示例。从这里开始了解使用模式。每个示例都是独立的,可以直接运行。

#### 🧪 `tests/` - Quality Assurance | 质量保证

**English**: Pytest-based test suite ensuring code correctness. Run `pytest tests/` to validate your modifications.

**中文**: 基于 Pytest 的测试套件,确保代码正确性。运行 `pytest tests/` 来验证您的修改。

#### 📖 `docs/` - Technical Documentation | 技术文档

**English**: Deep-dive mathematical documentation for each algorithm. Essential for understanding theoretical foundations.

**中文**: 每个算法的深入数学文档。对于理解理论基础至关重要。

#### 📊 `Cheng-3D-Budyko/` - Research Module | 研究模块

**English**: Independent research project implementing ML-based baseflow separation using Budyko framework. Can be used as a 13th method.

**中文**: 独立的研究项目,使用 Budyko 框架实现基于机器学习的基流分割。可以作为第 13 种方法使用。

---

## 🧭 Core Code Navigation | 核心代码导航

### Tier 1: Entry Points (Start Here) | 第一层:入口点(从这里开始)

| File | Purpose | Importance | Lines of Code |
|------|---------|------------|---------------|
| `src/baseflow/separation.py` | 🌟🌟🌟🌟🌟 Main API for single/multi-station separation <br> 🌟🌟🌟🌟🌟 单站/多站分割的主要 API | **CRITICAL** | ~400 |
| `examples/comprehensive_example.py` | 🌟🌟🌟🌟 5 complete usage workflows <br> 🌟🌟🌟🌟 5 个完整的使用工作流 | **ESSENTIAL** | ~300 |
| `src/baseflow/__init__.py` | 🌟🌟🌟 Public API exports <br> 🌟🌟🌟 公共 API 导出 | **IMPORTANT** | ~50 |

**English**: These three files form the **Golden Triangle** of the codebase. Read them first to understand the overall architecture.

**中文**: 这三个文件构成了代码库的**黄金三角**。首先阅读它们以了解整体架构。

### Tier 2: Core Algorithms | 第二层:核心算法

| File | Algorithm Family | Calibration Required? |
|------|------------------|----------------------|
| `methods/LH.py` | Digital Filter <br> 数字滤波器 | ❌ No (β=0.925 default) <br> ❌ 否(默认 β=0.925) |
| `methods/Eckhardt.py` | Digital Filter <br> 数字滤波器 | ✅ Yes (BFImax, α) <br> ✅ 是(BFImax, α) |
| `methods/UKIH.py` | Digital Filter <br> 数字滤波器 | ❌ No (fixed rules) <br> ❌ 否(固定规则) |
| `methods/Local.py` | HYSEP Graphical <br> HYSEP 图解法 | ⚠️ Requires drainage area <br> ⚠️ 需要流域面积 |
| `methods/Fixed.py` | HYSEP Graphical <br> HYSEP 图解法 | ⚠️ Requires drainage area <br> ⚠️ 需要流域面积 |
| `methods/Slide.py` | HYSEP Graphical <br> HYSEP 图解法 | ⚠️ Requires drainage area <br> ⚠️ 需要流域面积 |
| `methods/Boughton.py` | Parameterized <br> 参数化方法 | ✅ Yes (C parameter) <br> ✅ 是(C 参数) |
| `methods/Chapman.py` | Digital Filter <br> 数字滤波器 | ✅ Yes (recession coef.) <br> ✅ 是(退水系数) |

**English**: Each method file (~100-300 lines) contains both a Numba-optimized function and an OOP wrapper class. Start with `LH.py` (simplest) before exploring others.

**中文**: 每个方法文件(约 100-300 行)都包含一个 Numba 优化的函数和一个面向对象的包装类。在探索其他方法之前,先从 `LH.py`(最简单)开始。

### Tier 3: Support Infrastructure | 第三层:支持基础设施

| File | Functionality | When You Need It |
|------|---------------|------------------|
| `param_estimate.py` | Parameter calibration <br> 参数校准 | When implementing new methods requiring optimization <br> 实现需要优化的新方法时 |
| `comparision.py` | Evaluation metrics (KGE, NSE) <br> 评估指标(KGE, NSE) | When validating algorithm performance <br> 验证算法性能时 |
| `utils.py` | Data cleaning, coordinate transforms <br> 数据清洗,坐标转换 | When preprocessing input data <br> 预处理输入数据时 |
| `config.py` | Parameter ranges, method metadata <br> 参数范围,方法元数据 | When customizing method behavior <br> 自定义方法行为时 |
| `synthetic_data.py` | Generate test datasets <br> 生成测试数据集 | When testing with known ground truth <br> 使用已知真值进行测试时 |

---

## 📖 Code Reading Guide | 源码阅读指南

### 🎯 Recommended Reading Path | 推荐阅读路径

This is a **step-by-step guide** for understanding the codebase from scratch. Follow this order:

这是**从零开始**理解代码库的逐步指南。按此顺序进行:

#### **Step 1: Understand the Problem Domain | 第一步:理解问题领域** (30 minutes | 30 分钟)

```
📓 Read: baseflow_tutorial_freshmen.ipynb
```

**English**: Open this Jupyter notebook to grasp the **"banking account analogy"** - baseflow is like your steady salary (groundwater), quickflow is like bonus income (rainfall). The interactive sliders demonstrate how parameters affect separation.

**中文**: 打开这个 Jupyter notebook 以理解**"银行账户类比"** - 基流就像你的稳定工资(地下水),快速流就像奖金收入(降雨)。交互式滑块演示了参数如何影响分割。

**Key Concepts Learned | 学习到的关键概念**:
- What is baseflow? | 什么是基流?
- Why separate streamflow? | 为什么要分离流量?
- What is the Eckhardt filter? | 什么是 Eckhardt 滤波器?

---

#### **Step 2: See the API in Action | 第二步:查看 API 实际应用** (20 minutes | 20 分钟)

```
📄 Read: examples/comprehensive_example.py
```

**English**: This file contains **5 complete workflows**:
1. **Quick test with synthetic data** - See how `single()` works
2. **Real-world single station** - Using sample CSV data
3. **Multi-station batch processing** - Using `separation()` with parallel computing
4. **Custom parameter configuration** - Adjusting method parameters
5. **Method comparison** - Comparing all 12 algorithms

**中文**: 这个文件包含 **5 个完整的工作流**:
1. **使用合成数据快速测试** - 查看 `single()` 如何工作
2. **真实世界单站** - 使用示例 CSV 数据
3. **多站批量处理** - 使用 `separation()` 进行并行计算
4. **自定义参数配置** - 调整方法参数
5. **方法比较** - 比较所有 12 种算法

**What You'll Learn | 你会学到什么**:
- How to call the main APIs | 如何调用主要 API
- What inputs are required | 需要什么输入
- What outputs to expect | 期望什么输出

---

#### **Step 3: Dive into the Main API | 第三步:深入主要 API** (45 minutes | 45 分钟)

```
📄 Read: src/baseflow/separation.py
```

**English**: This is the **most important file**. Read it line-by-line to understand:

**中文**: 这是**最重要的文件**。逐行阅读以理解:

**Key Functions & Data Flow | 关键函数与数据流**:

```python
# Function 1: single() - Single Station Processing | 函数 1: single() - 单站处理
def single(flow: pd.Series, method: str | list, area: float = None, ...) -> tuple:
    """
    Data Flow | 数据流:
    1. Input validation | 输入验证 (clean_streamflow)
    2. Frozen period detection | 冻结期检测 (exist_ice)
    3. Parameter estimation | 参数估计 (param_calibrate / recession_coefficient)
    4. Baseflow separation | 基流分离 (call method function)
    5. Performance evaluation | 性能评估 (KGE calculation)
    6. Return results | 返回结果
    """
    pass

# Function 2: separation() - Multi-Station Processing | 函数 2: separation() - 多站处理
def separation(flow: pd.DataFrame, df_sta: pd.DataFrame, method: str | list, ...) -> tuple:
    """
    Data Flow | 数据流:
    1. Loop through all stations | 循环所有站点
    2. For each station, call single() | 对每个站点调用 single()
    3. Parallel processing with joblib | 使用 joblib 并行处理
    4. Aggregate results into DataFrame | 将结果聚合到 DataFrame
    5. Optionally calculate BFI/KGE matrices | 可选计算 BFI/KGE 矩阵
    """
    pass
```

**Critical Code Sections to Focus On | 重点关注的代码部分**:

1. **Lines 50-120**: `single()` function - parameter handling logic | 参数处理逻辑
2. **Lines 200-250**: Method selection and dispatch | 方法选择和调度
3. **Lines 300-350**: `separation()` function - parallel processing | 并行处理
4. **Lines 400-450**: BFI/KGE calculation | BFI/KGE 计算

---

#### **Step 4: Understand Algorithm Implementation | 第四步:理解算法实现** (60 minutes | 60 分钟)

```
📄 Start with: src/baseflow/methods/LH.py (simplest)
📄 Then read: src/baseflow/methods/Eckhardt.py (most popular)
```

**English**: Each algorithm file follows this **dual architecture**:

**中文**: 每个算法文件遵循这个**双重架构**:

```python
# Part 1: Numba-optimized function (for performance) | 部分 1: Numba 优化函数(性能)
@njit
def lh(flow, beta=0.925, direction='b'):
    """
    Pure numerical implementation
    纯数值实现

    - Uses numpy arrays | 使用 numpy 数组
    - JIT compiled for speed | JIT 编译以提高速度
    - No pandas dependencies | 无 pandas 依赖
    """
    pass

# Part 2: OOP wrapper class (for user-friendliness) | 部分 2: OOP 包装类(用户友好)
class LHMethod(BaseflowMethod):
    """
    Object-oriented interface
    面向对象接口

    - Accepts pandas Series | 接受 pandas Series
    - Handles parameter validation | 处理参数验证
    - Integrates with separation.py | 与 separation.py 集成
    """
    def separate(self, flow: pd.Series, **params) -> pd.Series:
        return lh(flow.values, **params)
```

**Reading Order for Methods | 方法阅读顺序**:

1. **LH.py** (100 lines) - Simplest recursive filter | 最简单的递归滤波器
2. **Eckhardt.py** (150 lines) - Two-parameter filter with calibration | 带校准的双参数滤波器
3. **Local.py** (200 lines) - Graphical method with window calculations | 带窗口计算的图解法
4. **ChengBudykoML.py** (300 lines) - ML-based method (advanced) | 基于机器学习的方法(高级)

**What to Look For | 要注意什么**:
- How does the algorithm transform input flow? | 算法如何转换输入流量?
- What parameters control the separation? | 什么参数控制分离?
- How are edge cases handled? | 如何处理边缘情况?

---

#### **Step 5: Parameter Estimation Logic | 第五步:参数估计逻辑** (30 minutes | 30 分钟)

```
📄 Read: src/baseflow/param_estimate.py
```

**English**: This file answers **"How do we find optimal parameters?"**

**中文**: 这个文件回答了**"我们如何找到最优参数?"**

**Key Functions | 关键函数**:

```python
# Function 1: Estimate recession coefficient from data | 从数据估计退水系数
def recession_coefficient(flow: pd.Series) -> float:
    """
    Analyzes recession periods (no rain) to find decay rate
    分析退水期(无降雨)以找到衰减率

    Algorithm | 算法:
    1. Identify recession events | 识别退水事件
    2. Fit exponential decay | 拟合指数衰减
    3. Return average coefficient | 返回平均系数
    """
    pass

# Function 2: Grid search for optimal parameters | 网格搜索最优参数
def param_calibrate(flow: pd.Series, method: str, param_range: dict) -> dict:
    """
    Finds best parameters by maximizing NSE
    通过最大化 NSE 找到最佳参数

    Algorithm | 算法:
    1. Generate parameter grid | 生成参数网格
    2. For each combination, run separation | 对每个组合运行分离
    3. Calculate NSE score | 计算 NSE 分数
    4. Return parameters with highest NSE | 返回具有最高 NSE 的参数
    """
    pass
```

**Real-World Example | 真实示例**:
```python
# For Eckhardt method, calibrate BFImax and alpha
# 对于 Eckhardt 方法,校准 BFImax 和 alpha
optimal_params = param_calibrate(
    flow,
    method='Eckhardt',
    param_range={'BFImax': (0.2, 0.8, 0.05), 'alpha': (0.9, 0.99, 0.01)}
)
# Result: {'BFImax': 0.45, 'alpha': 0.95}
# 结果: {'BFImax': 0.45, 'alpha': 0.95}
```

---

#### **Step 6: Evaluation Metrics | 第六步:评估指标** (20 minutes | 20 分钟)

```
📄 Read: src/baseflow/comparision.py
```

**English**: This file provides **quality control** for separation results.

**中文**: 这个文件为分离结果提供**质量控制**。

**Key Functions | 关键函数**:

```python
# Function 1: Identify "strict baseflow" periods | 识别"严格基流"期
def strict_baseflow(flow: pd.Series) -> pd.Series:
    """
    4-step filtering to find pure baseflow periods
    4 步过滤以找到纯基流期

    Filters | 过滤器:
    1. Remove frozen periods | 去除冻结期
    2. Remove high-flow events | 去除高流量事件
    3. Require long dry periods | 要求长时间干旱期
    4. Check flow stability | 检查流量稳定性

    Used for: Parameter calibration benchmark
    用途: 参数校准基准
    """
    pass

# Function 2: Calculate Kling-Gupta Efficiency | 计算 Kling-Gupta 效率
def KGE(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    Evaluates how well separated baseflow matches "truth"
    评估分离的基流与"真值"的匹配程度

    Formula | 公式: KGE = 1 - sqrt((r-1)^2 + (α-1)^2 + (β-1)^2)

    Components | 组成部分:
    - r: Correlation | 相关性
    - α: Variability ratio | 变异性比率
    - β: Bias ratio | 偏差比率

    Range | 范围: (-∞, 1], higher is better | 越高越好
    """
    pass
```

---

#### **Step 7: Configuration System | 第七步:配置系统** (15 minutes | 15 分钟)

```
📄 Read: src/baseflow/config.py
```

**English**: This file is the **central control panel** for all method parameters.

**中文**: 这个文件是所有方法参数的**中央控制面板**。

**Key Components | 关键组件**:

```python
# Class 1: Method metadata | 方法元数据
class MethodConfig:
    """
    Defines each method's characteristics
    定义每个方法的特征

    Attributes | 属性:
    - name: Method identifier | 方法标识符
    - description: Brief explanation | 简要说明
    - required_params: Mandatory parameters | 必需参数
    - optional_params: Optional parameters | 可选参数
    - default_ranges: Parameter search space | 参数搜索空间
    - requires_area: Whether drainage area is needed | 是否需要流域面积
    """
    pass

# Class 2: Separation configuration | 分离配置
class SeparationConfig:
    """
    Runtime configuration for separation jobs
    分离作业的运行时配置

    Attributes | 属性:
    - method: Which algorithm(s) to use | 使用哪种算法
    - calibrate: Whether to run parameter optimization | 是否运行参数优化
    - n_jobs: Number of parallel workers | 并行工作器数量
    - frozen_detection: Enable/disable frozen period handling | 启用/禁用冻结期处理
    """
    pass
```

**Why This Matters | 为什么这很重要**:
- Want to add a new method? Register it here. | 想添加新方法?在这里注册。
- Want to change default parameters? Modify here. | 想更改默认参数?在这里修改。
- Want to customize calibration ranges? Edit here. | 想自定义校准范围?在这里编辑。

---

#### **Step 8: Batch Processing Scripts | 第八步:批处理脚本** (30 minutes | 30 分钟)

```
📄 Read: scripts/batch/daily_batch_run.py
📄 Read: scripts/run_all_methods.py
```

**English**: These are **production-ready** scripts for operational hydrology.

**中文**: 这些是用于业务水文的**生产就绪**脚本。

**Workflow Pattern | 工作流模式**:

```python
# Step 1: Load configuration | 加载配置
config = yaml.safe_load('config.yml')

# Step 2: Load multi-station data | 加载多站数据
flow_df = pd.read_csv(config['input_path'])
station_info = pd.read_csv(config['station_path'])

# Step 3: Run batch separation | 运行批量分离
results = separation(
    flow_df,
    station_info,
    method=config['methods'],
    n_jobs=config['n_jobs']
)

# Step 4: Save results | 保存结果
for method, baseflow_df in results.items():
    baseflow_df.to_csv(f"output/{method}_baseflow.csv")

# Step 5: Generate summary report | 生成摘要报告
report = calculate_bfi_and_kge(results)
report.to_csv("output/summary.csv")
```

---

### 🗺️ Data Flow Diagram | 数据流图

**English**: Here's how data flows through the system:

**中文**: 以下是数据在系统中的流动方式:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT | 用户输入                          │
│  • Streamflow time series (pandas Series/DataFrame)              │
│    流量时间序列 (pandas Series/DataFrame)                          │
│  • Station metadata (area, coordinates)                          │
│    站点元数据 (面积, 坐标)                                          │
│  • Method selection (string or list)                             │
│    方法选择 (字符串或列表)                                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING | 预处理 (utils.py)                   │
│  1. Data validation & cleaning | 数据验证和清洗                    │
│     - Remove NaN values | 删除 NaN 值                             │
│     - Check for negative flows | 检查负流量                        │
│     - Verify datetime index | 验证日期时间索引                      │
│  2. Frozen period detection | 冻结期检测                           │
│     - Load global permafrost mask | 加载全球永久冻土掩膜              │
│     - Identify winter periods | 识别冬季期                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│        PARAMETER ESTIMATION | 参数估计 (param_estimate.py)        │
│  IF method requires calibration | 如果方法需要校准:                 │
│  1. Extract recession events | 提取退水事件                         │
│  2. Estimate recession coefficient | 估计退水系数                   │
│  3. Grid search for optimal parameters | 网格搜索最优参数            │
│  4. Use NSE as objective function | 使用 NSE 作为目标函数            │
│  ELSE: Use default parameters | 否则:使用默认参数                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│       BASEFLOW SEPARATION | 基流分离 (methods/*.py)               │
│  FOR each selected method | 对于每个选定的方法:                     │
│  1. Retrieve method function | 检索方法函数                         │
│  2. Apply algorithm to flow data | 将算法应用于流量数据               │
│  3. Handle edge cases (start/end periods) | 处理边缘情况(开始/结束期) │
│  4. Ensure baseflow ≤ total flow | 确保基流 ≤ 总流量                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│       EVALUATION | 评估 (comparision.py)                         │
│  IF evaluation requested | 如果请求评估:                           │
│  1. Identify strict baseflow periods | 识别严格基流期                │
│  2. Calculate KGE metric | 计算 KGE 指标                           │
│  3. Compare separated baseflow vs benchmark | 比较分离的基流与基准     │
│  4. Calculate BFI (Baseflow Index) | 计算 BFI(基流指数)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT | 输出                                    │
│  • Baseflow time series (DataFrame) | 基流时间序列 (DataFrame)      │
│  • KGE scores (dict or DataFrame) | KGE 分数 (dict 或 DataFrame)  │
│  • BFI values (float or DataFrame) | BFI 值 (float 或 DataFrame)  │
│  • Optional: Plots and CSV files | 可选:图表和 CSV 文件             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌍 Business Logic Mapping | 业务场景映射

### Real-World Concepts → Code Components | 现实世界概念 → 代码组件

| Real-World Concept<br>现实世界概念 | Code Component<br>代码组件 | Explanation<br>解释 |
|-----------------------------------|----------------------------|-------------------|
| **River streamflow observation**<br>河流流量观测 | `pd.Series` with `DatetimeIndex`<br>带 `DatetimeIndex` 的 `pd.Series` | Input data structure representing daily/hourly flow measurements<br>表示每日/每小时流量测量的输入数据结构 |
| **Groundwater contribution (baseflow)**<br>地下水贡献(基流) | `baseflow_df` output from `single()`<br>`single()` 输出的 `baseflow_df` | The slow-responding component from aquifers<br>来自含水层的慢响应组件 |
| **Surface runoff (quickflow)**<br>地表径流(快速流) | `flow - baseflow`<br>`flow - baseflow` | The fast-responding component from rainfall<br>来自降雨的快速响应组件 |
| **Watershed/Catchment area**<br>流域/集水区面积 | `area` parameter (km²)<br>`area` 参数 (平方公里) | Used by HYSEP methods to determine recession duration<br>被 HYSEP 方法用来确定退水持续时间 |
| **Recession period (dry spell)**<br>退水期(干旱期) | `recession_coefficient()` analysis<br>`recession_coefficient()` 分析 | Period when flow decreases exponentially without rain<br>流量在无降雨时呈指数下降的时期 |
| **Permafrost/Frozen ground**<br>永久冻土/冻土 | `thawed.npz` + `exist_ice()`<br>`thawed.npz` + `exist_ice()` | Regions where baseflow separation is unreliable in winter<br>冬季基流分离不可靠的地区 |
| **Hydrological station network**<br>水文站网 | `df_sta` DataFrame with metadata<br>带元数据的 `df_sta` DataFrame | Multiple gauging stations managed by a water agency<br>由水务机构管理的多个测站 |
| **Baseflow Index (BFI)**<br>基流指数 (BFI) | `sum(baseflow) / sum(total_flow)`<br>`sum(baseflow) / sum(total_flow)` | Ratio indicating groundwater dominance (0-1 scale)<br>指示地下水主导程度的比率(0-1 范围) |
| **Hydrograph separation accuracy**<br>水文过程线分离精度 | `KGE()` metric<br>`KGE()` 指标 | How well the method matches "true" baseflow (if known)<br>方法与"真实"基流的匹配程度(如果已知) |
| **Operational hydrology workflow**<br>业务水文工作流 | `scripts/batch/*.py` + `config.yml`<br>`scripts/batch/*.py` + `config.yml` | Automated daily/monthly processing for water resource management<br>水资源管理的自动每日/每月处理 |
| **Research experiment**<br>研究实验 | `synthetic_data.py` + `examples/`<br>`synthetic_data.py` + `examples/` | Controlled tests with known ground truth for method validation<br>使用已知真值进行方法验证的受控测试 |

---

### 📊 Typical Use Case Scenarios | 典型使用场景

#### Scenario 1: Academic Research | 场景 1: 学术研究

**English**: A PhD student wants to compare 12 baseflow separation methods on a watershed.

**中文**: 一位博士生想在一个流域上比较 12 种基流分离方法。

**Code Workflow | 代码工作流**:
```python
# Step 1: Load data | 加载数据
flow = pd.read_csv('watershed_data.csv', index_col=0, parse_dates=True)['Q']

# Step 2: Run all methods | 运行所有方法
baseflow_df, kge_scores = single(flow, method='all', area=500)

# Step 3: Visualize comparison | 可视化比较
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(flow.index, flow, label='Total Flow | 总流量', color='blue')
for method in baseflow_df.columns:
    ax.plot(baseflow_df.index, baseflow_df[method], label=method, alpha=0.7)
ax.legend()
plt.show()

# Step 4: Analyze results | 分析结果
print(f"KGE scores | KGE 分数:\n{kge_scores}")
print(f"BFI values | BFI 值:\n{baseflow_df.sum() / flow.sum()}")
```

**Files to Read | 要阅读的文件**: `examples/comprehensive_example.py`, `src/baseflow/separation.py`

---

#### Scenario 2: Water Resource Management | 场景 2: 水资源管理

**English**: A government agency needs to process 50 stations every month to monitor groundwater contribution.

**中文**: 一个政府机构需要每月处理 50 个站点以监测地下水贡献。

**Code Workflow | 代码工作流**:
```python
# Step 1: Configure workflow | 配置工作流
# Edit config.yml:
# methods: ["LH", "UKIH", "Eckhardt"]
# n_jobs: 8  # use 8 CPU cores | 使用 8 个 CPU 核心

# Step 2: Run batch script | 运行批处理脚本
# Execute: python scripts/batch/monthly_batch_run.py

# Step 3: Results saved automatically | 结果自动保存
# Output files | 输出文件:
# - outputs/LH_monthly_baseflow.csv
# - outputs/UKIH_monthly_baseflow.csv
# - outputs/Eckhardt_monthly_baseflow.csv
# - outputs/summary_report.csv (BFI, KGE for all stations)
```

**Files to Read | 要阅读的文件**: `scripts/batch/monthly_batch_run.py`, `config.yml`

---

#### Scenario 3: Method Development | 场景 3: 方法开发

**English**: You want to implement a new baseflow separation algorithm and compare it with existing methods.

**中文**: 你想实现一个新的基流分离算法并与现有方法进行比较。

**Code Workflow | 代码工作流**:
```python
# Step 1: Create new method file | 创建新方法文件
# File: src/baseflow/methods/MyNewMethod.py

from numba import njit
from ._base import BaseflowMethod

@njit
def my_new_method(flow, alpha=0.95, beta=0.5):
    """
    Your algorithm implementation
    你的算法实现
    """
    baseflow = np.zeros_like(flow)
    # ... your logic here | 你的逻辑在这里 ...
    return baseflow

class MyNewMethod(BaseflowMethod):
    """OOP wrapper | 面向对象包装器"""
    def separate(self, flow, **params):
        return my_new_method(flow.values, **params)

# Step 2: Register in config.py | 在 config.py 中注册
# Add to METHOD_REGISTRY:
METHOD_REGISTRY['MyNew'] = MethodConfig(
    name='MyNew',
    description='My innovative method',
    required_params=[],
    optional_params=['alpha', 'beta'],
    default_ranges={'alpha': (0.9, 0.99, 0.01), 'beta': (0.3, 0.7, 0.1)}
)

# Step 3: Test with synthetic data | 使用合成数据测试
from baseflow.synthetic_data import generate_streamflow
flow, true_baseflow = generate_streamflow(days=365)

baseflow_df, kge = single(flow, method=['MyNew', 'LH', 'Eckhardt'])
print(f"My method KGE: {kge['MyNew']}")  # Compare performance | 比较性能

# Step 4: Run comprehensive tests | 运行综合测试
pytest tests/test_separation.py::test_new_method
```

**Files to Read | 要阅读的文件**: `src/baseflow/methods/_base.py`, `src/baseflow/config.py`, `tests/test_separation.py`

---

## 🛠️ Development Guide | 开发指南

### Setting Up Development Environment | 设置开发环境

```bash
# 1. Clone and install | 克隆并安装
git clone https://github.com/licm13/Baseflow_Seperation.git
cd Baseflow_Seperation
pip install -e ".[dev]"  # Installs with development dependencies | 安装开发依赖

# 2. Install pre-commit hooks (optional) | 安装 pre-commit 钩子(可选)
# pip install pre-commit
# pre-commit install

# 3. Verify installation | 验证安装
python -c "import baseflow; print(baseflow.__version__)"
pytest tests/  # Run all tests | 运行所有测试
```

---

### How to Modify Code | 如何修改代码

#### Adding a New Separation Method | 添加新的分离方法

**Steps | 步骤**:
1. Create `src/baseflow/methods/YourMethod.py`
2. Implement both Numba function and OOP wrapper | 实现 Numba 函数和 OOP 包装器
3. Register in `src/baseflow/config.py`
4. Add unit test in `tests/test_separation.py`
5. Update documentation in `docs/algorithm_details.md`

**Template | 模板**:
```python
# YourMethod.py
from numba import njit
import numpy as np
from ._base import BaseflowMethod

@njit
def your_method_core(flow, param1, param2):
    """
    Numba-optimized implementation
    Args:
        flow: 1D numpy array
        param1, param2: Algorithm parameters
    Returns:
        baseflow: 1D numpy array
    """
    n = len(flow)
    baseflow = np.zeros(n)

    # Your algorithm logic here | 你的算法逻辑
    for i in range(1, n):
        baseflow[i] = ...  # calculation | 计算

    return baseflow

class YourMethod(BaseflowMethod):
    """User-friendly wrapper"""
    name = 'YourMethod'

    def separate(self, flow, param1=0.95, param2=0.5, **kwargs):
        result = your_method_core(flow.values, param1, param2)
        return pd.Series(result, index=flow.index)
```

---

#### Customizing Parameter Ranges | 自定义参数范围

**English**: To change parameter search ranges for calibration:

**中文**: 要更改校准的参数搜索范围:

```python
# Edit: src/baseflow/config.py

# Original | 原始:
METHOD_REGISTRY['Eckhardt'] = MethodConfig(
    default_ranges={'BFImax': (0.2, 0.8, 0.05), 'alpha': (0.9, 0.99, 0.01)}
)

# Modified | 修改后:
METHOD_REGISTRY['Eckhardt'] = MethodConfig(
    default_ranges={'BFImax': (0.3, 0.7, 0.025), 'alpha': (0.92, 0.98, 0.005)}
    # Narrower range, finer grid | 更窄的范围,更精细的网格
)
```

---

#### Adding New Evaluation Metrics | 添加新的评估指标

**English**: To add metrics like RMSE or NSE:

**中文**: 要添加 RMSE 或 NSE 等指标:

```python
# Edit: src/baseflow/comparision.py

def RMSE(simulated, observed):
    """
    Root Mean Square Error
    均方根误差
    """
    return np.sqrt(np.mean((simulated - observed) ** 2))

def NSE(simulated, observed):
    """
    Nash-Sutcliffe Efficiency
    纳什-萨特克利夫效率系数
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator)

# Then use in separation.py | 然后在 separation.py 中使用
```

---

### Code Style Guidelines | 代码风格指南

**English**: This project follows PEP 8 with these conventions:

**中文**: 该项目遵循 PEP 8,具有以下约定:

- **Type hints**: 100% coverage for public APIs | 公共 API 100% 覆盖
- **Docstrings**: Google style with Args/Returns/Examples | Google 风格,包含 Args/Returns/Examples
- **Function names**: `snake_case` | 蛇形命名法
- **Class names**: `PascalCase` | 帕斯卡命名法
- **Constants**: `UPPER_CASE` | 大写
- **Private functions**: Prefix with `_` | 前缀 `_`
- **Line length**: Max 100 characters | 最多 100 字符

---

## 🧪 Testing | 测试

### Running Tests | 运行测试

```bash
# Run all tests | 运行所有测试
pytest tests/

# Run specific test file | 运行特定测试文件
pytest tests/test_separation.py

# Run with coverage report | 运行并生成覆盖率报告
pytest --cov=src/baseflow --cov-report=html tests/

# Run fast tests only (skip slow calibrations) | 仅运行快速测试(跳过慢速校准)
pytest -m "not slow" tests/
```

### Writing Tests | 编写测试

**Template | 模板**:
```python
# tests/test_your_feature.py
import pytest
import pandas as pd
from baseflow import single
from baseflow.synthetic_data import generate_streamflow

def test_your_feature():
    """
    Test description | 测试描述
    """
    # Arrange | 准备
    flow, true_baseflow = generate_streamflow(days=365)

    # Act | 执行
    result, kge = single(flow, method='LH')

    # Assert | 断言
    assert not result.isna().any(), "No NaN values | 无 NaN 值"
    assert (result <= flow).all(), "Baseflow ≤ total flow | 基流 ≤ 总流量"
    assert kge['LH'] > 0.5, "Reasonable KGE score | 合理的 KGE 分数"
```

---

## 🤝 Contributing | 贡献

### How to Contribute | 如何贡献

**English**:
1. Fork the repository | 分叉仓库
2. Create a feature branch: `git checkout -b feature/my-new-method`
3. Make your changes with tests | 进行更改并添加测试
4. Run tests: `pytest tests/`
5. Commit with clear messages: `git commit -m "Add: New baseflow method"`
6. Push to your fork: `git push origin feature/my-new-method`
7. Submit a Pull Request | 提交拉取请求

**中文**:
1. 分叉仓库
2. 创建功能分支: `git checkout -b feature/my-new-method`
3. 进行更改并添加测试
4. 运行测试: `pytest tests/`
5. 使用清晰的消息提交: `git commit -m "Add: New baseflow method"`
6. 推送到你的分叉: `git push origin feature/my-new-method`
7. 提交拉取请求

---

### Priority Areas for Contribution | 优先贡献领域

**English**:
- ⭐ Implement additional metrics (NSE, RMSE, MAE)
- ⭐ Add uncertainty quantification for parameter estimates
- ⭐ Create interactive web visualization interface
- ⭐ Improve documentation with more examples
- ⭐ Optimize performance for large datasets (> 10 years)

**中文**:
- ⭐ 实现额外的指标(NSE、RMSE、MAE)
- ⭐ 为参数估计添加不确定性量化
- ⭐ 创建交互式 Web 可视化界面
- ⭐ 通过更多示例改进文档
- ⭐ 优化大数据集(> 10 年)的性能

---

## 📚 Additional Resources | 额外资源

### Documentation | 文档
- **Algorithm details** | 算法详情: `docs/algorithm_details.md`
- **Batch processing manual** | 批处理手册: `docs/batch_processing_manual.md`
- **API reference** | API 参考: Docstrings in source code | 源代码中的文档字符串

### Examples | 示例
- **Quick start** | 快速开始: `examples/quick_test.py`
- **Interactive exploration** | 交互式探索: `examples/interactive_explorer.py`
- **Advanced workflows** | 高级工作流: `examples/comprehensive_example.py`
- **Visualization** | 可视化: `examples/advanced_visualization.py`

### Educational | 教育
- **Tutorial notebook** | 教程笔记本: `baseflow_tutorial_freshmen.ipynb`
- **Banking analogy** | 银行类比: Explains baseflow as "steady salary" | 将基流解释为"稳定工资"

### Research Papers | 研究论文
- **Cheng Budyko ML method** | Cheng Budyko 机器学习方法: `Cheng-3D-Budyko/paper.pdf`
- **Original methods** | 原始方法: References in `docs/algorithm_details.md`

---

## 📞 Contact & Support | 联系与支持

**English**:
- **Author** | 作者: Cody James (xiejx5@gmail.com)
- **Repository** | 仓库: https://github.com/licm13/Baseflow_Seperation
- **Issues** | 问题: Report bugs or request features on GitHub Issues
- **License** | 许可证: MIT License

**中文**:
- **作者**: Cody James (xiejx5@gmail.com)
- **仓库**: https://github.com/licm13/Baseflow_Seperation
- **问题**: 在 GitHub Issues 上报告错误或请求功能
- **许可证**: MIT 许可证

---

## 🎓 Learning Path Summary | 学习路径总结

### For Complete Beginners | 完全新手

**English**: If you're new to hydrology and Python:

**中文**: 如果你是水文学和 Python 的新手:

1. **Day 1**: Read tutorial notebook (`baseflow_tutorial_freshmen.ipynb`) | 阅读教程笔记本
2. **Day 2**: Run `examples/quick_test.py` to see basic usage | 运行快速测试查看基本用法
3. **Day 3**: Read `src/baseflow/methods/LH.py` to understand a simple algorithm | 阅读 LH.py 理解简单算法
4. **Day 4**: Experiment with `examples/comprehensive_example.py` | 尝试综合示例
5. **Day 5**: Try processing your own data | 尝试处理自己的数据

### For Experienced Developers | 经验丰富的开发者

**English**: If you want to contribute or customize:

**中文**: 如果你想贡献或自定义:

1. **Hour 1**: Read `src/baseflow/separation.py` and `src/baseflow/config.py`
2. **Hour 2**: Study two methods: `LH.py` (simple) and `Eckhardt.py` (complex)
3. **Hour 3**: Understand `param_estimate.py` for calibration logic | 理解参数估计的校准逻辑
4. **Hour 4**: Run all tests and examine test code | 运行所有测试并检查测试代码
5. **Hour 5**: Implement a custom method or modify existing ones | 实现自定义方法或修改现有方法

---

## 🏆 Key Takeaways | 关键要点

**English**:
- 🎯 **Core API**: `single()` for one station, `separation()` for multiple | 单站用 `single()`,多站用 `separation()`
- 🔧 **12 Methods**: 3 families (digital filters, HYSEP, parameterized) | 3 个家族(数字滤波器、HYSEP、参数化)
- ⚡ **Performance**: Numba JIT + parallel processing for speed | Numba JIT + 并行处理提高速度
- 📊 **Evaluation**: KGE, BFI, NSE metrics for quality control | KGE、BFI、NSE 指标用于质量控制
- 🧪 **Testing**: Synthetic data with known ground truth | 使用已知真值的合成数据
- 🌍 **Production-ready**: Batch scripts + config files for operations | 批处理脚本 + 配置文件用于运营

**中文**:
- 🎯 **核心 API**: 单站用 `single()`,多站用 `separation()`
- 🔧 **12 种方法**: 3 个家族(数字滤波器、HYSEP、参数化)
- ⚡ **性能**: Numba JIT + 并行处理提高速度
- 📊 **评估**: KGE、BFI、NSE 指标用于质量控制
- 🧪 **测试**: 使用已知真值的合成数据
- 🌍 **生产就绪**: 批处理脚本 + 配置文件用于运营

---

**Happy Coding! | 编码愉快!** 🚀💧

---

*Last Updated | 最后更新: 2025-01-08*
*Version | 版本: 1.0.0*
