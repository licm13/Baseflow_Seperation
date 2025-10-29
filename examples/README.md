# Baseflow Separation Examples

This directory contains comprehensive examples demonstrating how to use the baseflow separation package. All examples include random data generation for testing and validation.

## 📋 Quick Start

```bash
# Quick test with random data
python quick_test.py

# Interactive exploration
python interactive_explorer.py

# Comprehensive examples
python comprehensive_example.py
```

## 📁 Example Files

### 1. **quick_test.py** - Fast Testing & Validation

Quick test script for rapid validation of baseflow separation methods using randomly generated data.

**Features:**
- Random synthetic data generation with configurable parameters
- Tests 5 popular methods (LH, Chapman, Eckhardt, UKIH, Boughton)
- Performance evaluation with KGE, BFI error, and RMSE metrics
- Support for multiple test runs with different random data

**Usage:**
```bash
# Single test with random data
python quick_test.py

# Test with specific number of days
python quick_test.py --days 730

# Reproducible test with seed
python quick_test.py --seed 42

# Run 10 tests with different random data
python quick_test.py --multiple 10
```

**Example Output:**
```
QUICK TEST: Baseflow Separation with Random Data
==================================================================
Using random seed: 7342

Generated Streamflow Statistics:
  Mean: 25.43 m³/s
  Max: 156.32 m³/s
  True BFI: 0.632

Performance Evaluation:
  Method          KGE Score       Est. BFI     BFI Error
  ----------------------------------------------------------
  LH                 0.8234         0.6289         0.0031
  Eckhardt           0.8567         0.6315         0.0005
  ...
```

---

### 2. **interactive_explorer.py** - Interactive Analysis Tool

Interactive command-line tool for exploring baseflow separation with different hydrological scenarios.

**Features:**
- 5 predefined scenario types (perennial, intermittent, ephemeral, balanced, random)
- Interactive parameter selection
- Real-time results visualization
- Robustness analysis across multiple scenarios
- Batch testing mode

**Usage:**
```bash
# Interactive mode (default)
python interactive_explorer.py

# Run specific scenario
python interactive_explorer.py --scenario ephemeral --days 365

# Run 10 random scenarios
python interactive_explorer.py --multiple 10
```

**Scenario Types:**

| Scenario | Description | Typical BFI | Use Case |
|----------|-------------|-------------|----------|
| **Perennial** | High baseflow, few storms | 0.70-0.85 | Snow-dominated basins |
| **Intermittent** | Medium baseflow, moderate storms | 0.50-0.65 | Mixed hydrology |
| **Ephemeral** | Low baseflow, many storms | 0.30-0.45 | Arid/semi-arid regions |
| **Balanced** | Balanced characteristics | 0.50-0.70 | Temperate regions |
| **Random** | Random parameters | 0.30-0.85 | Stress testing |

---

### 3. **advanced_visualization.py** - Comprehensive Visualization

Creates publication-quality figures with detailed analysis of separation results.

**Features:**
- 8-panel comprehensive visualization
- Time series, flow duration curves, error analysis
- Performance heatmaps with multiple metrics
- Scatter plots with regression analysis
- Automated saving of high-resolution figures

**Usage:**
```bash
python advanced_visualization.py
```

**Generated Plots:**
1. `advanced_visualization.png` - Multi-panel comprehensive analysis
2. `performance_heatmap.png` - Method comparison heatmap

**Visualizations Include:**
- Full time series with multiple methods
- Zoomed recession period detail
- Flow duration curves
- KGE performance scores
- BFI comparison bar charts
- Error distribution boxplots
- Cumulative baseflow curves
- True vs. estimated scatter plots with R² and RMSE

---

### 4. **benchmark_performance.py** - Performance Benchmarking

Comprehensive performance analysis and computational benchmarking of all methods.

**Features:**
- Method speed comparison across different data sizes
- Batch processing throughput analysis
- Computational scaling analysis (complexity estimation)
- Method family comparison
- Memory usage estimation

**Usage:**
```bash
python benchmark_performance.py
```

**Benchmark Categories:**

1. **Method Comparison** - Tests all 12 methods with 1, 3, 5, and 10 years of data
2. **Batch Processing** - Multi-station throughput (5, 10, 25, 50 stations)
3. **Scaling Analysis** - Computational complexity assessment
4. **Family Comparison** - Digital filters vs. graphical vs. parameterized methods
5. **Memory Estimation** - Memory requirements for different configurations

**Example Output:**
```
Benchmark Results (1 year):
Method              Mean (s)    Std (s)
----------------------------------------
LH                   0.0234      0.0012
UKIH                 0.0256      0.0015
Eckhardt             0.1523      0.0087
...

Batch Processing (50 stations):
  Throughput: 327.5 separations/second
  Time per station: 0.459 seconds
```

**Output Files:**
- `benchmark_methods.csv` - Detailed method timing results
- `benchmark_batch.csv` - Batch processing statistics
- `benchmark_scaling.csv` - Scaling analysis data

---

### 5. **comprehensive_example.py** - Complete Workflow Demo

Full-featured example demonstrating the complete workflow from data generation to result analysis.

**Features:**
- 5 detailed examples covering all major use cases
- Single-station and multi-station processing
- Visualization techniques
- Parameter sensitivity analysis
- Method comparison across flow regimes
- Extensive documentation and comments

**Examples Included:**

1. **Example 1**: Basic single-station separation
2. **Example 2**: Visualization (time series, FDC, recession analysis)
3. **Example 3**: Multi-station batch processing
4. **Example 4**: Parameter sensitivity analysis
5. **Example 5**: Comprehensive method comparison

**Usage:**
```bash
python comprehensive_example.py
```

**Generated Output:**
- `separation_results.png` - Multi-panel visualization
- Console output with detailed statistics

---

### 6. **run_all_methods.py** - Simple Minimal Example

Minimal working example for quick testing. Good starting point for beginners.

**Features:**
- Simple, straightforward code
- All 12 methods applied
- CSV output
- Minimal dependencies

**Usage:**
```bash
python run_all_methods.py
```

---

## 🎯 Use Case Guide

### I want to...

**...quickly test if the package works**
→ Use `quick_test.py`

**...explore different hydrological scenarios**
→ Use `interactive_explorer.py`

**...create publication-quality figures**
→ Use `advanced_visualization.py`

**...benchmark computational performance**
→ Use `benchmark_performance.py`

**...learn the complete workflow**
→ Use `comprehensive_example.py`

**...see the simplest possible example**
→ Use `run_all_methods.py`

---

## 🔧 Common Parameters

All examples support synthetic data generation with these parameters:

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `n_days` | Time series length | 365-3650 | days |
| `base_flow` | Mean baseflow | 5-40 | m³/s |
| `seasonal_amplitude` | Seasonal variation | 2-15 | m³/s |
| `n_storm_events` | Number of storms | 10-60 | count |
| `storm_intensity` | Storm peak magnitude | 20-120 | m³/s |
| `bfi` | Target Baseflow Index | 0.3-0.85 | - |
| `random_seed` | Reproducibility seed | any integer | - |

---

## 📊 Performance Metrics

All examples report these standard metrics:

| Metric | Description | Range | Optimal |
|--------|-------------|-------|---------|
| **KGE** | Kling-Gupta Efficiency | -∞ to 1 | > 0.8 |
| **BFI** | Baseflow Index (ratio) | 0 to 1 | Match true |
| **BFI Error** | Absolute BFI difference | 0 to 1 | < 0.05 |
| **RMSE** | Root Mean Square Error | 0 to ∞ | Minimize |
| **R²** | Coefficient of determination | 0 to 1 | > 0.9 |
| **MAE** | Mean Absolute Error | 0 to ∞ | Minimize |

---

## 🚀 Advanced Usage

### Custom Scenario Testing

```python
from baseflow import single
from baseflow.synthetic_data import generate_streamflow
import pandas as pd
import numpy as np

# Generate custom scenario
Q, true_baseflow, quickflow = generate_streamflow(
    n_days=365,
    base_flow=20.0,          # Custom value
    seasonal_amplitude=8.0,   # Custom value
    n_storm_events=30,        # Custom value
    storm_intensity=75.0,     # Custom value
    bfi=0.65,                # Custom target
    random_seed=42           # Reproducible
)

# Create series
dates = pd.date_range('2020-01-01', periods=len(Q), freq='D')
flow_series = pd.Series(Q, index=dates)

# Apply separation
baseflow_df, kge_scores = single(
    flow_series,
    area=1500.0,  # km²
    method=['LH', 'Eckhardt', 'Chapman'],
    return_kge=True
)

# Evaluate
true_bfi = true_baseflow.sum() / Q.sum()
for method in baseflow_df.columns:
    est_bfi = baseflow_df[method].sum() / Q.sum()
    print(f"{method}: BFI={est_bfi:.3f}, Error={abs(est_bfi-true_bfi):.3f}")
```

### Multi-Station Random Testing

```python
from baseflow import separation
from baseflow.synthetic_data import create_test_dataframe

# Generate multi-station data
flow_df, true_baseflow_df, station_info = create_test_dataframe(
    n_days=730,
    n_stations=10,
    random_seed=123
)

# Run separation
results, bfi, kge = separation(
    flow_df,
    df_sta=station_info,
    method=['LH', 'Eckhardt', 'Chapman'],
    return_bfi=True,
    return_kge=True
)

print("BFI Results:")
print(bfi)
print("\nKGE Scores:")
print(kge)
```

---

## 📝 Notes

1. **Random Data Generation**: All examples use synthetic data with known ground truth, enabling validation of method accuracy.

2. **Reproducibility**: Use `random_seed` parameter for reproducible results. Omit for true random data.

3. **Method Selection**: Choose methods based on your needs:
   - Fast: LH, UKIH
   - Accurate: Eckhardt, Chapman
   - No calibration: LH, UKIH, CM
   - Physically based: Eckhardt, Boughton

4. **Performance**: For large datasets (>1000 stations, >10 years), consider:
   - Using fewer methods
   - Batch processing
   - Parallel execution (built-in)

5. **Visualization**: Matplotlib required for visualization examples. Install with:
   ```bash
   pip install matplotlib
   ```

---

## 🐛 Troubleshooting

**Q: Import error when running examples**
```bash
# Make sure you're in the examples directory
cd examples
# Or install the package
cd .. && pip install -e .
```

**Q: Memory error with large datasets**
- Reduce number of methods
- Process stations in batches
- Reduce time series length

**Q: Slow performance**
- Check if Numba is installed and JIT compilation is working
- Reduce parameter search grid (see `config.py`)
- Use faster methods (LH, UKIH) for initial testing

---

## 📚 Additional Resources

- **Package Documentation**: See main README in parent directory
- **Method References**: Check docstrings in `src/baseflow/methods/`
- **API Documentation**: See `src/baseflow/separation.py`
- **Configuration**: See `src/baseflow/config.py` for parameter tuning

---

## 🤝 Contributing

Found a bug or have an idea for a new example? Please open an issue or submit a pull request!

---

**Happy Separating! 🌊**
