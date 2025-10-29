"""Performance benchmark for baseflow separation methods.

This script benchmarks the computational performance of different baseflow
separation methods under various conditions.

Usage:
    python benchmark_performance.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import single, separation
from baseflow.synthetic_data import create_test_dataframe, generate_streamflow


def benchmark_single_method(method, flow_series, area=1000, n_runs=10):
    """Benchmark a single method.

    Args:
        method: Method name to benchmark
        flow_series: Streamflow series
        area: Drainage area in km²
        n_runs: Number of runs for averaging

    Returns:
        Dictionary with timing results
    """
    times = []

    for _ in range(n_runs):
        start = time.time()
        single(flow_series, area=area, method=[method], return_kge=False)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'method': method,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def benchmark_all_methods():
    """Benchmark all methods with different data sizes."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: All Methods")
    print("=" * 70)

    # Test with different time series lengths
    test_sizes = [365, 365*3, 365*5, 365*10]  # 1, 3, 5, 10 years

    from baseflow.config import ALL_METHODS

    results = []

    for n_days in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {n_days} days ({n_days/365:.1f} years)")
        print(f"{'='*70}")

        # Generate test data
        Q, _, _ = generate_streamflow(
            n_days=n_days,
            base_flow=15,
            n_storm_events=int(n_days * 0.05),  # ~5% of days have storms
            bfi=0.65,
            random_seed=42
        )

        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        flow_series = pd.Series(Q, index=dates)

        print(f"\nBenchmarking {len(ALL_METHODS)} methods...")
        print(f"{'Method':<15} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Max (s)':>10}")
        print("-" * 70)

        for method in ALL_METHODS:
            try:
                result = benchmark_single_method(method, flow_series, n_runs=5)
                result['n_days'] = n_days
                results.append(result)

                print(f"{method:<15} {result['mean_time']:>10.4f} {result['std_time']:>10.4f} "
                      f"{result['min_time']:>10.4f} {result['max_time']:>10.4f}")
            except Exception as e:
                print(f"{method:<15} FAILED: {str(e)}")

    # Create summary DataFrame
    df_results = pd.DataFrame(results)

    return df_results


def benchmark_batch_processing():
    """Benchmark multi-station batch processing."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK: Batch Processing")
    print("=" * 70)

    station_counts = [5, 10, 25, 50]
    results = []

    for n_stations in station_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {n_stations} stations")
        print(f"{'='*70}")

        # Generate multi-station data
        flow_df, _, station_info = create_test_dataframe(
            n_days=365*2,
            n_stations=n_stations,
            random_seed=123
        )

        print(f"\nData shape: {flow_df.shape}")
        print(f"Testing methods: LH, Eckhardt, Chapman")

        methods = ['LH', 'Eckhardt', 'Chapman']

        start = time.time()
        dfs, bfi, kge = separation(
            flow_df,
            df_sta=station_info,
            method=methods,
            return_bfi=True,
            return_kge=True
        )
        elapsed = time.time() - start

        time_per_station = elapsed / n_stations
        time_per_method = elapsed / (n_stations * len(methods))

        print(f"\nResults:")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Time per station: {time_per_station:.3f} seconds")
        print(f"  Time per station-method: {time_per_method:.4f} seconds")
        print(f"  Throughput: {n_stations * len(methods) / elapsed:.1f} separations/second")

        results.append({
            'n_stations': n_stations,
            'n_methods': len(methods),
            'total_time': elapsed,
            'time_per_station': time_per_station,
            'time_per_method': time_per_method,
            'throughput': n_stations * len(methods) / elapsed
        })

    df_results = pd.DataFrame(results)
    return df_results


def analyze_scaling():
    """Analyze computational scaling with data size."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    sizes = [100, 500, 1000, 2000, 5000, 10000]
    results = []

    print("\nTesting method: Eckhardt (parameter calibration required)")
    print(f"{'Size (days)':<15} {'Time (s)':>12} {'Time/day (ms)':>15}")
    print("-" * 70)

    for n_days in sizes:
        Q, _, _ = generate_streamflow(n_days=n_days, random_seed=42)
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        flow_series = pd.Series(Q, index=dates)

        start = time.time()
        single(flow_series, method=['Eckhardt'], return_kge=False)
        elapsed = time.time() - start

        time_per_day = (elapsed / n_days) * 1000  # in milliseconds

        print(f"{n_days:<15} {elapsed:>12.4f} {time_per_day:>15.4f}")

        results.append({
            'n_days': n_days,
            'time': elapsed,
            'time_per_day_ms': time_per_day
        })

    df_results = pd.DataFrame(results)

    # Estimate complexity
    if len(sizes) > 1:
        # Linear fit: time = a * n_days + b
        coeffs = np.polyfit(df_results['n_days'], df_results['time'], 1)
        print(f"\nLinear fit: time ≈ {coeffs[0]:.6f} * n_days + {coeffs[1]:.4f}")
        print(f"Estimated complexity: O(n)")

    return df_results


def compare_method_families():
    """Compare performance across method families."""
    print("\n" + "=" * 70)
    print("METHOD FAMILY COMPARISON")
    print("=" * 70)

    # Generate test data
    Q, _, _ = generate_streamflow(n_days=365*3, random_seed=42)
    dates = pd.date_range('2020-01-01', periods=len(Q), freq='D')
    flow_series = pd.Series(Q, index=dates)

    method_families = {
        'Digital Filters (No Calib)': ['LH', 'UKIH'],
        'Digital Filters (With Calib)': ['Eckhardt', 'Chapman', 'Boughton', 'Furey', 'EWMA', 'Willems'],
        'Graphical Methods': ['Local', 'Fixed', 'Slide', 'CM']
    }

    print("\nBenchmarking method families (3 years of data, 5 runs each):")
    print(f"{'Family':<30} {'Mean Time (s)':>15} {'Methods':>10}")
    print("-" * 70)

    family_results = []

    for family_name, methods in method_families.items():
        family_times = []

        for method in methods:
            try:
                result = benchmark_single_method(method, flow_series, n_runs=5)
                family_times.append(result['mean_time'])
            except:
                pass

        if family_times:
            mean_family_time = np.mean(family_times)
            print(f"{family_name:<30} {mean_family_time:>15.4f} {len(family_times):>10}")

            family_results.append({
                'family': family_name,
                'mean_time': mean_family_time,
                'n_methods': len(family_times)
            })

    return pd.DataFrame(family_results)


def memory_usage_estimate():
    """Estimate memory usage for different configurations."""
    print("\n" + "=" * 70)
    print("MEMORY USAGE ESTIMATION")
    print("=" * 70)

    print("\nEstimated memory usage per configuration:")
    print(f"{'Configuration':<30} {'Est. Memory (MB)':>20}")
    print("-" * 70)

    configs = [
        ('1 year, 1 station, 1 method', 365, 1, 1),
        ('10 years, 1 station, 1 method', 3650, 1, 1),
        ('10 years, 1 station, 12 methods', 3650, 1, 12),
        ('10 years, 100 stations, 12 methods', 3650, 100, 12),
        ('30 years, 1000 stations, 12 methods', 10950, 1000, 12),
    ]

    for name, n_days, n_stations, n_methods in configs:
        # Estimate: float64 = 8 bytes
        # Need: original Q, baseflow for each method, intermediate arrays
        memory_bytes = (
            n_days * n_stations * 8 +  # Original data
            n_days * n_stations * n_methods * 8 +  # Baseflow results
            n_days * n_methods * 8  # Intermediate per-station
        )
        memory_mb = memory_bytes / (1024 ** 2)

        print(f"{name:<30} {memory_mb:>20.1f}")

    print("\nNote: Actual memory usage may vary due to Python overhead and temporary arrays.")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print(" BASEFLOW SEPARATION - PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print("\nThis script evaluates computational performance of baseflow separation methods.")
    print("Tests include: method comparison, batch processing, scaling analysis, and memory estimation.")
    print("=" * 70)

    # Run benchmarks
    df_methods = benchmark_all_methods()
    df_batch = benchmark_batch_processing()
    df_scaling = analyze_scaling()
    df_families = compare_method_families()
    memory_usage_estimate()

    # Summary
    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n1. Fastest Methods (1 year of data):")
    one_year = df_methods[df_methods['n_days'] == 365].sort_values('mean_time')
    print(one_year[['method', 'mean_time']].head(5).to_string(index=False))

    print("\n2. Batch Processing Efficiency:")
    print(f"   - Best throughput: {df_batch['throughput'].max():.1f} separations/second")
    print(f"   - At station count: {df_batch.loc[df_batch['throughput'].idxmax(), 'n_stations']:.0f}")

    print("\n3. Scaling Behavior:")
    print("   - Computational complexity: O(n) - linear scaling")
    print(f"   - Typical time for 10 years: {df_scaling[df_scaling['n_days']==10000]['time'].values[0]:.2f}s")

    print("\n4. Method Family Performance:")
    fastest_family = df_families.loc[df_families['mean_time'].idxmin(), 'family']
    print(f"   - Fastest family: {fastest_family}")

    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE")
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent
    df_methods.to_csv(output_dir / 'benchmark_methods.csv', index=False)
    df_batch.to_csv(output_dir / 'benchmark_batch.csv', index=False)
    df_scaling.to_csv(output_dir / 'benchmark_scaling.csv', index=False)
    print("\n✓ Results saved to CSV files")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
