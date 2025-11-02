"""Comprehensive example demonstrating baseflow separation workflow.

This script showcases:
1. Synthetic data generation
2. Single-station baseflow separation
3. Multi-station batch processing
4. Performance evaluation and visualization
5. Method comparison

Run this script to understand how to use the baseflow separation package.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import separation, single
from baseflow.synthetic_data import create_test_dataframe, generate_streamflow


def example_1_single_station():
    """Example 1: Basic single-station baseflow separation."""
    print("=" * 70)
    print("EXAMPLE 1: Single Station Baseflow Separation")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic streamflow data...")
    Q, true_baseflow, _ = generate_streamflow(
        n_days=365,
        base_flow=15.0,
        seasonal_amplitude=5.0,
        n_storm_events=25,
        storm_intensity=60.0,
        bfi=0.65,
        random_seed=42
    )

    # Create pandas Series
    dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
    flow_series = pd.Series(Q, index=dates, name="Streamflow")

    print(f"   - Generated {len(Q)} days of data")
    print(f"   - Mean flow: {Q.mean():.2f} m³/s")
    print(f"   - True BFI: {true_baseflow.sum() / Q.sum():.3f}")

    # Apply baseflow separation
    print("\n2. Applying baseflow separation methods...")
    methods = ["LH", "Chapman", "Eckhardt", "UKIH"]
    baseflow_df, kge_scores = single(
        flow_series,
        area=1000.0,  # km²
        method=methods,
        return_kge=True
    )

    print(f"   - Applied {len(methods)} methods")
    print(f"   - Methods: {', '.join(methods)}")

    # Evaluate results
    print("\n3. Performance Metrics (KGE scores):")
    for method in methods:
        kge = kge_scores[method]
        estimated_bfi = baseflow_df[method].sum() / Q.sum()
        true_bfi = true_baseflow.sum() / Q.sum()
        bfi_error = abs(estimated_bfi - true_bfi)

        print(f"   - {method:12s}: KGE = {kge:6.3f}, BFI = {estimated_bfi:.3f} (error: {bfi_error:.3f})")

    # Find best method
    best_method = kge_scores.idxmax()
    print(f"\n   Best method: {best_method} (KGE = {kge_scores[best_method]:.3f})")

    return flow_series, baseflow_df, true_baseflow, dates


def example_2_visualization(flow_series, baseflow_df, true_baseflow, dates):
    """Example 2: Visualize separation results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Visualization")
    print("=" * 70)

    print("\n1. Creating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Full time series comparison
    ax = axes[0]
    ax.plot(dates, flow_series.values, 'k-', label='Total Streamflow', alpha=0.7, linewidth=1)
    ax.plot(dates, true_baseflow, 'b--', label='True Baseflow', linewidth=2)
    for method in baseflow_df.columns[:3]:  # Plot first 3 methods
        ax.plot(dates, baseflow_df[method], label=f'{method} (separated)', alpha=0.7, linewidth=1.5)

    ax.set_ylabel('Flow (m³/s)', fontsize=11)
    ax.set_title('Baseflow Separation Results - Full Time Series', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoom on recession period
    ax = axes[1]
    zoom_start, zoom_end = 100, 150
    zoom_dates = dates[zoom_start:zoom_end]

    ax.plot(zoom_dates, flow_series.values[zoom_start:zoom_end], 'k-',
            label='Total Streamflow', alpha=0.7, linewidth=2)
    ax.plot(zoom_dates, true_baseflow[zoom_start:zoom_end], 'b--',
            label='True Baseflow', linewidth=2)
    for method in baseflow_df.columns[:3]:
        ax.plot(zoom_dates, baseflow_df[method].values[zoom_start:zoom_end],
                label=f'{method}', alpha=0.8, linewidth=1.5)

    ax.set_ylabel('Flow (m³/s)', fontsize=11)
    ax.set_title('Detailed View - Recession Period', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Flow duration curves
    ax = axes[2]
    flow_sorted = np.sort(flow_series.values)[::-1]
    exceedance = np.arange(1, len(flow_sorted) + 1) / len(flow_sorted) * 100

    ax.semilogy(exceedance, flow_sorted, 'k-', label='Total Streamflow', linewidth=2)
    ax.semilogy(exceedance, np.sort(true_baseflow)[::-1], 'b--',
                label='True Baseflow', linewidth=2)

    for method in baseflow_df.columns[:3]:
        baseflow_sorted = np.sort(baseflow_df[method].values)[::-1]
        ax.semilogy(exceedance, baseflow_sorted, label=f'{method}', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Exceedance Probability (%)', fontsize=11)
    ax.set_ylabel('Flow (m³/s)', fontsize=11)
    ax.set_title('Flow Duration Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_file = Path(__file__).parent / "separation_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   - Plot saved to: {output_file}")

    plt.close()


def example_3_multi_station():
    """Example 3: Multi-station batch processing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Station Batch Processing")
    print("=" * 70)

    # Generate multi-station data
    print("\n1. Generating multi-station dataset...")
    flow_df, true_baseflow_df, station_info = create_test_dataframe(
        n_days=365,
        n_stations=5,
        start_date="2020-01-01",
        random_seed=123
    )

    print(f"   - Generated {flow_df.shape[1]} stations")
    print(f"   - Time period: {flow_df.shape[0]} days")
    print(f"\n   Station Information:")
    print(station_info[['area', 'base_flow', 'bfi']].round(2).to_string())

    # Apply separation to all stations
    print("\n2. Running batch baseflow separation...")
    methods = ["LH", "Eckhardt", "Chapman"]
    results, bfi_df, kge_df = separation(
        flow_df,
        df_sta=station_info,
        method=methods,
        return_bfi=True,
        return_kge=True
    )

    print(f"   - Processed {len(flow_df.columns)} stations")
    print(f"   - Applied {len(methods)} methods")

    # Summary statistics
    print("\n3. Baseflow Index (BFI) Results:")
    print(bfi_df.round(3).to_string())

    print("\n4. KGE Performance Scores:")
    print(kge_df.round(3).to_string())

    # Compare with true BFI
    print("\n5. BFI Validation (comparing with true BFI):")
    for station in flow_df.columns:
        true_bfi = true_baseflow_df[station].sum() / flow_df[station].sum()
        print(f"\n   {station}:")
        print(f"      True BFI: {true_bfi:.3f}")
        for method in methods:
            estimated_bfi = bfi_df.loc[station, method]
            error = abs(estimated_bfi - true_bfi)
            print(f"      {method:12s}: {estimated_bfi:.3f} (error: {error:.3f})")

    return results, bfi_df, kge_df, station_info


def example_4_parameter_sensitivity():
    """Example 4: Demonstrate parameter configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Parameter Sensitivity Analysis")
    print("=" * 70)

    from baseflow.config import DEFAULT_PARAM_RANGES, update_param_range

    print("\n1. Default parameter ranges:")
    for method, config in DEFAULT_PARAM_RANGES.items():
        if config.param_range is not None:
            print(f"   - {method:12s}: {len(config.param_range)} values, "
                  f"range [{config.param_range.min():.4f}, {config.param_range.max():.4f}]")

    # Demonstrate parameter customization
    print("\n2. Customizing Eckhardt parameter range...")
    print("   - Default: np.arange(0.001, 1.0, 0.001) -> 999 values")

    # Update to coarser grid for faster computation
    update_param_range("Eckhardt", start=0.01, stop=0.99, step=0.01)
    print("   - Updated: np.arange(0.01, 0.99, 0.01) -> 98 values")
    print("   - Trade-off: Faster computation vs. slightly less precise calibration")

    # Generate test data
    Q, _, _ = generate_streamflow(n_days=180, random_seed=99)
    dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
    flow_series = pd.Series(Q, index=dates)

    # Test with different parameter settings
    print("\n3. Testing parameter sensitivity...")
    import time

    start_time = time.time()
    baseflow_df, kge_scores = single(flow_series, method=["Eckhardt"], return_kge=True)
    elapsed = time.time() - start_time

    print(f"   - Separation completed in {elapsed:.3f} seconds")
    print(f"   - Eckhardt KGE: {kge_scores['Eckhardt']:.3f}")


def example_5_method_comparison():
    """Example 5: Comprehensive method comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comprehensive Method Comparison")
    print("=" * 70)

    # Generate test data with varying characteristics
    print("\n1. Testing methods on different flow regimes...")

    regimes = {
        "Perennial (high BFI)": {"base_flow": 20, "n_storm_events": 10, "bfi": 0.75},
        "Intermittent (medium BFI)": {"base_flow": 10, "n_storm_events": 25, "bfi": 0.55},
        "Ephemeral (low BFI)": {"base_flow": 5, "n_storm_events": 40, "bfi": 0.35},
    }

    results_summary = []

    for regime_name, params in regimes.items():
        print(f"\n   Testing: {regime_name}")

        Q, true_baseflow, _ = generate_streamflow(
            n_days=365,
            base_flow=params["base_flow"],
            n_storm_events=params["n_storm_events"],
            bfi=params["bfi"],
            random_seed=42
        )

        dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
        flow_series = pd.Series(Q, index=dates)

        # Test all methods
        baseflow_df, kge_scores = single(
            flow_series,
            area=1000.0,
            method="all",
            return_kge=True
        )

        true_bfi = true_baseflow.sum() / Q.sum()

        for method in baseflow_df.columns:
            estimated_bfi = baseflow_df[method].sum() / Q.sum()
            results_summary.append({
                "Regime": regime_name,
                "Method": method,
                "KGE": kge_scores[method],
                "BFI_error": abs(estimated_bfi - true_bfi)
            })

    # Create summary table
    summary_df = pd.DataFrame(results_summary)

    print("\n2. Performance Summary by Regime:")
    for regime in regimes.keys():
        regime_data = summary_df[summary_df["Regime"] == regime]
        print(f"\n   {regime}:")
        print(f"      Best KGE: {regime_data.loc[regime_data['KGE'].idxmax(), 'Method']} "
              f"(KGE = {regime_data['KGE'].max():.3f})")
        print(f"      Best BFI: {regime_data.loc[regime_data['BFI_error'].idxmin(), 'Method']} "
              f"(error = {regime_data['BFI_error'].min():.3f})")

    print("\n3. Overall Method Ranking (by mean KGE):")
    method_ranking = summary_df.groupby("Method")["KGE"].mean().sort_values(ascending=False)
    for i, (method, mean_kge) in enumerate(method_ranking.items(), 1):
        print(f"      {i:2d}. {method:12s}: {mean_kge:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" COMPREHENSIVE BASEFLOW SEPARATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the complete workflow for baseflow separation")
    print("using synthetic data with known ground truth.\n")

    # Run examples
    flow_series, baseflow_df, true_baseflow, dates = example_1_single_station()
    example_2_visualization(flow_series, baseflow_df, true_baseflow, dates)
    example_3_multi_station()
    example_4_parameter_sensitivity()
    example_5_method_comparison()

    # Summary
    print("\n" + "=" * 70)
    print(" EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Multiple methods available (digital filters, graphical, parameterized)")
    print("  2. Automatic parameter calibration for optimal performance")
    print("  3. Built-in evaluation using KGE metric")
    print("  4. Supports both single and multi-station processing")
    print("  5. Flexible parameter configuration for speed/accuracy trade-offs")
    print("\nFor real data analysis, replace synthetic data with your own streamflow time series.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
