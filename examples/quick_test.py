"""Quick test script for baseflow separation with random data.

This script provides a fast way to test baseflow separation methods using
randomly generated synthetic data. Perfect for quick validation and experimentation.

Usage:
    python quick_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import single
from baseflow.synthetic_data import generate_streamflow


def run_quick_test(n_days=365, random_seed=None):
    """Run a quick test of baseflow separation methods.

    Args:
        n_days: Number of days to generate
        random_seed: Random seed for reproducibility (None for random)
    """
    print("=" * 70)
    print("QUICK TEST: Baseflow Separation with Random Data")
    print("=" * 70)

    # Generate random parameters
    if random_seed is None:
        random_seed = np.random.randint(0, 10000)
        print(f"\nUsing random seed: {random_seed}")

    np.random.seed(random_seed)

    # Random hydrological characteristics
    base_flow = np.random.uniform(5, 30)
    seasonal_amplitude = np.random.uniform(2, base_flow * 0.5)
    n_storm_events = np.random.randint(15, 50)
    storm_intensity = np.random.uniform(20, 100)
    target_bfi = np.random.uniform(0.4, 0.8)

    print("\n1. Generating random synthetic streamflow data...")
    print(f"   - Number of days: {n_days}")
    print(f"   - Base flow: {base_flow:.2f} m³/s")
    print(f"   - Seasonal amplitude: {seasonal_amplitude:.2f} m³/s")
    print(f"   - Number of storm events: {n_storm_events}")
    print(f"   - Storm intensity: {storm_intensity:.2f} m³/s")
    print(f"   - Target BFI: {target_bfi:.3f}")

    # Generate streamflow
    Q, true_baseflow, quickflow = generate_streamflow(
        n_days=n_days,
        base_flow=base_flow,
        seasonal_amplitude=seasonal_amplitude,
        n_storm_events=n_storm_events,
        storm_intensity=storm_intensity,
        bfi=target_bfi,
        random_seed=random_seed
    )

    # Create pandas Series
    start_date = pd.Timestamp.now().floor('D') - pd.Timedelta(days=n_days)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    flow_series = pd.Series(Q, index=dates)

    print("\n2. Streamflow statistics:")
    print(f"   - Mean: {Q.mean():.2f} m³/s")
    print(f"   - Std: {Q.std():.2f} m³/s")
    print(f"   - Min: {Q.min():.2f} m³/s")
    print(f"   - Max: {Q.max():.2f} m³/s")
    print(f"   - True BFI: {true_baseflow.sum() / Q.sum():.3f}")

    # Apply baseflow separation with popular methods
    print("\n3. Applying baseflow separation methods...")
    methods = ["LH", "Chapman", "Eckhardt", "UKIH", "Boughton"]

    baseflow_df, kge_scores = single(
        flow_series,
        area=np.random.uniform(500, 5000),  # Random drainage area
        method=methods,
        return_kge=True
    )

    print(f"   - Applied {len(methods)} methods successfully")

    # Evaluate results
    print("\n4. Performance Evaluation:")
    print("   " + "-" * 66)
    print(f"   {'Method':<15} {'KGE Score':>12} {'Est. BFI':>12} {'BFI Error':>12}")
    print("   " + "-" * 66)

    true_bfi = true_baseflow.sum() / Q.sum()
    results = []

    for method in methods:
        kge = kge_scores[method]
        estimated_bfi = baseflow_df[method].sum() / Q.sum()
        bfi_error = abs(estimated_bfi - true_bfi)

        results.append({
            'method': method,
            'kge': kge,
            'bfi': estimated_bfi,
            'error': bfi_error
        })

        print(f"   {method:<15} {kge:>12.4f} {estimated_bfi:>12.4f} {bfi_error:>12.4f}")

    print("   " + "-" * 66)

    # Find best method
    best_kge = max(results, key=lambda x: x['kge'])
    best_bfi = min(results, key=lambda x: x['error'])

    print(f"\n5. Best Performing Methods:")
    print(f"   - Highest KGE: {best_kge['method']} (KGE = {best_kge['kge']:.4f})")
    print(f"   - Best BFI: {best_bfi['method']} (error = {best_bfi['error']:.4f})")

    # Additional statistics
    print(f"\n6. Summary Statistics:")
    print(f"   - Mean estimated BFI: {np.mean([r['bfi'] for r in results]):.4f}")
    print(f"   - Std of estimated BFI: {np.std([r['bfi'] for r in results]):.4f}")
    print(f"   - Mean KGE: {np.mean([r['kge'] for r in results]):.4f}")
    print(f"   - Mean BFI error: {np.mean([r['error'] for r in results]):.4f}")

    # Quick flow statistics
    storm_days = (quickflow > 1.0).sum()
    print(f"\n7. Quick Flow Statistics:")
    print(f"   - Days with storms: {storm_days} ({storm_days/n_days*100:.1f}%)")
    print(f"   - Mean quickflow (storm days): {quickflow[quickflow > 1.0].mean():.2f} m³/s")
    print(f"   - Peak quickflow: {quickflow.max():.2f} m³/s")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        'streamflow': flow_series,
        'baseflow': baseflow_df,
        'true_baseflow': true_baseflow,
        'kge_scores': kge_scores,
        'results': results
    }


def run_multiple_tests(n_tests=5, n_days=365):
    """Run multiple tests with different random data.

    Args:
        n_tests: Number of tests to run
        n_days: Number of days per test
    """
    print("\n" + "=" * 70)
    print(f"RUNNING {n_tests} RANDOM TESTS")
    print("=" * 70)

    all_results = []

    for i in range(n_tests):
        print(f"\n{'='*70}")
        print(f"Test {i+1}/{n_tests}")
        print(f"{'='*70}")

        result = run_quick_test(n_days=n_days, random_seed=None)
        all_results.append(result)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL TESTS")
    print("=" * 70)

    methods = result['baseflow'].columns.tolist()

    print(f"\nAverage Performance Across {n_tests} Tests:")
    print("-" * 70)
    print(f"{'Method':<15} {'Mean KGE':>12} {'Std KGE':>12} {'Mean BFI Err':>15}")
    print("-" * 70)

    for method in methods:
        kges = [r['kge_scores'][method] for r in all_results]
        errors = [abs(r['baseflow'][method].sum() / r['streamflow'].sum() -
                      r['true_baseflow'].sum() / r['streamflow'].sum())
                  for r in all_results]

        mean_kge = np.mean(kges)
        std_kge = np.std(kges)
        mean_error = np.mean(errors)

        print(f"{method:<15} {mean_kge:>12.4f} {std_kge:>12.4f} {mean_error:>15.4f}")

    print("-" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quick test for baseflow separation')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days to generate (default: 365)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    parser.add_argument('--multiple', type=int, default=None,
                        help='Run multiple tests with different random data')

    args = parser.parse_args()

    if args.multiple:
        run_multiple_tests(n_tests=args.multiple, n_days=args.days)
    else:
        run_quick_test(n_days=args.days, random_seed=args.seed)
