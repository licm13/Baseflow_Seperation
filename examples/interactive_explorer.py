"""Interactive explorer for baseflow separation with random data.

This script provides an interactive way to explore baseflow separation methods
by generating random synthetic data with user-specified parameters.

Usage:
    python interactive_explorer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import single
from baseflow.synthetic_data import generate_streamflow


def generate_random_scenario(scenario_type='balanced'):
    """Generate random scenario parameters based on type.

    Args:
        scenario_type: Type of scenario ('perennial', 'intermittent', 'ephemeral', 'balanced', 'random')

    Returns:
        Dictionary of parameters
    """
    np.random.seed(None)  # Use true random

    if scenario_type == 'perennial':
        # High baseflow, few storms
        params = {
            'base_flow': np.random.uniform(20, 40),
            'seasonal_amplitude': np.random.uniform(5, 15),
            'n_storm_events': np.random.randint(10, 25),
            'storm_intensity': np.random.uniform(30, 60),
            'bfi': np.random.uniform(0.70, 0.85)
        }
    elif scenario_type == 'intermittent':
        # Medium baseflow, moderate storms
        params = {
            'base_flow': np.random.uniform(10, 20),
            'seasonal_amplitude': np.random.uniform(3, 8),
            'n_storm_events': np.random.randint(20, 35),
            'storm_intensity': np.random.uniform(40, 80),
            'bfi': np.random.uniform(0.50, 0.65)
        }
    elif scenario_type == 'ephemeral':
        # Low baseflow, many storms
        params = {
            'base_flow': np.random.uniform(3, 10),
            'seasonal_amplitude': np.random.uniform(1, 5),
            'n_storm_events': np.random.randint(35, 60),
            'storm_intensity': np.random.uniform(50, 120),
            'bfi': np.random.uniform(0.30, 0.45)
        }
    elif scenario_type == 'balanced':
        # Balanced characteristics
        params = {
            'base_flow': np.random.uniform(12, 25),
            'seasonal_amplitude': np.random.uniform(4, 10),
            'n_storm_events': np.random.randint(20, 40),
            'storm_intensity': np.random.uniform(40, 80),
            'bfi': np.random.uniform(0.50, 0.70)
        }
    else:  # random
        # Completely random
        params = {
            'base_flow': np.random.uniform(3, 40),
            'seasonal_amplitude': np.random.uniform(1, 15),
            'n_storm_events': np.random.randint(10, 60),
            'storm_intensity': np.random.uniform(20, 120),
            'bfi': np.random.uniform(0.30, 0.85)
        }

    return params


def print_scenario_info(params, scenario_type, n_days):
    """Print scenario information."""
    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario_type.upper()}")
    print("=" * 70)
    print(f"\nHydrological Parameters:")
    print(f"  Time period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Base flow: {params['base_flow']:.2f} m³/s")
    print(f"  Seasonal amplitude: {params['seasonal_amplitude']:.2f} m³/s")
    print(f"  Number of storm events: {params['n_storm_events']}")
    print(f"  Storm intensity: {params['storm_intensity']:.2f} m³/s")
    print(f"  Target BFI: {params['bfi']:.3f}")


def analyze_scenario(params, n_days=365, methods=None):
    """Analyze a scenario with baseflow separation.

    Args:
        params: Scenario parameters
        n_days: Number of days
        methods: List of methods to apply (None for default set)

    Returns:
        Results dictionary
    """
    if methods is None:
        methods = ['LH', 'Eckhardt', 'Chapman', 'UKIH', 'Boughton']

    # Generate streamflow
    Q, true_baseflow, quickflow = generate_streamflow(
        n_days=n_days,
        base_flow=params['base_flow'],
        seasonal_amplitude=params['seasonal_amplitude'],
        n_storm_events=params['n_storm_events'],
        storm_intensity=params['storm_intensity'],
        bfi=params['bfi'],
        random_seed=None
    )

    # Create series
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    flow_series = pd.Series(Q, index=dates)

    print(f"\nGenerated Streamflow Statistics:")
    print(f"  Mean: {Q.mean():.2f} m³/s")
    print(f"  Median: {np.median(Q):.2f} m³/s")
    print(f"  Std: {Q.std():.2f} m³/s")
    print(f"  Min: {Q.min():.2f} m³/s")
    print(f"  Max: {Q.max():.2f} m³/s")
    print(f"  Actual BFI: {true_baseflow.sum() / Q.sum():.3f}")
    print(f"  Storm days: {(quickflow > 1.0).sum()} ({(quickflow > 1.0).sum()/n_days*100:.1f}%)")

    # Apply separation
    print(f"\nApplying {len(methods)} baseflow separation methods...")
    drainage_area = np.random.uniform(500, 5000)
    baseflow_df, kge_scores = single(
        flow_series,
        area=drainage_area,
        method=methods,
        return_kge=True
    )

    # Analyze results
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'KGE':>8} {'BFI':>8} {'BFI Err':>10} {'RMSE':>10}")
    print("-" * 70)

    true_bfi = true_baseflow.sum() / Q.sum()
    results = []

    for method in methods:
        kge = kge_scores[method]
        estimated = baseflow_df[method].values
        estimated_bfi = estimated.sum() / Q.sum()
        bfi_error = abs(estimated_bfi - true_bfi)
        rmse = np.sqrt(np.mean((estimated - true_baseflow) ** 2))

        results.append({
            'method': method,
            'kge': kge,
            'bfi': estimated_bfi,
            'bfi_error': bfi_error,
            'rmse': rmse
        })

        print(f"{method:<15} {kge:>8.4f} {estimated_bfi:>8.4f} {bfi_error:>10.4f} {rmse:>10.2f}")

    print("-" * 70)

    # Best methods
    best_kge = max(results, key=lambda x: x['kge'])
    best_bfi = min(results, key=lambda x: x['bfi_error'])
    best_rmse = min(results, key=lambda x: x['rmse'])

    print(f"\nBest Performers:")
    print(f"  Highest KGE: {best_kge['method']} (KGE={best_kge['kge']:.4f})")
    print(f"  Best BFI: {best_bfi['method']} (error={best_bfi['bfi_error']:.4f})")
    print(f"  Lowest RMSE: {best_rmse['method']} (RMSE={best_rmse['rmse']:.2f})")

    return {
        'streamflow': flow_series,
        'true_baseflow': true_baseflow,
        'quickflow': quickflow,
        'baseflow_df': baseflow_df,
        'kge_scores': kge_scores,
        'results': results
    }


def run_multiple_random_scenarios(n_scenarios=5, n_days=365):
    """Run multiple random scenarios and aggregate results.

    Args:
        n_scenarios: Number of random scenarios to test
        n_days: Number of days per scenario
    """
    print("\n" + "=" * 70)
    print(f"TESTING {n_scenarios} RANDOM SCENARIOS")
    print("=" * 70)

    scenario_types = ['perennial', 'intermittent', 'ephemeral', 'balanced', 'random']
    methods = ['LH', 'Eckhardt', 'Chapman', 'UKIH', 'Boughton']

    all_results = {method: {'kge': [], 'bfi_error': [], 'rmse': []} for method in methods}

    for i in range(n_scenarios):
        scenario_type = np.random.choice(scenario_types)
        print(f"\n{'='*70}")
        print(f"Scenario {i+1}/{n_scenarios}: {scenario_type.upper()}")
        print(f"{'='*70}")

        params = generate_random_scenario(scenario_type)
        print_scenario_info(params, scenario_type, n_days)

        scenario_results = analyze_scenario(params, n_days, methods)

        # Aggregate results
        for result in scenario_results['results']:
            method = result['method']
            all_results[method]['kge'].append(result['kge'])
            all_results[method]['bfi_error'].append(result['bfi_error'])
            all_results[method]['rmse'].append(result['rmse'])

    # Summary statistics
    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ACROSS {n_scenarios} SCENARIOS")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Mean KGE':>10} {'Mean BFI Err':>14} {'Mean RMSE':>12}")
    print("-" * 70)

    for method in methods:
        mean_kge = np.mean(all_results[method]['kge'])
        mean_bfi_error = np.mean(all_results[method]['bfi_error'])
        mean_rmse = np.mean(all_results[method]['rmse'])

        print(f"{method:<15} {mean_kge:>10.4f} {mean_bfi_error:>14.4f} {mean_rmse:>12.2f}")

    print("-" * 70)

    # Robustness analysis
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Std KGE':>10} {'Std BFI Err':>14} {'Std RMSE':>12}")
    print("-" * 70)

    for method in methods:
        std_kge = np.std(all_results[method]['kge'])
        std_bfi_error = np.std(all_results[method]['bfi_error'])
        std_rmse = np.std(all_results[method]['rmse'])

        print(f"{method:<15} {std_kge:>10.4f} {std_bfi_error:>14.4f} {std_rmse:>12.2f}")

    print("-" * 70)

    # Find most robust method (low std, high mean KGE)
    robustness_scores = {}
    for method in methods:
        mean_kge = np.mean(all_results[method]['kge'])
        std_kge = np.std(all_results[method]['kge'])
        # Robustness score: high mean KGE, low std
        robustness_scores[method] = mean_kge - std_kge

    most_robust = max(robustness_scores, key=robustness_scores.get)

    print(f"\nMost Robust Method: {most_robust}")
    print(f"  (Robustness Score = Mean KGE - Std KGE = {robustness_scores[most_robust]:.4f})")


def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 70)
    print(" INTERACTIVE BASEFLOW SEPARATION EXPLORER")
    print("=" * 70)
    print("\nThis tool helps you explore baseflow separation methods using")
    print("randomly generated synthetic data.")
    print("\nAvailable scenario types:")
    print("  1. Perennial - High baseflow, few storms (BFI: 0.70-0.85)")
    print("  2. Intermittent - Medium baseflow, moderate storms (BFI: 0.50-0.65)")
    print("  3. Ephemeral - Low baseflow, many storms (BFI: 0.30-0.45)")
    print("  4. Balanced - Balanced characteristics (BFI: 0.50-0.70)")
    print("  5. Random - Completely random parameters")
    print("  6. Multiple - Run multiple random scenarios")

    while True:
        print("\n" + "-" * 70)
        choice = input("\nSelect scenario type (1-6, or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            print("\nExiting. Thank you for using the interactive explorer!")
            break

        scenario_map = {
            '1': 'perennial',
            '2': 'intermittent',
            '3': 'ephemeral',
            '4': 'balanced',
            '5': 'random'
        }

        if choice == '6':
            n_scenarios = int(input("Number of scenarios to test (default: 5): ").strip() or "5")
            n_days = int(input("Days per scenario (default: 365): ").strip() or "365")
            run_multiple_random_scenarios(n_scenarios, n_days)
        elif choice in scenario_map:
            scenario_type = scenario_map[choice]
            n_days = int(input("Number of days (default: 365): ").strip() or "365")

            params = generate_random_scenario(scenario_type)
            print_scenario_info(params, scenario_type, n_days)

            analyze_scenario(params, n_days)
        else:
            print("Invalid choice. Please select 1-6 or 'q'.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive explorer for baseflow separation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python interactive_explorer.py

  # Run 10 random scenarios
  python interactive_explorer.py --multiple 10

  # Run single balanced scenario with 730 days
  python interactive_explorer.py --scenario balanced --days 730

  # Run ephemeral scenario
  python interactive_explorer.py --scenario ephemeral
        """
    )

    parser.add_argument('--scenario', choices=['perennial', 'intermittent', 'ephemeral', 'balanced', 'random'],
                        help='Run specific scenario type')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days (default: 365)')
    parser.add_argument('--multiple', type=int,
                        help='Run multiple random scenarios')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    if args.multiple:
        run_multiple_random_scenarios(n_scenarios=args.multiple, n_days=args.days)
    elif args.scenario:
        params = generate_random_scenario(args.scenario)
        print_scenario_info(params, args.scenario, args.days)
        analyze_scenario(params, args.days)
    else:
        # Default to interactive mode if no arguments
        interactive_mode()


if __name__ == "__main__":
    main()
