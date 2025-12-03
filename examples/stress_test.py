"""Comprehensive robustness testing for baseflow separation methods.

This script performs stress tests to validate the robustness of baseflow separation
algorithms under various edge cases and challenging scenarios:

1. Data with missing values (gaps)
2. Zero flow periods
3. Massive batch processing (performance benchmarking)

Author: Baseflow Separation Team
Date: 2025-12-03
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow.separation import separation, single
from baseflow.synthetic_data import generate_streamflow, create_test_dataframe


class StressTestRunner:
    """Runner for baseflow separation stress tests."""

    def __init__(self, verbose: bool = True):
        """Initialize stress test runner.

        Args:
            verbose: Whether to print detailed test results
        """
        self.verbose = verbose
        self.test_results = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled.

        Args:
            message: Message to log
            level: Log level (INFO, SUCCESS, WARNING, ERROR)
        """
        if self.verbose:
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌"
            }.get(level, "")
            print(f"{prefix} {message}")

    def assert_test(self, condition: bool, test_name: str, message: str):
        """Assert a test condition and record the result.

        Args:
            condition: Boolean condition to test
            test_name: Name of the test
            message: Description of what is being tested
        """
        if condition:
            self.log(f"PASS: {test_name} - {message}", "SUCCESS")
            self.test_results.append((test_name, True, message))
        else:
            self.log(f"FAIL: {test_name} - {message}", "ERROR")
            self.test_results.append((test_name, False, message))
            raise AssertionError(f"Test failed: {test_name} - {message}")

    def scenario_1_gaps_handling(self):
        """Scenario 1: Test handling of missing values (NaN gaps).

        This test verifies that the separation function can handle data with
        random missing values. The expected behavior is:
        - The function should either interpolate NaN values
        - Or skip them and return NaN for those positions
        - It should NOT crash or produce invalid results
        """
        self.log("\n" + "=" * 70)
        self.log("SCENARIO 1: Missing Values (NaN Gaps) Handling", "INFO")
        self.log("=" * 70)

        # Generate synthetic data
        Q, B_true, QF = generate_streamflow(
            n_days=365,
            base_flow=15.0,
            n_storm_events=20,
            bfi=0.65,
            random_seed=42
        )

        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        flow_series = pd.Series(Q, index=dates, name='flow')

        # Randomly insert NaN values (10% of data)
        n_gaps = int(0.1 * len(flow_series))
        gap_indices = np.random.choice(len(flow_series), size=n_gaps, replace=False)
        flow_with_gaps = flow_series.copy()
        flow_with_gaps.iloc[gap_indices] = np.nan

        self.log(f"Original data length: {len(flow_series)}", "INFO")
        self.log(f"Inserted {n_gaps} NaN values ({n_gaps/len(flow_series)*100:.1f}%)", "INFO")

        # Test separation with gaps
        try:
            baseflow, kge = single(
                flow_with_gaps,
                method=["LH", "Eckhardt"],
                return_kge=True
            )

            # Check that results are produced
            self.assert_test(
                baseflow is not None,
                "Scenario 1.1",
                "Function returns results with NaN input"
            )

            # Check that baseflow has same length as input
            self.assert_test(
                len(baseflow) == len(flow_with_gaps),
                "Scenario 1.2",
                "Output length matches input length"
            )

            # Check that non-NaN positions have valid baseflow
            valid_positions = ~flow_with_gaps.isna()
            if valid_positions.sum() > 0:
                # Check that baseflow at valid positions is not all NaN
                has_valid_output = baseflow.loc[valid_positions].notna().any().any()
                self.assert_test(
                    has_valid_output,
                    "Scenario 1.3",
                    "Valid positions produce non-NaN baseflow"
                )

            # Check that baseflow doesn't exceed streamflow (where both are valid)
            for method in baseflow.columns:
                valid_comparison = valid_positions & baseflow[method].notna()
                if valid_comparison.sum() > 0:
                    exceeds = (baseflow.loc[valid_comparison, method] >
                              flow_with_gaps.loc[valid_comparison]).any()
                    self.assert_test(
                        not exceeds,
                        f"Scenario 1.4 ({method})",
                        f"{method}: Baseflow ≤ streamflow constraint maintained"
                    )

            self.log(f"✓ All {len(baseflow.columns)} methods handled gaps successfully", "SUCCESS")

        except Exception as e:
            self.log(f"Exception during gap handling: {e}", "ERROR")
            self.assert_test(
                False,
                "Scenario 1.5",
                f"No exceptions should occur (got: {type(e).__name__})"
            )

    def scenario_2_zero_flow(self):
        """Scenario 2: Test handling of zero flow periods.

        This test verifies that:
        - Baseflow output is also 0 during zero flow periods
        - Baseflow is never negative
        - The algorithm doesn't produce artifacts after zero flow periods
        """
        self.log("\n" + "=" * 70)
        self.log("SCENARIO 2: Zero Flow Period Handling", "INFO")
        self.log("=" * 70)

        # Generate synthetic data
        Q, B_true, QF = generate_streamflow(
            n_days=365,
            base_flow=15.0,
            n_storm_events=20,
            bfi=0.65,
            random_seed=123
        )

        dates = pd.date_range('2020-01-01', periods=365, freq='D')

        # Insert zero flow period (30 days in the middle)
        zero_start = 150
        zero_end = 180
        Q[zero_start:zero_end] = 0.0

        flow_series = pd.Series(Q, index=dates, name='flow')

        self.log(f"Data length: {len(flow_series)}", "INFO")
        self.log(f"Zero flow period: days {zero_start} to {zero_end} ({zero_end-zero_start} days)", "INFO")

        # Test separation with zero flow
        try:
            baseflow, kge = single(
                flow_series,
                method=["LH", "Eckhardt", "Chapman"],
                return_kge=True
            )

            # Test 1: Baseflow should be 0 during zero flow period
            for method in baseflow.columns:
                zero_period_baseflow = baseflow.iloc[zero_start:zero_end][method]
                all_zero_or_near = (zero_period_baseflow <= 1e-10).all()
                self.assert_test(
                    all_zero_or_near,
                    f"Scenario 2.1 ({method})",
                    f"{method}: Baseflow ≈ 0 during zero flow period"
                )

            # Test 2: Baseflow should never be negative
            for method in baseflow.columns:
                has_negative = (baseflow[method] < -1e-10).any()
                self.assert_test(
                    not has_negative,
                    f"Scenario 2.2 ({method})",
                    f"{method}: No negative baseflow values"
                )

            # Test 3: Baseflow should not exceed streamflow anywhere
            for method in baseflow.columns:
                exceeds = (baseflow[method] > flow_series + 1e-10).any()
                self.assert_test(
                    not exceeds,
                    f"Scenario 2.3 ({method})",
                    f"{method}: Baseflow ≤ streamflow everywhere"
                )

            # Test 4: Algorithm should recover after zero flow period
            # Check that baseflow is positive after the zero period ends
            recovery_period = baseflow.iloc[zero_end:zero_end+30]
            for method in baseflow.columns:
                has_recovery = (recovery_period[method] > 1e-10).any()
                self.assert_test(
                    has_recovery,
                    f"Scenario 2.4 ({method})",
                    f"{method}: Algorithm recovers after zero flow period"
                )

            self.log(f"✓ All {len(baseflow.columns)} methods handled zero flow correctly", "SUCCESS")

        except Exception as e:
            self.log(f"Exception during zero flow test: {e}", "ERROR")
            self.assert_test(
                False,
                "Scenario 2.5",
                f"No exceptions should occur (got: {type(e).__name__})"
            )

    def scenario_3_massive_batch(self):
        """Scenario 3: Performance benchmark with massive batch processing.

        This test generates a large dataset with 100 stations and benchmarks
        the performance of the separation function with parallel processing.
        """
        self.log("\n" + "=" * 70)
        self.log("SCENARIO 3: Massive Batch Processing Benchmark", "INFO")
        self.log("=" * 70)

        n_stations = 100
        n_days = 365

        self.log(f"Generating synthetic data: {n_stations} stations × {n_days} days", "INFO")

        # Generate multi-station test data
        flow_df, baseflow_df, station_info = create_test_dataframe(
            n_days=n_days,
            n_stations=n_stations,
            random_seed=42
        )

        self.log(f"Data shape: {flow_df.shape}", "INFO")
        self.log(f"Memory usage: {flow_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "INFO")

        # Benchmark parallel processing
        methods_to_test = ["LH", "Eckhardt", "Chapman"]

        self.log(f"\nBenchmarking with methods: {methods_to_test}", "INFO")
        self.log("Running separation with parallel processing (n_jobs=-1)...", "INFO")

        start_time = time.time()

        try:
            results, bfi, kge = separation(
                flow_df,
                df_sta=station_info,
                method=methods_to_test,
                return_bfi=True,
                return_kge=True,
                n_jobs=-1  # Use all available cores
            )

            elapsed_time = time.time() - start_time

            # Test 1: All methods should produce results
            self.assert_test(
                len(results) == len(methods_to_test),
                "Scenario 3.1",
                f"All {len(methods_to_test)} methods produced results"
            )

            # Test 2: Results should have correct shape
            for method, result_df in results.items():
                correct_shape = result_df.shape == flow_df.shape
                self.assert_test(
                    correct_shape,
                    f"Scenario 3.2 ({method})",
                    f"{method}: Output shape {result_df.shape} matches input shape {flow_df.shape}"
                )

            # Test 3: BFI values should be in valid range [0, 1]
            bfi_valid = ((bfi >= 0) & (bfi <= 1)).all().all()
            self.assert_test(
                bfi_valid,
                "Scenario 3.3",
                "All BFI values in valid range [0, 1]"
            )

            # Test 4: No NaN in results (unless input had NaN)
            for method, result_df in results.items():
                # Count NaN values
                n_nan_output = result_df.isna().sum().sum()
                n_nan_input = flow_df.isna().sum().sum()
                # Output NaN should not exceed input NaN significantly
                nan_reasonable = n_nan_output <= n_nan_input * 1.1  # Allow 10% tolerance
                self.assert_test(
                    nan_reasonable,
                    f"Scenario 3.4 ({method})",
                    f"{method}: NaN count reasonable (input: {n_nan_input}, output: {n_nan_output})"
                )

            # Performance metrics
            self.log("\n" + "-" * 70, "INFO")
            self.log("PERFORMANCE METRICS", "INFO")
            self.log("-" * 70, "INFO")
            self.log(f"Total processing time: {elapsed_time:.2f} seconds", "INFO")
            self.log(f"Time per station: {elapsed_time / n_stations * 1000:.1f} ms", "INFO")
            self.log(f"Throughput: {n_stations / elapsed_time:.1f} stations/second", "INFO")
            self.log(f"Data processed: {n_stations * n_days * len(methods_to_test):,} data points", "INFO")

            # Test 5: Performance should be reasonable (< 60 seconds for 100 stations)
            performance_acceptable = elapsed_time < 60.0
            self.assert_test(
                performance_acceptable,
                "Scenario 3.5",
                f"Processing time ({elapsed_time:.2f}s) is acceptable (< 60s)"
            )

            self.log(f"\n✓ Batch processing completed successfully", "SUCCESS")

        except Exception as e:
            self.log(f"Exception during batch processing: {e}", "ERROR")
            self.assert_test(
                False,
                "Scenario 3.6",
                f"No exceptions should occur (got: {type(e).__name__}: {e})"
            )

    def run_all_tests(self):
        """Run all stress test scenarios."""
        self.log("\n" + "=" * 70)
        self.log("🧪 BASEFLOW SEPARATION STRESS TEST SUITE", "INFO")
        self.log("=" * 70)
        self.log(f"Test started at: {pd.Timestamp.now()}", "INFO")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Run all scenarios
        scenarios = [
            ("Scenario 1: Gaps Handling", self.scenario_1_gaps_handling),
            ("Scenario 2: Zero Flow", self.scenario_2_zero_flow),
            ("Scenario 3: Massive Batch", self.scenario_3_massive_batch),
        ]

        failed_scenarios = []

        for scenario_name, scenario_func in scenarios:
            try:
                scenario_func()
            except AssertionError as e:
                self.log(f"\n{scenario_name} FAILED: {e}", "ERROR")
                failed_scenarios.append(scenario_name)
            except Exception as e:
                self.log(f"\n{scenario_name} CRASHED: {type(e).__name__}: {e}", "ERROR")
                failed_scenarios.append(scenario_name)

        # Print summary
        self.print_summary(failed_scenarios)

    def print_summary(self, failed_scenarios):
        """Print test summary.

        Args:
            failed_scenarios: List of scenario names that failed
        """
        self.log("\n" + "=" * 70)
        self.log("📊 TEST SUMMARY", "INFO")
        self.log("=" * 70)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, result, _ in self.test_results if result)
        failed_tests = total_tests - passed_tests

        self.log(f"Total tests run: {total_tests}", "INFO")
        self.log(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)", "SUCCESS")
        if failed_tests > 0:
            self.log(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)", "ERROR")
        else:
            self.log(f"Failed: {failed_tests}", "INFO")

        if failed_scenarios:
            self.log("\n⚠️  Failed scenarios:", "WARNING")
            for scenario in failed_scenarios:
                self.log(f"  - {scenario}", "WARNING")
        else:
            self.log("\n🎉 ALL SCENARIOS PASSED! 🎉", "SUCCESS")

        self.log(f"\nTest completed at: {pd.Timestamp.now()}", "INFO")
        self.log("=" * 70)


def main():
    """Main entry point for stress tests."""
    runner = StressTestRunner(verbose=True)
    runner.run_all_tests()


if __name__ == "__main__":
    main()
