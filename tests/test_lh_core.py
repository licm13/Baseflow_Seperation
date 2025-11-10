"""Unit tests for LH core implementation.

Verifies that the centralized lh_filter produces identical results to the
original implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow.methods.LH import LH as LH_original
from baseflow.methods.lh_core import lh_filter
from baseflow.synthetic_data import generate_streamflow


class TestLHCore:
    """Tests for centralized LH core implementation."""

    @pytest.fixture
    def sample_flow(self):
        """Create sample streamflow data."""
        Q, _, _ = generate_streamflow(n_days=365, random_seed=42)
        return Q

    def test_numeric_parity_default(self, sample_flow):
        """Test that lh_filter produces identical results to LH_original with default parameters."""
        result_original = LH_original(sample_flow)
        result_new = lh_filter(sample_flow)

        # Results should be identical (within floating point precision)
        np.testing.assert_array_almost_equal(
            result_original,
            result_new,
            decimal=15,
            err_msg="lh_filter should produce identical results to LH_original"
        )

    def test_numeric_parity_custom_beta(self, sample_flow):
        """Test numeric parity with custom beta parameter."""
        beta = 0.95
        result_original = LH_original(sample_flow, beta=beta)
        result_new = lh_filter(sample_flow, beta=beta)

        np.testing.assert_array_almost_equal(
            result_original,
            result_new,
            decimal=15,
            err_msg=f"lh_filter should produce identical results for beta={beta}"
        )

    def test_numeric_parity_with_exceed(self, sample_flow):
        """Test numeric parity with return_exceed=True."""
        result_original = LH_original(sample_flow, return_exceed=True)
        result_new = lh_filter(sample_flow, return_exceed=True)

        np.testing.assert_array_almost_equal(
            result_original,
            result_new,
            decimal=15,
            err_msg="lh_filter should produce identical results with return_exceed=True"
        )

    def test_multiple_beta_values(self, sample_flow):
        """Test numeric parity across a range of beta values."""
        beta_values = [0.85, 0.90, 0.925, 0.95, 0.98]

        for beta in beta_values:
            result_original = LH_original(sample_flow, beta=beta)
            result_new = lh_filter(sample_flow, beta=beta)

            np.testing.assert_array_almost_equal(
                result_original,
                result_new,
                decimal=15,
                err_msg=f"Results differ for beta={beta}"
            )

    def test_baseflow_constraints(self, sample_flow):
        """Test that lh_filter respects physical constraints."""
        result = lh_filter(sample_flow)

        # Baseflow should never exceed streamflow
        assert np.all(result <= sample_flow + 1e-10), "Baseflow exceeds streamflow"

        # Baseflow should be non-negative
        assert np.all(result >= 0), "Baseflow contains negative values"

    def test_short_series(self):
        """Test with a very short time series."""
        Q = np.array([10.0, 15.0, 12.0, 8.0, 6.0])

        result_original = LH_original(Q)
        result_new = lh_filter(Q)

        np.testing.assert_array_almost_equal(result_original, result_new, decimal=15)

    def test_constant_flow(self):
        """Test with constant flow (edge case)."""
        Q = np.ones(100) * 10.0

        result_original = LH_original(Q)
        result_new = lh_filter(Q)

        np.testing.assert_array_almost_equal(result_original, result_new, decimal=15)

    def test_exceedance_count_consistency(self, sample_flow):
        """Test that exceedance counts match between implementations."""
        result_original = LH_original(sample_flow, return_exceed=True)
        result_new = lh_filter(sample_flow, return_exceed=True)

        # Check that exceedance count (last element) matches
        assert result_original[-1] == result_new[-1], \
            f"Exceedance counts differ: {result_original[-1]} vs {result_new[-1]}"

    def test_different_random_seeds(self):
        """Test numeric parity with different random data."""
        for seed in [0, 1, 42, 100, 999]:
            Q, _, _ = generate_streamflow(n_days=200, random_seed=seed)

            result_original = LH_original(Q)
            result_new = lh_filter(Q)

            np.testing.assert_array_almost_equal(
                result_original,
                result_new,
                decimal=15,
                err_msg=f"Results differ for random seed {seed}"
            )


class TestLHRefactoredIntegration:
    """Test that LH_refactored.LH also uses the centralized implementation."""

    def test_refactored_uses_centralized(self):
        """Verify that LH_refactored.LH delegates to lh_filter."""
        from baseflow.methods.LH_refactored import LH as LH_refactored

        Q, _, _ = generate_streamflow(n_days=100, random_seed=42)

        result_original = LH_original(Q)
        result_refactored = LH_refactored(Q)
        result_core = lh_filter(Q)

        # All should produce identical results
        np.testing.assert_array_almost_equal(result_original, result_refactored, decimal=15)
        np.testing.assert_array_almost_equal(result_refactored, result_core, decimal=15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
