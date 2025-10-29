"""Unit tests for baseflow separation functions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import separation, single
from baseflow.synthetic_data import create_test_dataframe, generate_streamflow


class TestSingle:
    """Tests for single-station separation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample streamflow data."""
        Q, baseflow, _ = generate_streamflow(n_days=365, random_seed=42)
        dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
        series = pd.Series(Q, index=dates)
        return series, baseflow

    def test_basic_separation(self, sample_data):
        """Test basic separation with default parameters."""
        series, _ = sample_data
        baseflow_df, kge_scores = single(series, method="LH", return_kge=True)

        assert isinstance(baseflow_df, pd.DataFrame)
        assert "LH" in baseflow_df.columns
        assert len(baseflow_df) == len(series)
        assert isinstance(kge_scores, pd.Series)

    def test_multiple_methods(self, sample_data):
        """Test separation with multiple methods."""
        series, _ = sample_data
        methods = ["LH", "Chapman", "Eckhardt"]
        baseflow_df, kge_scores = single(series, method=methods, return_kge=True)

        assert len(baseflow_df.columns) == 3
        assert all(m in baseflow_df.columns for m in methods)
        assert len(kge_scores) == 3

    def test_all_methods(self, sample_data):
        """Test separation with all available methods."""
        series, _ = sample_data
        baseflow_df, kge_scores = single(series, method="all", return_kge=True)

        assert len(baseflow_df.columns) == 12  # All 12 methods
        assert all(kge >= -1 and kge <= 1 for kge in kge_scores)

    def test_no_kge(self, sample_data):
        """Test separation without KGE calculation."""
        series, _ = sample_data
        baseflow_df, kge_scores = single(series, method="LH", return_kge=False)

        assert isinstance(baseflow_df, pd.DataFrame)
        assert kge_scores is None

    def test_with_area(self, sample_data):
        """Test separation with drainage area (for HYSEP methods)."""
        series, _ = sample_data
        baseflow_df, _ = single(series, area=1000, method="Local", return_kge=False)

        assert "Local" in baseflow_df.columns
        assert np.all(np.isfinite(baseflow_df["Local"]))

    def test_baseflow_constraints(self, sample_data):
        """Test that baseflow doesn't exceed streamflow."""
        series, _ = sample_data
        baseflow_df, _ = single(series, method=["LH", "Eckhardt"], return_kge=False)

        for method in baseflow_df.columns:
            assert np.all(baseflow_df[method] <= series.values + 1e-10)

    def test_positive_baseflow(self, sample_data):
        """Test that baseflow is always non-negative."""
        series, _ = sample_data
        baseflow_df, _ = single(series, method="all", return_kge=False)

        for method in baseflow_df.columns:
            assert np.all(baseflow_df[method] >= 0)


class TestSeparation:
    """Tests for multi-station separation."""

    @pytest.fixture
    def sample_multistation_data(self):
        """Create sample multi-station data."""
        flow_df, base_df, info_df = create_test_dataframe(
            n_days=180,
            n_stations=3,
            random_seed=42
        )
        return flow_df, base_df, info_df

    def test_basic_multistation(self, sample_multistation_data):
        """Test basic multi-station separation."""
        flow_df, _, info_df = sample_multistation_data
        results = separation(flow_df, df_sta=info_df, method="LH")

        assert isinstance(results, dict)
        assert "LH" in results
        assert results["LH"].shape == flow_df.shape

    def test_with_bfi(self, sample_multistation_data):
        """Test multi-station with BFI calculation."""
        flow_df, _, info_df = sample_multistation_data
        results, bfi_df = separation(
            flow_df,
            df_sta=info_df,
            method=["LH", "Eckhardt"],
            return_bfi=True
        )

        assert isinstance(bfi_df, pd.DataFrame)
        assert bfi_df.shape == (3, 2)  # 3 stations, 2 methods
        assert np.all(bfi_df >= 0)
        assert np.all(bfi_df <= 1)

    def test_with_kge(self, sample_multistation_data):
        """Test multi-station with KGE calculation."""
        flow_df, _, info_df = sample_multistation_data
        results, kge_df = separation(
            flow_df,
            df_sta=info_df,
            method=["LH"],
            return_kge=True
        )

        assert isinstance(kge_df, pd.DataFrame)
        assert kge_df.shape == (3, 1)  # 3 stations, 1 method

    def test_with_both_metrics(self, sample_multistation_data):
        """Test multi-station with both BFI and KGE."""
        flow_df, _, info_df = sample_multistation_data
        results, bfi_df, kge_df = separation(
            flow_df,
            df_sta=info_df,
            method=["LH", "Eckhardt"],
            return_bfi=True,
            return_kge=True
        )

        assert isinstance(results, dict)
        assert isinstance(bfi_df, pd.DataFrame)
        assert isinstance(kge_df, pd.DataFrame)
        assert len(results) == 2  # 2 methods

    def test_no_station_info(self, sample_multistation_data):
        """Test separation without station info."""
        flow_df, _, _ = sample_multistation_data
        results = separation(flow_df, method="LH")

        assert isinstance(results, dict)
        assert "LH" in results

    def test_progress_tracking(self, sample_multistation_data):
        """Test that processing completes for all stations."""
        flow_df, _, info_df = sample_multistation_data
        results = separation(flow_df, df_sta=info_df, method="LH")

        # All stations should be processed
        assert not results["LH"].isna().all(axis=0).any()


class TestValidation:
    """Tests for validation against known results."""

    def test_bfi_accuracy(self):
        """Test that estimated BFI is close to true BFI."""
        # Generate data with known BFI
        target_bfi = 0.65
        Q, true_baseflow, _ = generate_streamflow(
            n_days=365,
            bfi=target_bfi,
            random_seed=42
        )

        dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
        series = pd.Series(Q, index=dates)

        # Separate baseflow
        baseflow_df, _ = single(series, method=["Eckhardt"], return_kge=False)

        # Calculate BFI
        estimated_bfi = baseflow_df["Eckhardt"].sum() / Q.sum()

        # Should be reasonably close (within 10%)
        assert abs(estimated_bfi - target_bfi) < 0.1

    def test_consistency_across_methods(self):
        """Test that different methods give reasonable results."""
        Q, _, _ = generate_streamflow(n_days=365, bfi=0.6, random_seed=42)
        dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
        series = pd.Series(Q, index=dates)

        baseflow_df, _ = single(series, method="all", return_kge=False)

        # Calculate BFI for each method
        bfis = {method: baseflow_df[method].sum() / Q.sum()
                for method in baseflow_df.columns}

        # All methods should give BFI in reasonable range (0.3-0.9)
        for method, bfi in bfis.items():
            assert 0.3 <= bfi <= 0.9, f"{method} gave unrealistic BFI: {bfi}"

    def test_kge_range(self):
        """Test that KGE scores are in valid range."""
        Q, _, _ = generate_streamflow(n_days=365, random_seed=42)
        dates = pd.date_range("2020-01-01", periods=len(Q), freq="D")
        series = pd.Series(Q, index=dates)

        _, kge_scores = single(series, method="all", return_kge=True)

        # KGE can technically be -∞ to 1, but should be reasonable for good data
        for method, kge in kge_scores.items():
            assert kge >= -1, f"{method} gave KGE < -1: {kge}"
            assert kge <= 1, f"{method} gave KGE > 1: {kge}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
