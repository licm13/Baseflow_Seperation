"""Unit tests for synthetic data generation module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow.synthetic_data import (
    create_test_dataframe,
    generate_baseflow,
    generate_storm_events,
    generate_streamflow,
)


class TestGenerateBaseflow:
    """Tests for baseflow generation."""

    def test_basic_generation(self):
        """Test basic baseflow generation."""
        baseflow = generate_baseflow(n_days=365, random_seed=42)
        assert len(baseflow) == 365
        assert np.all(baseflow > 0)
        assert np.all(np.isfinite(baseflow))

    def test_reproducibility(self):
        """Test that random seed produces reproducible results."""
        bf1 = generate_baseflow(n_days=100, random_seed=123)
        bf2 = generate_baseflow(n_days=100, random_seed=123)
        np.testing.assert_array_equal(bf1, bf2)

    def test_seasonal_pattern(self):
        """Test that seasonal variation is present."""
        baseflow = generate_baseflow(
            n_days=365,
            base_flow=10,
            seasonal_amplitude=5,
            random_seed=42
        )
        # Should have variation due to seasonal component
        assert baseflow.std() > 1.0

    def test_mean_approximation(self):
        """Test that mean approximates base_flow parameter."""
        base_flow = 15.0
        baseflow = generate_baseflow(
            n_days=3650,  # 10 years for good averaging
            base_flow=base_flow,
            seasonal_amplitude=3,
            noise_level=0.05,
            random_seed=42
        )
        # Mean should be close to base_flow (within 10%)
        assert abs(baseflow.mean() - base_flow) < base_flow * 0.1


class TestGenerateStormEvents:
    """Tests for storm event generation."""

    def test_basic_generation(self):
        """Test basic storm generation."""
        quickflow = generate_storm_events(n_days=365, n_events=20, random_seed=42)
        assert len(quickflow) == 365
        assert np.all(quickflow >= 0)
        assert np.all(np.isfinite(quickflow))

    def test_event_presence(self):
        """Test that storm events are generated."""
        quickflow = generate_storm_events(
            n_days=365,
            n_events=30,
            event_intensity=50,
            random_seed=42
        )
        # Should have some days with significant quickflow
        assert (quickflow > 10).sum() > 20

    def test_zero_events(self):
        """Test handling of zero events."""
        quickflow = generate_storm_events(n_days=100, n_events=0, random_seed=42)
        np.testing.assert_array_equal(quickflow, np.zeros(100))


class TestGenerateStreamflow:
    """Tests for complete streamflow generation."""

    def test_basic_generation(self):
        """Test basic streamflow generation."""
        Q, baseflow, quickflow = generate_streamflow(n_days=365, random_seed=42)
        assert len(Q) == len(baseflow) == len(quickflow) == 365
        assert np.all(Q >= 0)
        assert np.all(baseflow >= 0)
        assert np.all(quickflow >= 0)

    def test_components_sum(self):
        """Test that Q = baseflow + quickflow."""
        Q, baseflow, quickflow = generate_streamflow(n_days=365, random_seed=42)
        np.testing.assert_allclose(Q, baseflow + quickflow, rtol=1e-10)

    def test_bfi_target(self):
        """Test that BFI approximately matches target."""
        target_bfi = 0.65
        Q, baseflow, _ = generate_streamflow(
            n_days=365,
            bfi=target_bfi,
            random_seed=42
        )
        actual_bfi = baseflow.sum() / Q.sum()
        # Should be very close (within 1%)
        assert abs(actual_bfi - target_bfi) < 0.01

    def test_different_bfi_values(self):
        """Test generation with different BFI targets."""
        for target_bfi in [0.3, 0.5, 0.7, 0.9]:
            Q, baseflow, _ = generate_streamflow(
                n_days=365,
                bfi=target_bfi,
                random_seed=42
            )
            actual_bfi = baseflow.sum() / Q.sum()
            assert abs(actual_bfi - target_bfi) < 0.02


class TestCreateTestDataframe:
    """Tests for multi-station dataframe creation."""

    def test_basic_creation(self):
        """Test basic dataframe creation."""
        flow_df, base_df, info_df = create_test_dataframe(
            n_days=100,
            n_stations=3,
            random_seed=42
        )

        assert flow_df.shape == (100, 3)
        assert base_df.shape == (100, 3)
        assert len(info_df) == 3

    def test_date_index(self):
        """Test that date index is correct."""
        start_date = "2020-01-01"
        flow_df, _, _ = create_test_dataframe(
            n_days=365,
            n_stations=2,
            start_date=start_date,
            random_seed=42
        )

        assert isinstance(flow_df.index, pd.DatetimeIndex)
        assert flow_df.index[0] == pd.Timestamp(start_date)
        assert len(flow_df.index) == 365

    def test_station_info_columns(self):
        """Test that station info has expected columns."""
        _, _, info_df = create_test_dataframe(n_days=100, n_stations=2, random_seed=42)

        expected_cols = ['area', 'lon', 'lat', 'base_flow', 'bfi']
        for col in expected_cols:
            assert col in info_df.columns

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        flow1, base1, info1 = create_test_dataframe(
            n_days=100,
            n_stations=2,
            random_seed=99
        )
        flow2, base2, info2 = create_test_dataframe(
            n_days=100,
            n_stations=2,
            random_seed=99
        )

        pd.testing.assert_frame_equal(flow1, flow2)
        pd.testing.assert_frame_equal(base1, base2)
        pd.testing.assert_frame_equal(info1, info2)

    def test_unique_stations(self):
        """Test that each station has unique characteristics."""
        flow_df, _, info_df = create_test_dataframe(
            n_days=100,
            n_stations=5,
            random_seed=42
        )

        # Each station should have different metadata
        assert len(info_df['area'].unique()) == 5
        assert len(info_df['lon'].unique()) == 5
        assert len(info_df['lat'].unique()) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
