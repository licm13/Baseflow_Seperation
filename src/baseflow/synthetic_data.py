"""Synthetic streamflow data generator for testing and demonstration.

This module provides utilities to generate realistic synthetic streamflow
time series that mimic natural hydrological behavior, including:
- Seasonal baseflow variation
- Storm events (quickflow)
- Exponential recession curves
- Realistic noise and variability

Useful for:
- Testing baseflow separation algorithms
- Demonstrating method capabilities
- Benchmarking and validation
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def generate_baseflow(
    n_days: int = 365,
    base_flow: float = 10.0,
    seasonal_amplitude: float = 5.0,
    recession_coef: float = 0.95,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic baseflow with seasonal variation.

    Creates a realistic baseflow signal with:
    - Sinusoidal seasonal pattern
    - Exponential recession behavior
    - Random fluctuations

    Args:
        n_days: Length of time series in days
        base_flow: Mean baseflow value (e.g., m³/s)
        seasonal_amplitude: Amplitude of seasonal variation
        recession_coef: Recession coefficient (0.9-0.995 typical)
        noise_level: Relative noise standard deviation (0-1)
        random_seed: Random seed for reproducibility

    Returns:
        Baseflow time series array

    Example:
        >>> baseflow = generate_baseflow(365, base_flow=15, seasonal_amplitude=8)
        >>> print(f"Mean: {baseflow.mean():.1f}, Range: {baseflow.min():.1f}-{baseflow.max():.1f}")
        Mean: 15.0, Range: 7.2-22.8
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Seasonal component (sinusoidal)
    time = np.arange(n_days)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * time / 365 - np.pi / 2)

    # Base signal
    baseflow = base_flow + seasonal

    # Add recession behavior (smooth temporal correlation)
    for i in range(1, n_days):
        baseflow[i] = recession_coef * baseflow[i - 1] + (1 - recession_coef) * baseflow[i]

    # Add realistic noise
    noise = np.random.normal(0, noise_level * base_flow, n_days)
    baseflow += noise

    # Ensure positive values
    baseflow = np.maximum(baseflow, 0.1)

    return baseflow


def generate_storm_events(
    n_days: int = 365,
    n_events: int = 20,
    event_intensity: float = 50.0,
    event_duration: int = 3,
    recession_coef: float = 0.90,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic quickflow from storm events.

    Creates realistic storm hydrographs with:
    - Random event timing
    - Variable peak intensities
    - Exponential recession

    Args:
        n_days: Length of time series in days
        n_events: Number of storm events
        event_intensity: Mean storm peak intensity
        event_duration: Typical storm duration in days
        recession_coef: Recession coefficient for quickflow
        random_seed: Random seed for reproducibility

    Returns:
        Quickflow time series array

    Example:
        >>> quickflow = generate_storm_events(365, n_events=15, event_intensity=40)
        >>> n_storm_days = (quickflow > 1).sum()
        >>> print(f"Storm-affected days: {n_storm_days}")
        Storm-affected days: 87
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    quickflow = np.zeros(n_days)

    # Generate random storm events
    event_times = np.random.choice(n_days, size=n_events, replace=False)
    event_times.sort()

    for event_time in event_times:
        # Random peak intensity (log-normal distribution)
        peak = np.random.lognormal(np.log(event_intensity), 0.5)

        # Generate storm hydrograph with exponential recession
        storm_length = min(event_duration + np.random.randint(0, 5), n_days - event_time)

        for i in range(storm_length):
            idx = event_time + i
            if idx < n_days:
                # Rising limb (first part of duration)
                if i < event_duration // 2:
                    intensity = peak * (i + 1) / (event_duration // 2)
                # Peak
                elif i == event_duration // 2:
                    intensity = peak
                # Recession limb
                else:
                    intensity = peak * (recession_coef ** (i - event_duration // 2))

                quickflow[idx] += intensity

    return quickflow


def generate_streamflow(
    n_days: int = 365,
    base_flow: float = 10.0,
    seasonal_amplitude: float = 5.0,
    n_storm_events: int = 20,
    storm_intensity: float = 50.0,
    bfi: float = 0.6,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete synthetic streamflow with known baseflow.

    Creates realistic streamflow by combining baseflow and quickflow
    components. Perfect for testing since true baseflow is known.

    Args:
        n_days: Length of time series in days
        base_flow: Mean baseflow value
        seasonal_amplitude: Seasonal variation amplitude
        n_storm_events: Number of storm events
        storm_intensity: Mean storm peak intensity
        bfi: Target Baseflow Index (baseflow/total flow ratio)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (total_streamflow, true_baseflow, quickflow)

    Example:
        >>> Q, baseflow, quickflow = generate_streamflow(
        ...     n_days=365,
        ...     base_flow=15,
        ...     n_storm_events=25,
        ...     bfi=0.65,
        ...     random_seed=42
        ... )
        >>> print(f"BFI: {baseflow.sum() / Q.sum():.2f}")
        BFI: 0.65
        >>> print(f"Flow range: {Q.min():.1f} - {Q.max():.1f}")
        Flow range: 9.2 - 78.3

    Note:
        - True baseflow is returned, enabling method validation
        - BFI is approximately achieved by scaling components
        - Realistic temporal patterns and variability
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate baseflow component
    baseflow = generate_baseflow(
        n_days=n_days,
        base_flow=base_flow,
        seasonal_amplitude=seasonal_amplitude,
        recession_coef=0.95,
        noise_level=0.05,
        random_seed=random_seed
    )

    # Generate quickflow component
    quickflow = generate_storm_events(
        n_days=n_days,
        n_events=n_storm_events,
        event_intensity=storm_intensity,
        event_duration=3,
        recession_coef=0.85,
        random_seed=random_seed + 1 if random_seed is not None else None
    )

    # Scale to achieve target BFI
    total_baseflow = baseflow.sum()
    total_quickflow = quickflow.sum()
    current_bfi = total_baseflow / (total_baseflow + total_quickflow)

    if current_bfi != 0 and bfi != current_bfi:
        # Adjust quickflow to achieve target BFI
        quickflow *= (total_baseflow * (1 - bfi)) / (bfi * total_quickflow)

    # Total streamflow
    streamflow = baseflow + quickflow

    return streamflow, baseflow, quickflow


def create_test_dataframe(
    n_days: int = 365,
    n_stations: int = 3,
    start_date: str = "2020-01-01",
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create multi-station test dataset with known baseflow.

    Generates realistic synthetic data for multiple stations with
    varying hydrological characteristics.

    Args:
        n_days: Length of time series in days
        n_stations: Number of stations to generate
        start_date: Start date for time series (ISO format)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (streamflow_df, true_baseflow_df, station_info_df)

        - streamflow_df: Total streamflow (index: dates, columns: station IDs)
        - true_baseflow_df: True baseflow for validation
        - station_info_df: Station metadata (area, coordinates, characteristics)

    Example:
        >>> flow, baseflow, info = create_test_dataframe(
        ...     n_days=730,
        ...     n_stations=5,
        ...     random_seed=123
        ... )
        >>> print(flow.shape, info.columns.tolist())
        (730, 5) ['area', 'lon', 'lat', 'base_flow', 'bfi']
        >>> print(f"Station 0 BFI: {info.loc['Station_0', 'bfi']:.2f}")
        Station 0 BFI: 0.55

    Note:
        - Each station has unique hydrological parameters
        - Coordinates span a realistic geographic region
        - True baseflow enables method validation
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create date index
    dates = pd.date_range(start_date, periods=n_days, freq='D')

    # Storage for results
    streamflow_data = {}
    baseflow_data = {}
    station_info = []

    for i in range(n_stations):
        station_id = f"Station_{i}"

        # Random station characteristics
        area = np.random.uniform(500, 5000)  # km²
        lon = np.random.uniform(-125, -110)  # Western North America
        lat = np.random.uniform(35, 50)
        base_flow_mean = np.random.uniform(5, 30)
        seasonal_amp = np.random.uniform(2, 10)
        n_events = np.random.randint(15, 40)
        storm_intensity = np.random.uniform(20, 80)
        target_bfi = np.random.uniform(0.45, 0.75)

        # Generate streamflow
        Q, B, _ = generate_streamflow(
            n_days=n_days,
            base_flow=base_flow_mean,
            seasonal_amplitude=seasonal_amp,
            n_storm_events=n_events,
            storm_intensity=storm_intensity,
            bfi=target_bfi,
            random_seed=random_seed + i if random_seed is not None else None
        )

        streamflow_data[station_id] = Q
        baseflow_data[station_id] = B

        station_info.append({
            'station_id': station_id,
            'area': area,
            'lon': lon,
            'lat': lat,
            'base_flow': base_flow_mean,
            'bfi': target_bfi
        })

    # Create DataFrames
    df_streamflow = pd.DataFrame(streamflow_data, index=dates)
    df_baseflow = pd.DataFrame(baseflow_data, index=dates)
    df_stations = pd.DataFrame(station_info).set_index('station_id')

    return df_streamflow, df_baseflow, df_stations


if __name__ == "__main__":
    # Quick demonstration
    print("Generating synthetic streamflow data...")

    # Single station example
    Q, B, QF = generate_streamflow(
        n_days=365,
        base_flow=15,
        n_storm_events=25,
        bfi=0.65,
        random_seed=42
    )

    print(f"\nSingle Station Statistics:")
    print(f"  Mean flow: {Q.mean():.2f}")
    print(f"  Flow range: {Q.min():.1f} - {Q.max():.1f}")
    print(f"  True BFI: {B.sum() / Q.sum():.3f}")
    print(f"  Number of storm events: ~25")

    # Multi-station example
    flow_df, base_df, info_df = create_test_dataframe(
        n_days=365,
        n_stations=3,
        random_seed=123
    )

    print(f"\nMulti-Station Dataset:")
    print(f"  Shape: {flow_df.shape}")
    print(f"  Date range: {flow_df.index[0]} to {flow_df.index[-1]}")
    print(f"\nStation Information:")
    print(info_df[['area', 'base_flow', 'bfi']].round(2))
