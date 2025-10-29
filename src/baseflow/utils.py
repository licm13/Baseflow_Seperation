"""Utility helpers for data cleaning, masking and numerical routines.

This module provides general-purpose utility functions for:
- Streamflow data cleaning and validation
- Frozen period detection and masking
- Numerical operations (moving averages, range generation)
- Geographic coordinate transformations
- Method name formatting and validation
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

__all__ = [
    "clean_streamflow",
    "exist_ice",
    "moving_average",
    "multi_arange",
    "geo2imagexy",
    "format_method",
    "validate_streamflow",
    "calculate_bfi",
    "get_flow_statistics",
]


def clean_streamflow(
    series: pd.Series,
) -> Tuple[pd.DatetimeIndex, npt.NDArray[np.float64]]:
    """Clean and validate streamflow time series data.

    Removes missing values (NaN, Inf) and converts negative flows to positive.

    Args:
        series: Pandas Series with DatetimeIndex and streamflow values

    Returns:
        Tuple of (cleaned dates, cleaned flow values)

    Note:
        - Negative values are converted to absolute values
        - Missing/infinite values are removed
        - Returned arrays have matching lengths
    """
    date, Q = series.index, series.values.astype(float)
    has_value = np.isfinite(Q)
    date, Q = date[has_value], np.abs(Q[has_value])
    return date, Q


def exist_ice(
    date: Optional[pd.DatetimeIndex],
    ice_period: Optional[Union[npt.NDArray[np.bool_], Tuple[List[int], List[int]]]]
) -> Optional[npt.NDArray[np.bool_]]:
    """Create boolean mask for frozen (ice-covered) periods.

    Generates a mask identifying dates during frozen periods when the
    groundwater-baseflow relationship may be invalidated by ice cover.

    Args:
        date: DatetimeIndex for the time series
        ice_period: Frozen period specification. Can be:
            - Boolean array of length 12 (True for frozen months)
            - Tuple: [(start_month, start_day), (end_month, end_day)]
            - None: Returns None (no masking)

    Returns:
        Boolean array (True = frozen) matching date length, or None

    Examples:
        >>> # Northern hemisphere winter: Nov 1 to Mar 31
        >>> ice = exist_ice(dates, ([11, 1], [3, 31]))
        >>>
        >>> # Monthly mask (Dec, Jan, Feb frozen)
        >>> months = np.array([False]*11 + [True] + [True]*2 + [False]*9)
        >>> ice = exist_ice(dates, months)

    Note:
        - Handles cross-year periods (e.g., Nov-Mar spans calendar year boundary)
        - Used to exclude frozen periods from recession analysis
    """
    if (date is None) or (ice_period is None):
        return None

    # Case 1: Monthly boolean mask
    if isinstance(ice_period, np.ndarray):
        return np.isin(date.month, np.where(ice_period)[0] + 1)

    # Case 2: Date range specification
    beg, end = ice_period

    # Check if period crosses year boundary
    if (end[0] > beg[0]) or ((end[0] == beg[0]) & (end[1] > beg[1])):
        # Within-year period (e.g., May to September)
        ice = (
            ((date.month > beg[0]) & (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    else:
        # Cross-year period (e.g., November to March)
        ice = (
            ((date.month > beg[0]) | (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    return ice


def moving_average(
    x: npt.NDArray[np.float64],
    w: int
) -> npt.NDArray[np.float64]:
    """Calculate moving average using convolution.

    Applies a simple moving average filter with equal weights.

    Args:
        x: Input array
        w: Window width (number of points to average)

    Returns:
        Smoothed array (length = len(x) - 2*(w-1))

    Note:
        - Uses np.convolve for efficient computation
        - Edge handling: Returns only fully-overlapping windows
        - Output is shorter than input by 2*(w-1) points
    """
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1 : -w + 1]


@njit
def multi_arange(
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Efficiently concatenate multiple np.arange() outputs.

    Numba-compiled function to create an array containing concatenated
    ranges [starts[i], stops[i]) for all i.

    Args:
        starts: Array of range start values (inclusive)
        stops: Array of range stop values (exclusive)

    Returns:
        Concatenated array of all ranges

    Example:
        >>> multi_arange(np.array([0, 5, 10]), np.array([3, 8, 12]))
        array([0, 1, 2, 5, 6, 7, 10, 11])

    Note:
        - JIT-compiled with Numba for performance
        - Much faster than using np.concatenate([np.arange(s, e) for s, e in ...])
        - Used in recession period identification
    """
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)

    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos : pos + num] = np.arange(starts[i], stops[i])
        pos += num

    return res


def geo2imagexy(
    x: float,
    y: float
) -> Tuple[int, int]:
    """Convert geographic coordinates to raster image indices.

    Transforms lon/lat coordinates to col/row indices for the included
    permafrost data raster (0.5° resolution, global extent).

    Args:
        x: Longitude in decimal degrees (-180 to 180)
        y: Latitude in decimal degrees (-90 to 90)

    Returns:
        Tuple of (column, row) indices for raster access

    Example:
        >>> col, row = geo2imagexy(-120.5, 45.2)
        >>> ice_data = thawed_mask[:, row, col]

    Note:
        - Assumes 0.5° grid resolution
        - Origin at (-180, 90) in lon/lat space
        - Used for extracting frozen period data from global permafrost raster
    """
    # Affine transformation matrix for 0.5° grid
    a = np.array([[0.5, 0.0], [0.0, -0.5]])
    b = np.array([x - -180, y - 90])
    col, row = np.linalg.solve(a, b) - 0.5
    return np.round(col).astype(int), np.round(row).astype(int)


def format_method(method: Union[str, List[str]]) -> List[str]:
    """Normalize method name specification to list format.

    Converts various method specifications into a consistent list format
    for internal processing.

    Args:
        method: Method specification. Can be:
            - "all": Expands to all 12 available methods
            - Single method name string (e.g., "LH")
            - List of method names (returned as-is)

    Returns:
        List of method names

    Example:
        >>> format_method("all")
        ['UKIH', 'Local', 'Fixed', ..., 'Willems']
        >>> format_method("Eckhardt")
        ['Eckhardt']
        >>> format_method(["LH", "Chapman"])
        ['LH', 'Chapman']

    Available methods:
        - Graphical: UKIH, Local, Fixed, Slide
        - Digital filters: LH, Chapman, CM, Eckhardt, EWMA, Willems
        - Parameterized: Boughton, Furey

    Note:
        No validation is performed on method names.
        Invalid names will cause errors in separation functions.
    """
    if method == "all":
        method = [
            "UKIH",
            "Local",
            "Fixed",
            "Slide",
            "LH",
            "Chapman",
            "CM",
            "Boughton",
            "Furey",
            "Eckhardt",
            "EWMA",
            "Willems",
        ]
    elif isinstance(method, str):
        method = [method]
    return method


def validate_streamflow(
    series: pd.Series,
    min_length: int = 365,
    max_gap_days: int = 30
) -> Tuple[bool, List[str]]:
    """Validate streamflow data quality and completeness.

    Checks for common data quality issues that may affect baseflow separation.

    Args:
        series: Streamflow time series
        min_length: Minimum required length in days
        max_gap_days: Maximum allowed gap between consecutive observations

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> is_valid, issues = validate_streamflow(flow_series)
        >>> if not is_valid:
        ...     print("Data issues:", issues)

    Note:
        - Returns (True, []) for valid data
        - Returns (False, [issue1, issue2, ...]) for invalid data
    """
    issues = []

    # Check length
    if len(series) < min_length:
        issues.append(f"Time series too short: {len(series)} days (minimum: {min_length})")

    # Check for missing values
    n_missing = series.isna().sum()
    if n_missing > 0:
        pct_missing = n_missing / len(series) * 100
        issues.append(f"Missing values: {n_missing} ({pct_missing:.1f}%)")

    # Check for negative values
    if (series < 0).any():
        n_negative = (series < 0).sum()
        issues.append(f"Negative values: {n_negative} (will be converted to absolute)")

    # Check for zero/very low flow dominance
    n_zero = (series == 0).sum()
    if n_zero > len(series) * 0.3:
        issues.append(f"Warning: {n_zero} zero values ({n_zero/len(series)*100:.1f}%)")

    # Check for unrealistic values (very large spikes)
    if len(series) > 0 and series.max() > 0:
        ratio = series.max() / series.median()
        if ratio > 100:
            issues.append(f"Warning: Large flow range (max/median = {ratio:.1f})")

    # Check for time gaps (if DatetimeIndex)
    if isinstance(series.index, pd.DatetimeIndex):
        time_diffs = series.index.to_series().diff()[1:]
        max_gap = time_diffs.max()
        if max_gap > pd.Timedelta(days=max_gap_days):
            issues.append(f"Large time gap: {max_gap.days} days (max allowed: {max_gap_days})")

    is_valid = len(issues) == 0
    return is_valid, issues


def calculate_bfi(
    streamflow: npt.NDArray[np.float64],
    baseflow: npt.NDArray[np.float64]
) -> float:
    """Calculate Baseflow Index (BFI).

    BFI is the ratio of baseflow volume to total streamflow volume.

    Args:
        streamflow: Total streamflow array
        baseflow: Baseflow array

    Returns:
        BFI value (0 to 1)

    Example:
        >>> bfi = calculate_bfi(Q, baseflow)
        >>> print(f"Baseflow Index: {bfi:.3f}")

    Note:
        - BFI = sum(baseflow) / sum(streamflow)
        - Returns 0 if total flow is zero
        - Typical ranges: 0.3-0.5 (flashy), 0.5-0.8 (perennial)
    """
    total_flow = np.sum(np.abs(streamflow))
    if total_flow == 0:
        return 0.0
    return np.sum(baseflow) / total_flow


def get_flow_statistics(
    streamflow: Union[pd.Series, npt.NDArray[np.float64]]
) -> dict:
    """Calculate comprehensive streamflow statistics.

    Computes descriptive statistics useful for characterizing flow regime.

    Args:
        streamflow: Streamflow time series or array

    Returns:
        Dictionary with statistics:
            - mean, median, std, min, max
            - percentiles (p10, p25, p75, p90)
            - cv (coefficient of variation)
            - n_days (length)

    Example:
        >>> stats = get_flow_statistics(flow_series)
        >>> print(f"Mean: {stats['mean']:.2f} m³/s")
        >>> print(f"CV: {stats['cv']:.2f}")

    Note:
        - Useful for quality control and regime classification
        - CV > 1.0 indicates highly variable flow
    """
    if isinstance(streamflow, pd.Series):
        Q = streamflow.values
    else:
        Q = streamflow

    # Remove NaN and infinite values
    Q = Q[np.isfinite(Q)]

    if len(Q) == 0:
        return {
            'n_days': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'cv': np.nan,
            'p10': np.nan,
            'p25': np.nan,
            'p75': np.nan,
            'p90': np.nan,
        }

    stats = {
        'n_days': len(Q),
        'mean': np.mean(Q),
        'median': np.median(Q),
        'std': np.std(Q),
        'min': np.min(Q),
        'max': np.max(Q),
        'p10': np.percentile(Q, 10),
        'p25': np.percentile(Q, 25),
        'p75': np.percentile(Q, 75),
        'p90': np.percentile(Q, 90),
    }

    # Coefficient of variation
    if stats['mean'] > 0:
        stats['cv'] = stats['std'] / stats['mean']
    else:
        stats['cv'] = np.nan

    return stats
